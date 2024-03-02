import warnings
from collections import defaultdict
from typing import Dict, Final, Iterable, List, Optional, Union
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from math import sqrt

from config import Csv, config
from model import ModelParams
from modules import LocalSnrTarget, erb_fb
from stoi import stoi
from utils import angle, as_complex, get_device
from libdf import DF
from DeepFilterNet.df.git.deepfilternet2 import DfNet

import torch_pruning as tp
from prune_utils import GLinearPruner, L2NormImportance
from modules import GroupedLinear, Conv2dNormAct, ConvTranspose2dNormAct


def wg(S: Tensor, X: Tensor, eps: float = 1e-10) -> Tensor:
    N = X - S
    SS = as_complex(S).abs().square()
    NN = as_complex(N).abs().square()
    return (SS / (SS + NN + eps)).clamp(0, 1)


def irm(S: Tensor, X: Tensor, eps: float = 1e-10) -> Tensor:
    N = X - S
    SS_mag = as_complex(S).abs()
    NN_mag = as_complex(N).abs()
    return (SS_mag / (SS_mag + NN_mag + eps)).clamp(0, 1)


def iam(S: Tensor, X: Tensor, eps: float = 1e-10) -> Tensor:
    SS_mag = as_complex(S).abs()
    XX_mag = as_complex(X).abs()
    return (SS_mag / (XX_mag + eps)).clamp(0, 1)


class Stft(nn.Module):
    def __init__(self, n_fft: int, hop: Optional[int] = None, window: Optional[Tensor] = None):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop or n_fft // 4
        if window is not None:
            assert window.shape[0] == n_fft
        else:
            window = torch.hann_window(self.n_fft)
        self.w: torch.Tensor
        self.register_buffer("w", window)

    def forward(self, input: Tensor):
        # Time-domain input shape: [B, *, T]
        t = input.shape[-1]
        sh = input.shape[:-1]
        out = torch.stft(
            input.reshape(-1, t),
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.w,
            normalized=True,
            return_complex=True,
        )
        out = out.view(*sh, *out.shape[-2:])
        return out


class Istft(nn.Module):
    def __init__(self, n_fft_inv: int, hop_inv: int, window_inv: Tensor):
        super().__init__()
        # Synthesis back to time domain
        self.n_fft_inv = n_fft_inv
        self.hop_inv = hop_inv
        self.w_inv: torch.Tensor
        assert window_inv.shape[0] == n_fft_inv
        self.register_buffer("w_inv", window_inv)

    def forward(self, input: Tensor):
        # Input shape: [B, * T, F, (2)]
        input = as_complex(input)
        t, f = input.shape[-2:]
        sh = input.shape[:-2]
        # Even though this is not the DF implementation, it numerical sufficiently close.
        # Pad one extra step at the end to get original signal length
        out = torch.istft(
            F.pad(input.reshape(-1, t, f).transpose(1, 2), (0, 1)),
            n_fft=self.n_fft_inv,
            hop_length=self.hop_inv,
            window=self.w_inv,
            normalized=True,
        )
        if input.ndim > 2:
            out = out.view(*sh, out.shape[-1])
        return out


class MultiResSpecLoss(nn.Module):
    gamma: Final[float]
    f: Final[float]
    f_complex: Final[Optional[List[float]]]

    def __init__(
        self,
        n_ffts: Iterable[int],
        gamma: float = 1,
        factor: float = 1,
        f_complex: Optional[Union[float, Iterable[float]]] = None,
    ):
        super().__init__()
        self.gamma = gamma
        self.f = factor
        self.stfts = nn.ModuleDict({str(n_fft): Stft(n_fft) for n_fft in n_ffts})
        if f_complex is None or f_complex == 0:
            self.f_complex = None
        elif isinstance(f_complex, Iterable):
            self.f_complex = list(f_complex)
        else:
            self.f_complex = [f_complex] * len(self.stfts)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss = torch.zeros((), device=input.device, dtype=input.dtype)
        for i, stft in enumerate(self.stfts.values()):
            Y = stft(input)
            S = stft(target)
            Y_abs = Y.abs()
            S_abs = S.abs()
            if self.gamma != 1:
                Y_abs = Y_abs.clamp_min(1e-12).pow(self.gamma)
                S_abs = S_abs.clamp_min(1e-12).pow(self.gamma)
            loss += F.mse_loss(Y_abs, S_abs) * self.f
            if self.f_complex is not None:
                if self.gamma != 1:
                    Y = Y_abs * torch.exp(1j * angle.apply(Y))
                    S = S_abs * torch.exp(1j * angle.apply(S))
                loss += F.mse_loss(torch.view_as_real(Y), torch.view_as_real(S)) * self.f_complex[i]
        return loss

    def change_weight(self, new_f):
        self.factor = new_f
        

class SpectralLoss(nn.Module):
    gamma: Final[float]
    f_m: Final[float]
    f_c: Final[float]
    f_u: Final[float]

    def __init__(
        self,
        gamma: float = 1,
        factor_magnitude: float = 1,
        factor_complex: float = 1,
        factor_under: float = 1,
    ):
        super().__init__()
        self.gamma = gamma
        self.f_m = factor_magnitude
        self.f_c = factor_complex
        self.f_u = factor_under

    def forward(self, input, target):
        input = as_complex(input)
        target = as_complex(target)
        input_abs = input.abs()
        target_abs = target.abs()
        if self.gamma != 1:
            input_abs = input_abs.clamp_min(1e-12).pow(self.gamma)
            target_abs = target_abs.clamp_min(1e-12).pow(self.gamma)
        tmp = (input_abs - target_abs).pow(2)
        if self.f_u != 1:
            # Weighting if predicted abs is too low
            tmp *= torch.where(input_abs < target_abs, self.f_u, 1.0)
        loss = torch.mean(tmp) * self.f_m
        if self.f_c > 0:
            if self.gamma != 1:
                input = input_abs * torch.exp(1j * angle.apply(input))
                target = target_abs * torch.exp(1j * angle.apply(target))
            loss_c = F.mse_loss(torch.view_as_real(input), target=torch.view_as_real(target)) * self.f_c
            loss = loss + loss_c
        return loss

    def change_weight(self, new_f):
        self.factor = new_f
                           

class L1SparsityLoss(nn.Module):
    factor: Final[float]
    model: DfNet
    not_spar: tuple

    def __init__(
        self,
        factor: float = 1,
        model: DfNet = None,
        not_spar: tuple = ("conv", "bias")
    ):
        super().__init__()
        self.factor = factor
        self.model = model
        self.not_spar = not_spar
    
    def forward(self):
        weight_sum = torch.tensor(0.0, requires_grad=True)
        weight_num = torch.tensor(0.0, requires_grad=True)
        for name, para in self.model.named_parameters():
            if not any(key in name for key in self.not_spar):
                weight_sum = weight_sum + torch.sum(torch.abs(para))
                weight_num = weight_num + torch.count_nonzero(para)
        loss = weight_sum / weight_num * self.factor

        return loss

    def change_weight(self, new_f):
        self.factor = new_f
    

class L2SparsityLoss(nn.Module):
    factor: Final[float]
    model: DfNet
    not_spar: tuple

    def __init__(
        self,
        factor: float = 1,
        model: DfNet = None,
        not_spar: tuple = ("conv", "bias")
    ):
        super().__init__()
        self.factor = factor
        self.model = model
        self.not_spar = not_spar
        self.g_num = self.cal_g_num()  # constant
    
    def cal_g_num(self):
        g_num = 0
        for name, para in self.model.named_parameters():
            # omit BN and not_spar para
            if len(para.shape) > 1 and not any(key in name for key in self.not_spar):
                g_num += para.shape[-1]
        return g_num

    def forward(self):
        l2_loss = torch.tensor(0.0, requires_grad=True)
        for name, para in self.model.named_parameters():
            # omit BN(dim=1) and not_spar para
            if len(para.shape) > 1 and not any(key in name for key in self.not_spar):
                l2_loss = l2_loss + torch.sum(sqrt(para.shape[-2]) * torch.norm(para, p=2, dim=-2))
        loss = l2_loss / self.g_num * self.factor
        return loss

    def change_weight(self, new_f):
        self.factor = new_f


class GroupedL2SparsityLoss(nn.Module):
    factor: Final[float]
    model: DfNet
    not_spar: tuple

    def __init__(
        self,
        factor: float = 1,
        model: DfNet = None,
    ):
        super().__init__()
        p = ModelParams()
        self.factor = factor
        self.model = model
        self.L2norm = L2NormImportance()
        self.g_list = get_DG_groups(p, model)

    def forward(self):
        gl2_loss = torch.tensor(0.0, requires_grad=True)
        for g in self.g_list:
            gl2_imp, pg = self.L2norm(g)
            gl2_loss = gl2_loss + sqrt(pg) * gl2_imp.sum(0)
        loss = gl2_loss / len(self.g_list) * self.factor
        return loss

    def change_weight(self, new_f):
        self.factor = new_f
    
    def change_g_list(self, p, model):
        self.g_list = get_DG_groups(p, model)


class Loss(nn.Module):
    """Loss wrapper containing several different loss functions within this file.

    The configuration is done via the config file.
    """
    def __init__(
            self, 
            state: DF, 
            istft: Optional[Istft] = None, 
            model: DfNet = None, 
            sl_f: float = 1, mrsl_f: float = 1,
            s1l_f: float = 1, s2l_f: float = 1, gs2l_f: float = 1
            ):
        """Loss wrapper containing all methods for loss calculation.

        Args:
            state (DF): DF state needed for MaskLoss.
            istft (Callable/Module): Istft method needed for time domain losses.
            model: needed for L1/2 Penalty Loss.
        """
        super().__init__()
        p = ModelParams()
        self.model = model
        self.lsnr = LocalSnrTarget(ws=20, target_snr_range=[p.lsnr_min - 5, p.lsnr_max + 5])
        self.istft = istft  # Could also be used for sdr loss
        self.sr = p.sr
        self.fft_size = p.fft_size
        self.nb_df = p.nb_df
        self.store_losses = False
        self.summaries: Dict[str, List[Tensor]] = self.reset_summaries()

        # SpectralLoss, fundamental
        self.sl_fm = config("factor_magnitude", sl_f, float, section="SpectralLoss")  # e.g. 1e4
        self.sl_fc = config("factor_complex", sl_f, float, section="SpectralLoss")
        self.sl_fu = config("factor_under", 1, float, section="SpectralLoss")
        self.sl_gamma = config("gamma", 0.6, float, section="SpectralLoss")
        self.sl_f = self.sl_fm + self.sl_fc
        if self.sl_f > 0:
            self.sl = SpectralLoss(
                factor_magnitude=self.sl_fm,
                factor_complex=self.sl_fc,
                factor_under=self.sl_fu,
                gamma=self.sl_gamma,
            )
        else:
            self.sl = None

        # Multi Resolution Spectrogram Loss, fundamental
        self.mrsl_f = config("factor", mrsl_f, float, section="MultiResSpecLoss")
        self.mrsl_fc = config("factor_complex", mrsl_f, float, section="MultiResSpecLoss")
        self.mrsl_gamma = config("gamma", 0.3, float, section="MultiResSpecLoss")
        if self.sr == 48000:  # {5, 10, 20, 40}
            self.mrsl_ffts: List[int] = config("fft_sizes", [240, 480, 960, 1920], Csv(int), section="MultiResSpecLoss")
        else:  # sr == 16000
            self.mrsl_ffts: List[int] = config("fft_sizes", [80, 160, 320, 640], Csv(int), section="MultiResSpecLoss")
        if self.mrsl_f > 0:
            assert istft is not None
            self.mrsl = MultiResSpecLoss(self.mrsl_ffts, self.mrsl_gamma, self.mrsl_f, self.mrsl_fc)
        else:
            self.mrsl = None

        # L1 Penalty Loss, pruning must
        if s1l_f == 0:
            self.s1l_f = config("factor", 0, float, section="L1SparsityLoss")
        else:
            self.s1l_f = config("factor", s1l_f, float, section="L1SparsityLoss")
        if self.s1l_f > 0:
            self.s1l = L1SparsityLoss(self.s1l_f, self.model)
        else:
            self.s1l = None

        # L2 Penalty Loss, pruning must
        if s2l_f == 0:
            self.s2l_f = config("factor", 0, float, section="L2SparsityLoss")
        else:
            self.s2l_f = config("factor", s2l_f, float, section="L2SparsityLoss")
        if self.s2l_f > 0:
            self.s2l = L2SparsityLoss(self.s2l_f, self.model)
        else:
            self.s2l = None

        # Grouped L2 Penalty Loss, for comparing experiments
        if gs2l_f == 0:
            self.gs2l_f = config("factor", 0, float, section="GroupedL2SparsityLoss")
        else:
            self.gs2l_f = config("factor", gs2l_f, float, section="GroupedL2SparsityLoss")
        if self.gs2l_f > 0:
            self.gs2l = GroupedL2SparsityLoss(self.gs2l_f, self.model)
        else:
            self.gs2l = None

        self.dev_str = get_device().type

    def forward(
        self,
        clean: Tensor,
        noisy: Tensor,
        enhanced: Tensor,
        mask: Tensor,
        snrs: Tensor,
        max_freq: Optional[Tensor] = None,
    ):
        """Computes all losses.

        Args:
            clean (Tensor): Clean complex spectrum of shape [B, C, T, F].
            noisy (Tensor): Noisy complex spectrum of shape [B, C, T, F].
            enhanced (Tensor): Enhanced complex spectrum of shape [B, C, T, F].
            mask (Tensor): Mask (real-valued) estimate of shape [B, C, T, E], E: Number of ERB bins.
            lsnr (Tensor): Local SNR estimates of shape [B, T, 1].
            snrs (Tensor): Input SNRs of the noisy mixture of shape [B].
            max_freq (Optional, Tensor): Maximum frequency present in the noisy mixture (e.g. due to a lower sampling rate) of shape [B].
        """
        max_bin: Optional[Tensor] = None
        if max_freq is not None:
            max_bin = (
                max_freq.to(device=clean.device)
                .mul(self.fft_size)
                .div(self.sr, rounding_mode="trunc")
            ).long()
        enhanced_td = None
        clean_td = None
        if self.istft is not None:
            if self.store_losses or self.mrsl is not None:
                enhanced_td = self.istft(enhanced)
                clean_td = self.istft(clean)

        ml, sl, mrsl, osl, s1l, s2l, gs2l = [torch.zeros((), device=clean.device)] * 7

        if self.ml_f != 0 and self.ml is not None:
            ml = self.ml(input=mask, clean=clean, noisy=noisy, max_bin=max_bin)

        if self.sl_f != 0 and self.sl is not None:
            sl = torch.zeros((), device=clean.device)
            sl = self.sl(input=enhanced, target=clean)

        if self.mrsl_f > 0 and self.mrsl is not None:
            mrsl = self.mrsl(enhanced_td, clean_td)

        if self.osl_f != 0:
            osl = self.osl(input=enhanced, target=clean)
        
        if self.s1l_f != 0:
            s1l = self.s1l()
        
        if self.s2l_f != 0:
            s2l = self.s2l()
        
        if self.gs2l_f != 0:
            gs2l = self.gs2l()

        if self.store_losses and enhanced_td is not None:
            assert clean_td is not None
            self.store_summaries(
                enh_td=enhanced_td,
                clean_td=clean_td,
                snrs=snrs,  # snr range list
                ml=ml,
                sl=sl,
                mrsl=mrsl,
                osl=osl,
                s1l=s1l,
                s2l=s2l,
                gs2l=gs2l
            )
        return ml + sl + mrsl + osl + s1l + s2l + gs2l

    def change_sparsity_weight(self, n_s1l_f, n_s2l_f, n_gs2l_f):
        # only change factor in sl, 
        # save self.sl_f as the original factor(constant)
        if self.s1l_f > 0:
            self.s1l.change_weight(n_s1l_f)
        if self.s2l_f > 0:
            self.s2l.change_weight(n_s2l_f)
        if self.gs2l_f > 0:
            self.gs2l.change_weight(n_gs2l_f)

    def change_gL2_DG(self, p, model):
        self.gs2l.change_g_list(p, model)

    def reset_summaries(self):
        self.summaries = defaultdict(list)
        return self.summaries

    @torch.jit.ignore  # type: ignore
    def get_summaries(self):
        return self.summaries.items()

    @torch.no_grad()
    @torch.jit.ignore  # type: ignore
    def store_summaries(
        self,
        enh_td: Tensor,
        clean_td: Tensor,
        snrs: Tensor,
        ml: Tensor,
        sl: Tensor,
        mrsl: Tensor,
        osl: Tensor, 
        s1l: Tensor,
        s2l: Tensor,
        gs2l: Tensor,
    ):
        if ml != 0:
            self.summaries["MaskLoss"].append(ml.detach())
        if sl != 0:
            self.summaries["SpectralLoss"].append(sl.detach())
        if mrsl != 0:
            self.summaries["MultiResSpecLoss"].append(mrsl.detach())
        if osl != 0:
            self.summaries["OverSuppersionLoss"].append(osl.detach())
        if s1l != 0:
            self.summaries["L1SparsityLoss"].append(s1l.detach())
        if s2l != 0:
            self.summaries["L2SparsityLoss"].append(s2l.detach())
        if gs2l != 0:
            self.summaries["GroupedL2SparsityLoss"].append(gs2l.detach())
        enh_td = enh_td.squeeze(1).detach()
        clean_td = clean_td.squeeze(1).detach()
        stoi_vals: Tensor = stoi(y=enh_td, x=clean_td, fs_source=self.sr)
        stoi_vals_ms = []
        for snr in torch.unique(snrs, sorted=False):
            self.summaries[f"stoi_snr_{snr.item()}"].extend(
                stoi_vals.masked_select(snr == snrs).detach().split(1)
            )
            for i, stoi_i in enumerate(stoi_vals_ms):
                self.summaries[f"stoi_stage_{i}_snr_{snr.item()}"].extend(
                    stoi_i.masked_select(snr == snrs).detach().split(1)
                )


def get_DG_groups(p, model):

    DG = tp.DependencyGraph()
    GLPruner = GLinearPruner(torch.device("cuda"))
    DG.register_customized_layer(GroupedLinear, GLPruner)

    model_inputs = {
        "spec": torch.randn((1, 1, 10, 481, 2), device=torch.device("cuda")),
        "feat_erb": torch.randn((1, 1, 10, 32), device=torch.device("cuda")),
        "feat_spec": torch.randn((1, 1, 10, 96, 2), device=torch.device("cuda"))
    }

    # ignore Conv and output layers for pruning
    skipped_layers = ['enc.lsnr_fc.0', 'df_dec.df_out.0', 'erb_dec.emb_gru_out']
    for l in range(p.fix_lin_groups):
        skipped_layers.append('df_dec.df_out.0.layers.' + str(l))
        # skipped_layers.append('enc.erb_fc_emb.layers.' + str(l))
        skipped_layers.append('erb_dec.emb_gru_out.layers.' + str(l))
    ignored_layers = []
    for idx, (name, m) in enumerate(model.named_modules()):
        if name in skipped_layers or isinstance(m, (Conv2dNormAct, ConvTranspose2dNormAct)):
            ignored_layers.append(m)
    root_module = [nn.Linear, nn.GRU]

    DG.build_dependency(model, example_inputs=model_inputs)
    g_list = []
    
    for g in DG.get_all_groups(ignored_layers=ignored_layers, root_module_types=root_module):
        g_list.append(g)

    return g_list