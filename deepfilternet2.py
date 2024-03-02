"""
this model is modified for compatibility with onnx and torch.jit
"""
from functools import partial
from typing import List, Optional, Tuple

import torch
from loguru import logger
from torch import Tensor, nn

import multiframe as MF
from config import Csv, DfParams, config
from modules import (
    Conv2dNormAct,
    ConvTranspose2dNormAct,
    oneGRU,
    doubleGRU,
    GroupedLinear,
    Mask,
    erb_fb,
    get_device,
)
from libdf import DF


class ModelParams(DfParams):
    section = "deepfilternet"

    def __init__(self):
        super().__init__(sr=48000)
        self.conv_lookahead: int = config(
            "CONV_LOOKAHEAD", cast=int, default=0, section=self.section
        )
        self.conv_ch: int = config("CONV_CH", cast=int, default=16, section=self.section)
        self.conv_depthwise: bool = config(
            "CONV_DEPTHWISE", cast=bool, default=True, section=self.section
        )
        self.convt_depthwise: bool = config(
            "CONVT_DEPTHWISE", cast=bool, default=True, section=self.section
        )
        self.conv_kernel: List[int] = config(
            "CONV_KERNEL", cast=Csv(int), default=(1, 3), section=self.section  # type: ignore
        )
        self.conv_kernel_inp: List[int] = config(
            "CONV_KERNEL_INP", cast=Csv(int), default=(3, 3), section=self.section  # type: ignore
        )
        self.emb_hidden_dim: int = config(
            "EMB_HIDDEN_DIM", cast=int, default=256, section=self.section
        )
        self.emb_num_layers: int = config(
            "EMB_NUM_LAYERS", cast=int, default=2, section=self.section
        )
        self.df_hidden_dim: int = config(
            "DF_HIDDEN_DIM", cast=int, default=256, section=self.section
        )
        self.df_pathway_kernel_size_t: int = config("DF_PATHWAY_KERNEL_SIZE_T", cast=int, default=1, section=self.section)
        self.df_num_layers: int = config("DF_NUM_LAYERS", cast=int, default=2, section=self.section)
        self.pr_lin_groups: int = config("PR_LINEAR_GROUPS", cast=int, default=2, section=self.section)
        self.fix_lin_groups: int = config("FIX_LINEAR_GROUPS", cast=int, default=8, section=self.section)
        self.group_shuffle: bool = config("GROUP_SHUFFLE", cast=bool, default=True, section=self.section)
        self.dfop_method: str = config("DFOP_METHOD", cast=str, default="real_loop", section=self.section)  # all types in modules.set_forward
        self.mask_pf: bool = config("MASK_PF", cast=bool, default=True, section=self.section)
        self.enc_concat: bool = config("ENC_CONCAT", cast=bool, default=True, section=self.section)


def init_model(df_state: Optional[DF] = None, run_df: bool = True, train_mask: bool = True):
    p = ModelParams()
    if df_state is None:
        df_state = DF(sr=p.sr, fft_size=p.fft_size, hop_size=p.hop_size, nb_bands=p.nb_erb)
    erb = erb_fb(df_state.erb_widths(), p.sr, inverse=False)
    erb_inverse = erb_fb(df_state.erb_widths(), p.sr, inverse=True)
    model = DfNet(erb, erb_inverse)
    return model.to(device=get_device())


class Add(nn.Module):
    def forward(self, a, b):
        return a + b


class Concat(nn.Module):
    def forward(self, a, b):
        return torch.cat((a, b), dim=-1)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        p = ModelParams()
        assert p.nb_erb % 4 == 0, "erb_bins should be divisible by 4"

        self.erb_conv0 = Conv2dNormAct(
            1, p.conv_ch, kernel_size=p.conv_kernel_inp, bias=False, separable=True
        )
        conv_layer = partial(  # partial: fix paras mentioned above in func
            Conv2dNormAct,
            in_ch=p.conv_ch,
            out_ch=p.conv_ch,
            kernel_size=p.conv_kernel,
            bias=False,
            separable=True,
        )
        self.erb_conv1 = conv_layer(fstride=2)
        self.erb_conv2 = conv_layer(fstride=2)
        self.erb_conv3 = conv_layer(fstride=1)
        self.df_conv0 = Conv2dNormAct(
            2, p.conv_ch, kernel_size=p.conv_kernel_inp, bias=False, separable=True
        )
        self.df_conv1 = conv_layer(fstride=2)
        self.erb_bins = p.nb_erb
        self.emb_in_dim = p.conv_ch * p.nb_erb // 4
        self.emb_out_dim = p.emb_hidden_dim  # 256
        if p.enc_concat:  # fusion method: concat/add
            self.emb_in_dim *= 2  # 96*2=192
            self.combine = Concat()
        else:
            self.combine = Add()
        if p.pr_lin_groups != 1:
            self.df_fc_emb = GroupedLinear(
                input_size=p.conv_ch * p.nb_df // 2,
                hidden_size=p.conv_ch * p.nb_erb // 4,
                groups=p.pr_lin_groups
            )
        else:
            self.df_fc_emb = nn.Linear(p.conv_ch * p.nb_df // 2, p.conv_ch * p.nb_erb // 4)  # 16*96/2=16*48, C*F flatten
        self.emb_gru = oneGRU(self.emb_in_dim, self.emb_out_dim, p.pr_lin_groups)

        # not use yet
        self.lsnr_fc = nn.Sequential(nn.Linear(self.emb_out_dim, 1), nn.Sigmoid())
        self.lsnr_scale = p.lsnr_max - p.lsnr_min
        self.lsnr_offset = p.lsnr_min

    def forward(
        self, feat_erb: Tensor, feat_spec: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        # Encodes erb; erb should be in dB scale + normalized; Fe are number of erb bands.
        # erb: [B, 1, T, Fe]
        # spec: [B, 2, T, Fc]
        # b, _, t, _ = feat_erb.shape
        e0 = self.erb_conv0(feat_erb)  # [B, C, T, E]
        e1 = self.erb_conv1(e0)  # [B, C, T, E/2]
        e2 = self.erb_conv2(e1)  # [B, C, T, E/4]
        e3 = self.erb_conv3(e2)  # [B, C, T, E/4]
        c0 = self.df_conv0(feat_spec)  # [B, C, T, df]
        c1 = self.df_conv1(c0)  # [B, C, T, df/2]

        # cemb = c1.permute(0, 2, 3, 1).flatten(2)  # [B, T, C*df/2]
        # compatible for jit
        cemb = c1.permute(0, 2, 3, 1).contiguous()
        cemb_shape = cemb.shape
        cemb = cemb.view(cemb_shape[0], cemb_shape[1], -1)  # [B, T, C*df/2]
        cemb = self.df_fc_emb(cemb)  # [B, T, C*E/4]

        # emb = e3.permute(0, 2, 3, 1).flatten(2)  # [B, T, C*E/4]
        emb = e3.permute(0, 2, 3, 1).contiguous()
        emb_shape = emb.shape
        emb = emb.view(emb_shape[0], emb_shape[1], -1)
        # emb = self.erb_fc_emb(emb)

        emb = self.combine(emb, cemb)  # add: [B, T, C*E/4]
        emb, _ = self.emb_gru(emb)  # [B, T, emb_out_dim]
        lsnr = self.lsnr_fc(emb) * self.lsnr_scale + self.lsnr_offset
        return e0, e1, e2, e3, emb, c0, lsnr


class ErbDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        p = ModelParams()
        assert p.nb_erb % 8 == 0, "erb_bins should be divisible by 8"

        self.emb_out_dim = p.conv_ch * p.nb_erb // 4  # 96

        self.emb_gru1 = nn.GRU(p.emb_hidden_dim, p.emb_hidden_dim)
        self.emb_gru2 = nn.GRU(p.emb_hidden_dim, p.emb_hidden_dim // 2)
        self.emb_gru_out = GroupedLinear(p.emb_hidden_dim // 2, self.emb_out_dim, p.fix_lin_groups)

        tconv_layer = partial(
            ConvTranspose2dNormAct,
            kernel_size=p.conv_kernel,
            bias=False,
            separable=True,
        )
        conv_layer = partial(
            Conv2dNormAct,
            bias=False,
            separable=True,
        )
        # convt: TransposedConvolution, convp: Pathway (encoder to decoder) convolutions
        self.conv3p = Conv2dNormAct(p.conv_ch, p.conv_ch, kernel_size=1, separable=False)  # change this to separate channel group
        # self.conv3p = conv_layer(p.conv_ch, p.conv_ch, kernel_size=1)
        self.convt3 = conv_layer(p.conv_ch, p.conv_ch, kernel_size=p.conv_kernel)
        self.conv2p = conv_layer(p.conv_ch, p.conv_ch, kernel_size=1)
        self.convt2 = tconv_layer(p.conv_ch, p.conv_ch, fstride=2)
        self.conv1p = conv_layer(p.conv_ch, p.conv_ch, kernel_size=1)
        self.convt1 = tconv_layer(p.conv_ch, p.conv_ch, fstride=2)
        self.conv0p = conv_layer(p.conv_ch, p.conv_ch, kernel_size=1)
        self.conv0_out = conv_layer(p.conv_ch, 1, kernel_size=p.conv_kernel, activation_layer=nn.Sigmoid)

    def forward(self, emb, e3, e2, e1, e0) -> Tensor:
        # Estimates erb mask
        b, _, t, f = e3.shape  # [B, C, T, E/4]
        emb, _ = self.emb_gru1(emb)  # [B, T, C*F/4]
        emb, _ = self.emb_gru2(emb)  # [B, T, C*F/4]
        emb = self.emb_gru_out(emb)  # [B, T, C*F/4]
        emb = emb.view(b, t, f, -1).permute(0, 3, 1, 2)  # [B, C, T, F/4]
        e3 = self.convt3(self.conv3p(e3) + emb)  # [B, C, T, F/4]
        e2 = self.convt2(self.conv2p(e2) + e3)  # [B, C, T, F/2]
        e1 = self.convt1(self.conv1p(e1) + e2)  # [B, C, T, F]
        m = self.conv0_out(self.conv0p(e0) + e1)  # [B, 1, T, F]
        return m


class DfOutputReshapeMF(nn.Module):
    """Coefficients output reshape for multiframe/MultiFrameModule

    Requires input of shape [B, T, F, O*2] -> [B, O, T, F, 2]
    """

    def __init__(self, df_order: int, df_bins: int):
        super().__init__()
        self.df_order = df_order
        self.df_bins = df_bins

    def forward(self, coefs: Tensor) -> Tensor:
        # [B, T, F, O*2] -> [B, O, T, F, 2]
        # coefs = coefs.unflatten(-1, (-1, 2)).permute(0, 3, 1, 2, 4)
        # Onnx compatible version:
        new_shape = list(coefs.shape)
        new_shape[-1] = -1
        new_shape.append(2)
        coefs = coefs.view(new_shape)
        coefs = coefs.permute(0, 3, 1, 2, 4)
        return coefs


class DfDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        p = ModelParams()
        layer_width = p.conv_ch  # 16
        self.emb_dim = p.emb_hidden_dim  # 256

        self.df_n_hidden = p.df_hidden_dim  # 256
        self.df_n_layers = p.df_num_layers  # 2
        self.df_order = p.df_order
        self.df_bins = p.nb_df
        self.df_lookahead = p.df_lookahead
        self.df_out_ch = p.df_order * 2

        conv_layer = partial(Conv2dNormAct, bias=False, separable=True)
        kt = p.df_pathway_kernel_size_t
        self.df_convp = conv_layer(layer_width, self.df_out_ch, fstride=1, kernel_size=(kt, 1))
        self.df_gru = doubleGRU(p.emb_hidden_dim, p.df_hidden_dim, p.pr_lin_groups)
        out_dim = self.df_bins * self.df_out_ch  # 96*5*2
        df_out = GroupedLinear(self.df_n_hidden, out_dim, groups=p.fix_lin_groups)
        self.df_out = nn.Sequential(df_out, nn.Tanh())

    def forward(self, emb: Tensor, c0: Tensor) -> Tensor:
        b, t, _ = emb.shape  # [B, T, H0]  H0: emb_hidden, H1: df_hidden
        c, _ = self.df_gru(emb)  # [B, T, H1]
        c0 = self.df_convp(c0).permute(0, 2, 3, 1)  # [B, T, df, order*2], channels_last
        c = self.df_out(c)  # [B, T, df*order*2]
        c = c.view(b, t, self.df_bins, self.df_out_ch) + c0  # [B, T, df, order*2]
        return c


class DfNet(nn.Module):
    def __init__(
        self,
        erb_fb: Tensor,  # ERB filterbank, [F, Erb]
        erb_inv_fb: Tensor,  # ERB filterbank Inversion, [Erb, F]
    ):
        super().__init__()
        p = ModelParams()
        layer_width = p.conv_ch
        self.lookahead: int = p.conv_lookahead
        self.freq_bins: int = p.fft_size // 2 + 1
        self.emb_dim: int = layer_width * p.nb_erb
        self.erb_bins: int = p.nb_erb
        if p.conv_lookahead > 0:
            pad = (0, 0, -p.conv_lookahead, p.conv_lookahead)
            self.pad = nn.ConstantPad2d(pad, 0.0)
        else:
            self.pad = nn.Identity()
        
        # ERB Gain
        self.register_buffer("erb_fb", erb_fb)
        self.enc = Encoder()
        self.erb_dec = ErbDecoder()
        self.mask = Mask(erb_inv_fb, post_filter=p.mask_pf)

        # DF Coefs
        self.df_order = p.df_order
        self.df_bins = p.nb_df
        self.df_lookahead = p.df_lookahead
        self.df_op = MF.DF(num_freqs=p.nb_df, frame_size=p.df_order, lookahead=p.df_lookahead)
        self.df_dec = DfDecoder()
        self.df_out_transform = DfOutputReshapeMF(self.df_order, self.df_bins)

    def forward(
        self,
        spec: Tensor,
        feat_erb: Tensor,
        feat_spec: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        feat_spec = feat_spec.squeeze(1).permute(0, 3, 1, 2)  # [B, 2, T, df]

        feat_erb = self.pad(feat_erb)
        feat_spec = self.pad(feat_spec)
        e0, e1, e2, e3, emb, c0, lsnr = self.enc(feat_erb, feat_spec)
        m = self.erb_dec(emb, e3, e2, e1, e0)  # erb gain

        spec = self.mask(spec, m)
        df_coefs = self.df_dec(emb, c0)
        df_coefs = self.df_out_transform(df_coefs)
        spec = self.df_op(spec, df_coefs)

        return spec, m, lsnr


if __name__ == "__main__":
    config.load("config.ini")
    model = init_model()
    model = torch.jit.script(model)
    model.save('model_jit.pt')
    model = torch.jit.load('model_jit.pt')
    device = get_device()
    spec = torch.randn([4, 1, 10, 481, 2]).to(device)
    feat_erb = torch.randn([4, 1, 10, 32]).to(device)
    feat_spec = torch.randn([4, 1, 10, 96, 2]).to(device)
    model.forward(spec, feat_erb, feat_spec)