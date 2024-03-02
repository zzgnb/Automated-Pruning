from model import ModelParams
from checkpoint import load_model
from config import config
from libdfdata import PytorchDataLoader as DataLoader
from libdf import DF
from loss import Istft, Loss
from utils import get_norm_alpha, as_real
from modules import Conv2dNormAct, ConvTranspose2dNormAct, GroupedLinear
from train import get_device, cleanup
from prune_utils import L1NormImportance, GLinearPruner

from loguru import logger
from logger import init_logger, log_metrics, log_model_summary
import torch
import torch.optim as optim
import os

from math import floor
from functools import partial
from torch import nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from copy import deepcopy
import numpy as np
import torch_pruning as tp


def set_losses(model):
    p = ModelParams()
    state = DF(sr=p.sr, fft_size=p.fft_size, hop_size=p.hop_size, 
               nb_bands=p.nb_erb, min_nb_erb_freqs=p.min_nb_freqs)
    istft = Istft(p.fft_size, p.hop_size, torch.as_tensor(state.fft_window().copy())).to(get_device())
    loss = Loss(state=state, istft=istft, model=model).to(get_device())  # loss weights identical to training
    return loss

# rewrite for your own validation function
def valid(model, loader: DataLoader, losses: Loss) -> float:
    losses.change_sparsity_weight(n_s1l_f=0, n_s2l_f=0, n_gs2l_f=0)
    dev = get_device()
    l_mem = []
    model.eval()
    
    for _, batch in enumerate(loader.iter_epoch('valid', seed=42)):
        assert batch.feat_spec is not None
        assert batch.feat_erb is not None
        feat_erb = batch.feat_erb.to(dev, non_blocking=True)
        feat_spec = as_real(batch.feat_spec.to(dev, non_blocking=True))
        noisy = batch.noisy.to(dev, non_blocking=True)
        clean = batch.speech.to(dev, non_blocking=True)
        snrs = batch.snr.to(dev, non_blocking=True)

        input = as_real(noisy).clone()
        enh, m, _ = model.forward(spec=input, feat_erb=feat_erb, feat_spec=feat_spec)
        err = losses.forward(clean, noisy, enh, m, snrs=snrs)
        l_mem.append(err.detach())

    try:
        cleanup(err, noisy, clean, enh, m, feat_erb, feat_spec, batch)
    except UnboundLocalError as err:
        logger.error(str(err))

    return torch.stack(l_mem).mean().cpu().item()


class sensi_analysis:
    def __init__(self, model, DataLoader, Loss: Loss, thres,
                 model_inputs=None, initDG=None, ignored_layers=None, root_module=None):
        # configurations and validset don't change throughout prune-finetune iterations
        self.L1Norm = L1NormImportance()
        self.losses = Loss
        self.valid_loss = partial(valid, loader=DataLoader, losses=self.losses)
        self.thres = thres
        self.model = model
        self.model_inputs = model_inputs
        self.DG: tp.DependencyGraph = initDG
        self.ignored_layers = ignored_layers
        self.root_module = root_module
        self.pr_list = np.linspace(start=0, stop=1, num=21)[:-1]
    
    # generate pr and pr_idxs based on DG
    def model_sen_alys_DG(self, g_imp_dict):
        base_l = self.valid_loss(model=self.model)
        logger.info(f"Model base_loss: {base_l:.4f}, Threshold: {self.thres}.")
        self.pr_idxs_dict = {}
        last_df = 0
        
        for m_n, g_imp in g_imp_dict.items():
            logger.info(f"[{m_n}] analysis begin.")
            nonzero_num = torch.count_nonzero(g_imp)
            zero_num = len(g_imp) - nonzero_num
            for pr in self.pr_list:
                pr_num = floor(nonzero_num * pr)
                pr_idxs = torch.argsort(g_imp).tolist()[zero_num: zero_num + pr_num]
                # tmp model and DG
                tmp_model = deepcopy(self.model)
                tmp_DG = self.DG.build_dependency(model=tmp_model, example_inputs=self.model_inputs)

                for _n, _m in tmp_model.named_modules():             
                    if _n == m_n:
                        tmp_m = _m
                        break
                pruner = tmp_DG.get_pruner_of_module(tmp_m)
                layer_channels = pruner.get_out_channels(tmp_m)
                g = tmp_DG.get_pruning_group(tmp_m, pruner.prune_out_channels, list(range(layer_channels)))
                g.prune(idxs=pr_idxs)
                
                # test for one second
                log_model_summary(tmp_model, verbose=False)

                # test performance and recover to org_para
                prune_l = self.valid_loss(tmp_model)
                l_df = prune_l - base_l
                if l_df > self.thres:   
                    logger.info(f"[{m_n}] {tmp_m} pruned_loss_df: {l_df:.3f} > {self.thres}, analysis stop.")
                    logger.info(f"[{m_n}] {tmp_m} final pr: {self.pr_idxs_dict[m_n][0]*100:.1f}%, loss_df: {last_df:.3f}.")
                    break
                self.pr_idxs_dict[m_n] = (pr, pr_idxs)
                last_df = l_df
                if pr == self.pr_list[-1]:
                    logger.info(f"[{m_n}] {tmp_m} final pr: {self.pr_idxs_dict[m_n][0]*100:.1f}% , loss_df: {l_df:.3f}.")
                else:
                    logger.info(f"[{m_n}] {tmp_m} pr: {self.pr_idxs_dict[m_n][0]*100:.1f}% , loss_df: {l_df:.3f}.")
                del tmp_model

    # generate pr and pr_idxs per weight matrix individually
    def model_sen_alys_nonDG(self):
        base_l = self.valid_loss(model=self.model)
        logger.info(f"Model base_loss: {base_l:.3f}, Stop Threshold: {self.thres}.")
        self.pr_idxs_dict = {}  # note that the key is (m, w_n) different from m_n in the DG algorithm
        last_df = 0

        for m_n, m in self.model.named_modules():
            if isinstance(m, (nn.LSTM, nn.GRU, nn.Linear)):
                logger.info(f"[{m_n}] {m} analysis begin.")
                for w_n, w in m.named_parameters():
                    if 'bias' not in w_n:
                        l1norm = w.data.abs().sum(0)
                        nonzero_num = torch.count_nonzero(l1norm)
                        zero_num = len(l1norm) - nonzero_num
                        org_w = deepcopy(w.data)
                        for pr in self.pr_list:
                            # prune tensor
                            pr_num = floor(nonzero_num * pr)
                            pr_idxs = torch.argsort(l1norm)[zero_num: zero_num + pr_num]
                            for pr_i in pr_idxs:
                                w.data[:, pr_i] = 0
                            
                            # test performance and recover to org_para
                            prune_l = self.valid_loss(self.model)   
                            l_df = prune_l - base_l
                            w.data[:] = org_w
                            if l_df > self.thres:
                                logger.info(f"[{m}] {w_n} analysis stop.")
                                logger.info(f"[{m}] {w_n} final pr: {self.pr_idxs_dict[(m, w_n)][0]*100:.1f}%, loss_df: {last_df:.3f}.")
                                break
                            self.pr_idxs_dict[(m, w_n)] = (pr, pr_idxs)
                            last_df = l_df
                            if pr == self.pr_list[-1]:
                                logger.info(f"[{m}] {w_n} final pr: {self.pr_idxs_dict[(m, w_n)][0]*100:.1f}% , loss_df: {l_df:.3f}.")
                            else:
                                logger.info(f"[{m}] {w_n} pr: {self.pr_idxs_dict[(m, w_n)][0]*100:.1f}% , loss_df: {l_df:.3f}.")
                        del org_w
            else:
                logger.info(f"[{m_n}] not in analysis category, skipped.")
                continue
    
    # get the max pruning num
    def max_pnum(self):
        max_pnum = 0
        for (pr, pr_idxs) in self.pr_idxs_dict.values():
            max_pnum = len(pr_idxs) if len(pr_idxs) > max_pnum else max_pnum
        return max_pnum

    # update the model for analysis
    def update_model(self, model):
        self.model = model
 

def DG_pruning(base_dir, sr, thres, start_epoch):
    min_prune_num = 3
    max_prft_epoch = 5
    ft_patience = 5
    ft_max_epoch = 30
    ft_lr = 5e-5

    # 1. load sparsify-trained model and dataset for validation and fine-tuning
    s_decay = 0.9
    DG_compression_dir = os.path.join(base_dir, "DG_compression")
    if not os.path.exists(DG_compression_dir):
        os.mkdir(DG_compression_dir)
    init_logger(file=os.path.join(DG_compression_dir, "pruning.log"), model=base_dir)
    config.load(os.path.join(base_dir, "config.ini"))
    model, _ = load_model(cp_dir=os.path.join(base_dir, "checkpoints"), df_state=None)
    logger.info("Running on device {}".format(get_device()))
    p = ModelParams()
    dataloader = DataLoader(
        ds_dir=f"/E/zhaozugang/SpeechEhancement/DeepFilterNet/self_test/small_hdf5/{sr}KHz",
        ds_config=f"/E/zhaozugang/SpeechEhancement/DeepFilterNet/self_test/small_hdf5/{sr}KHz/dataset.cfg",
        sr=p.sr,
        batch_size=64,
        batch_size_eval=32,
        num_workers=4,
        pin_memory=get_device().type == "cuda",
        max_len_s=5.0,
        fft_size=p.fft_size,
        hop_size=p.hop_size,
        nb_erb=p.nb_erb,
        nb_spec=p.nb_df,
        norm_alpha=get_norm_alpha(log=False),
        p_reverb=0.2,
        p_bw_ext=0,
        p_clipping=0,
        p_zeroing=0,
        p_air_absorption=0,
        prefetch=32,
        overfit=False,
        seed=42,
        min_nb_erb_freqs=p.min_nb_freqs,
        log_timings=False,
        global_sampling_factor=1,
        snrs=[-10, -5, 0, 5, 10, 20, 30],
        log_level="INFO",
    )
    losses = set_losses(model)

    # model inputs for test
    model_inputs = {
        "spec": torch.randn((1, 1, 10, 481, 2), device=torch.device("cuda")),
        "feat_erb": torch.randn((1, 1, 10, 32), device=torch.device("cuda")),
        "feat_spec": torch.randn((1, 1, 10, 96, 2), device=torch.device("cuda"))
    }

    # ignore layers for pruning(unprunable layers, output layers, etc) 
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

    # Register customized layer
    DG = tp.DependencyGraph()
    GLPruner = GLinearPruner(torch.device("cuda"))
    DG.register_customized_layer(GroupedLinear, GLPruner)

    # initial sensitivity analysis
    sen_alys = sensi_analysis(model=model, DataLoader=dataloader, Loss=losses, thres=thres,
                              model_inputs=model_inputs, initDG=DG,
                              ignored_layers=ignored_layers, root_module=root_module)

    # 2. prune-finetune process
    pr_dir = os.path.join(DG_compression_dir, "pruned")
    if not os.path.exists(pr_dir):
        os.mkdir(pr_dir)
    ft_dir = os.path.join(DG_compression_dir, "finetuned")
    if not os.path.exists(ft_dir):
        os.mkdir(ft_dir)

    for i_prft_epoch in range(start_epoch, max_prft_epoch):
        logger.info('*' * 120)
        logger.info(f"Begin [{i_prft_epoch}] prune-finetune process")
        if i_prft_epoch != 0:
            last_model_n = 'ft_ep_' + str(i_prft_epoch-1) +'.pth'
            model = torch.load(os.path.join(ft_dir, last_model_n))
            sen_alys.update_model(model)
            logger.info(f"Load finetuned model of last epoch [{i_prft_epoch-1}]")

        DG.build_dependency(model, example_inputs=model_inputs)
        # weight group importance
        g_imp_dict = {}
        for g in DG.get_all_groups(ignored_layers=ignored_layers, root_module_types=root_module):
            m_n = g[0][0].target.m_name
            g_imp = sen_alys.L1Norm(g)
            g_imp_dict[m_n] = g_imp
            print(m_n, g_imp.shape)

        logger.info('-' * 80)
        logger.info("Start Sensitivity Analysis")
        sen_alys.model_sen_alys_DG(g_imp_dict)
        if sen_alys.max_pnum() < min_prune_num:
            logger.info("Pruning is already sufficient, stop the prune-finetune process")
            break

        # prune model based on sen_alys
        logger.info('-' * 80)
        logger.info("Start Pruning")
        # test for one second
        log_model_summary(model, verbose=False)
        DG.build_dependency(model, example_inputs=model_inputs)
        for g in DG.get_all_groups(ignored_layers=ignored_layers, root_module_types=root_module):
            m_n = g[0][0].target.m_name
            pr, pr_idxs = sen_alys.pr_idxs_dict[m_n]
            g.prune(idxs=pr_idxs)
        log_model_summary(model, verbose=False)

        # save pruned model
        pr_model_n = 'pr_ep_' + str(i_prft_epoch) +'.pth'
        model.zero_grad()
        torch.save(model, os.path.join(pr_dir, pr_model_n))
        logger.info(f"Save Pruned model as {pr_model_n}")

        # following method will save the original model and prune-changed attributes
        # state_dict = tp.state_dict(model)
        # torch.save(state_dict, os.path.join(pr_dir, 'pruned_epoch_' + str(i) +'.pth'))

        # finetune model
        logger.info('-' * 80)
        logger.info("Start Fine-tuning")
        dev = get_device()
        model = torch.load(os.path.join(pr_dir, pr_model_n)).to(dev)
        opt = optim.Adam(model.parameters(), lr=ft_lr)
        losses.change_sparsity_weight(n_s1l_f=losses.s1l_f * pow(s_decay, i_prft_epoch), 
                                      n_s2l_f=losses.s2l_f * pow(s_decay, i_prft_epoch),
                                      n_gs2l_f=losses.gs2l_f * pow(s_decay, i_prft_epoch))
        if losses.gs2l_f > 0:
            losses.change_gL2_DG(p=p, model=model)
        ft_valid = partial(valid, loader=dataloader, losses=losses)
        min_loss = ft_valid(model)
        patience = 0
        ft_model_n = 'ft_ep_' + str(i_prft_epoch) +'.pth'
        for i_epoch in range(ft_max_epoch):
            model.train()
            seed = i_epoch
            l_mem = []
            for _, batch in enumerate(dataloader.iter_epoch('train', seed)):
                opt.zero_grad()
                assert batch.feat_spec is not None
                assert batch.feat_erb is not None
                feat_erb = batch.feat_erb.to(dev, non_blocking=True)
                feat_spec = as_real(batch.feat_spec.to(dev, non_blocking=True))
                noisy = batch.noisy.to(dev, non_blocking=True)
                clean = batch.speech.to(dev, non_blocking=True)
                snrs = batch.snr.to(dev, non_blocking=True)
                input = as_real(noisy)
                enh, m, lsnr = model.forward(spec=input, feat_erb=feat_erb, feat_spec=feat_spec)
                err = losses.forward(clean, noisy, enh, m, snrs=snrs)
                err.backward()
                clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                opt.step()
                l_mem.append(err.detach())

            l_mean = torch.stack(l_mem).mean().cpu()
            l_dict = {"loss": l_mean.item()}
            l_dict["lr"] = opt.param_groups[0]["lr"]
            log_metrics(f"[{i_epoch}]", l_dict)
            
            ft_loss = ft_valid(model)
            if ft_loss < min_loss:
                patience = 0
                min_loss = ft_loss
                if os.path.exists(os.path.join(ft_dir, ft_model_n)):
                    os.remove(os.path.join(ft_dir, ft_model_n))
                model.zero_grad()
                torch.save(model, os.path.join(ft_dir, ft_model_n))
                logger.info(f"Update fine-tuned model to {ft_model_n}")
            else: 
                patience += 1
                for param_group in opt.param_groups:
                    param_group['lr'] *= 0.9

            if patience >= ft_patience:
                logger.info(f"Stop fine-tuning at epoch[{i_epoch}] as no improvements")
                break

        logger.info(f"End [{i_prft_epoch}] prune-finetune process")
        logger.info('*' * 120)


if __name__ == "__main__":  
    base_dir = "/E/zhaozugang/SpeechEhancement/DeepFilterNet/self_test/train_result/48KHz/tmp/DFNet2"
    sr = 48
    thres = 0.1
    DG_pruning(base_dir, sr, thres, start_epoch=0)