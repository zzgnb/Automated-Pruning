from modules import GroupedLinear
from typing import Sequence
import torch
from torch import nn
import torch_pruning as tp
from copy import deepcopy


class L1NormImportance(tp.importance.Importance):
    def __call__(self, group):
        group_imp = [] 
        print("*" * 100)
        print(f'Root: [{group[0][0].target.m_name}] {group[0][0].target.module} L1norm_imp Including:')
        for dep, idxs in group:
            module = dep.target.module
            m_name = dep.target.m_name
            prune_fn = dep.handler    # get pruning function
            if isinstance(module, nn.Linear):
                fmt = f'[{m_name}] {module}: '
                if prune_fn == tp.function.prune_linear_out_channels:
                    print(fmt + 'out_channels')
                    w = module.weight.data[idxs]
                    b = module.bias.data[idxs].unsqueeze(-1)
                    local_norm = torch.cat((w, b), dim=1).abs().sum(1)
                elif prune_fn == tp.function.prune_linear_in_channels:
                    print(fmt + 'in_channels')
                    w = module.weight.data[:, idxs]
                    local_norm = w.abs().sum(0)
                else:
                    continue

            elif isinstance(module, (nn.LSTM, nn.GRU)):
                fmt = f'[{m_name}] {module}: '
                h_num = module.hidden_size
                if prune_fn in (tp.function.prune_lstm_out_channels, tp.function.prune_gru_out_channels):
                    print(fmt + 'out_channels')
                    extend_times = 4 if module.mode == "LSTM" else 3
                    _idxs = torch.tensor(idxs)
                    expand_idxs = torch.cat([_idxs+i*h_num for i in range(extend_times)], dim=0) 
                    local_norm = torch.zeros((h_num, ), device="cuda")
                    for l in range(module.num_layers):
                        w_hh = getattr(module, 'weight_hh_l' + str(l)).data[:, idxs]
                        b_hh = getattr(module, 'bias_hh_l' + str(l)).data[expand_idxs].unsqueeze(-1)
                        w_ih = getattr(module, 'weight_ih_l' + str(l)).data[expand_idxs]
                        b_ih = getattr(module, 'bias_ih_l' + str(l)).data[expand_idxs].unsqueeze(-1)

                        hh_norm = torch.cat((w_hh, b_hh), dim=1).abs().sum(1, keepdim=True).view(extend_times, -1)
                        ih_norm = torch.cat((w_ih, b_ih), dim=1).abs().sum(1, keepdim=True).view(extend_times, -1)
                        local_norm += hh_norm.sum(0)
                        local_norm += ih_norm.sum(0)
                elif prune_fn in (tp.function.prune_lstm_in_channels, tp.function.prune_gru_in_channels):
                    print(fmt + 'in_channels')
                    w = module.weight_ih_l0.data[:, idxs]
                    local_norm = w.abs().sum(0)
                else:
                    continue

            elif isinstance(module, tp.ops.RNN_ih): 
                fmt = f'[{m_name}] {module}: '
                layer = module.rnn_layer
                if prune_fn == tp.function.prune_RNN_ih_out_channels:
                    print(fmt + 'out_channels')
                    w = layer.weight_ih_l0.data[idxs]
                    local_norm = w.abs().sum(1)
                    extend_times = 4 if module.rnn_type == "LSTM" else 3
                    local_norm = local_norm.view(-1, extend_times)
                    local_norm = local_norm.sum(1)
                elif prune_fn == tp.function.prune_RNN_ih_in_channels:
                    print(fmt + 'in_channels')
                    w = layer.weight_ih_l0.data[idxs]
                    local_norm = w.abs().sum(0)
                else:
                    continue

            elif isinstance(module, tp.ops.RNN_hh): 
                fmt = f'[{m_name}] {module}: '
                layer = module.rnn_layer
                if prune_fn == tp.function.prune_RNN_hh_out_channels:
                    print(fmt + 'out_channels')
                    w = layer.weight_hh_l0.data[idxs]
                    local_norm = w.abs().sum(0)
                elif prune_fn == tp.function.prune_RNN_hh_in_channels:
                    print("warning: never have to RNN_hh_in_channels pruning, sth wrong")
                else:
                    continue
            
            elif isinstance(module, GroupedLinear):
                module: GroupedLinear
                fmt = f'[{m_name}] {module}: '
                if module.shuffle:
                    print(fmt + 'output shuffled, skip it')  
                    continue
                if prune_fn == GLinearPruner.prune_out_channels:
                    print(fmt + 'out_channels')
                    # index mapping
                    gidxs_list = [[] for _ in range(module.groups)]
                    for i in idxs:
                        for s in range(module.groups):
                            if i >= module.gout_slice[s] and i < module.gout_slice[s+1]:
                                gidxs_list[s].append(i-module.gout_slice[s])

                    local_norm = []
                    for g_i, gidxs in enumerate(gidxs_list):
                        g_w = module.layers[g_i].weight.data[gidxs]
                        g_b = module.layers[g_i].bias.data[gidxs].unsqueeze(-1)
                        g_local_norm = torch.cat((g_w, g_b), dim=1).abs().sum(1)
                        local_norm.append[g_local_norm]
                    local_norm = torch.stack(local_norm).flatten(0)

                elif prune_fn == GLinearPruner.prune_in_channels:
                    print(fmt + 'in_channels')
                    # index mapping
                    gidxs_list = [[] for _ in range(module.groups)]
                    for i in idxs:
                        for s in range(module.groups):
                            if i >= module.gin_slice[s] and i < module.gin_slice[s+1]:
                                gidxs_list[s].append(i-module.gin_slice[s])

                    local_norm = []
                    for g_i, gidxs in enumerate(gidxs_list):
                        g_w = module.layers[g_i].weight.data[:, gidxs]
                        g_local_norm = g_w.abs().sum(0)
                        local_norm.append(g_local_norm)
                    local_norm = torch.stack(local_norm).flatten(0)
                else:
                    continue

            else:
                continue
            group_imp.append(local_norm)

        # get importance per channel
        group_imp = torch.stack(group_imp, dim=0).mean(dim=0)
        print(f'g_imp shape: {group_imp.shape}')
        print("*" * 100)
        return group_imp


class L2NormImportance(tp.importance.Importance):
    def __call__(self, group):
        group_imp = [] 
        group_w = None
        print("*" * 100)
        print(f'Root: [{group[0][0].target.m_name}] {group[0][0].target.module} L2norm_imp Including:')
        for dep, idxs in group:
            module = dep.target.module
            m_name = dep.target.m_name
            prune_fn = dep.handler    # get pruning function
            
            if isinstance(module, nn.Linear):
                fmt = f'[{m_name}] {module}: '
                if prune_fn == tp.function.prune_linear_out_channels:
                    print(fmt + 'out_channels')
                    w = module.weight.data[idxs]
                    b = module.bias.data[idxs].unsqueeze(-1)
                    local_w = torch.cat((w, b), dim=1)
                elif prune_fn == tp.function.prune_linear_in_channels:
                    print(fmt + 'in_channels')
                    local_w = module.weight.data[:, idxs].T
                else:
                    continue

            elif isinstance(module, (nn.LSTM, nn.GRU)):
                fmt = f'[{m_name}] {module}: '
                h_num = module.hidden_size
                if prune_fn in (tp.function.prune_lstm_out_channels, tp.function.prune_gru_out_channels):
                    print(fmt + 'out_channels')
                    extend_times = 4 if module.mode == "LSTM" else 3
                    _idxs = torch.tensor(idxs)
                    expand_idxs = torch.cat([_idxs+i*h_num for i in range(extend_times)], dim=0) 
                    local_w = None
                    for l in range(module.num_layers):
                        w_hh = getattr(module, 'weight_hh_l' + str(l)).data[:, idxs]
                        b_hh = getattr(module, 'bias_hh_l' + str(l)).data[expand_idxs].unsqueeze(-1)
                        w_ih = getattr(module, 'weight_ih_l' + str(l)).data[expand_idxs]
                        b_ih = getattr(module, 'bias_ih_l' + str(l)).data[expand_idxs].unsqueeze(-1)

                        hh_tmp_w = torch.cat((w_hh, b_hh), dim=1).view(extend_times, h_num, -1).transpose(0, 1).contiguous().view(h_num, -1)
                        ih_tmp_w = torch.cat((w_ih, b_ih), dim=1).view(extend_times, h_num, -1).transpose(0, 1).contiguous().view(h_num, -1)
                        local_w = deepcopy(hh_tmp_w) if local_w is None else torch.cat((local_w, hh_tmp_w), dim=1)
                        local_w = torch.cat((local_w, ih_tmp_w), dim=1)
                elif prune_fn in (tp.function.prune_lstm_in_channels, tp.function.prune_gru_in_channels):
                    print(fmt + 'in_channels')
                    local_w = module.weight_ih_l0.data[:, idxs].T
                else:
                    continue
            
            elif isinstance(module, GroupedLinear):
                module: GroupedLinear
                fmt = f'[{m_name}] {module}: '
                print(fmt + 'output shuffled, skip it')
                continue

            else:
                continue

            group_w = deepcopy(local_w) if group_w is None else torch.cat((group_w, local_w), dim=1)

        # get l2 importance per channel
        pg = group_w.shape[-1]
        group_w = torch.pow(group_w, 2)
        group_imp = torch.sqrt(group_w.sum(1))
        print(f'g_imp shape: {group_imp.shape}, pg: {pg}')
        print("*" * 100)
        return group_imp, pg
 

class GLinearPruner(tp.pruner.BasePruningFunc):
    def prune_out_channels(self, layer: GroupedLinear, idxs: Sequence[int]) -> nn.Module: 
        # because of shuffle, we don't use idxs but compute L1norm by itself
        glinear = layer.layers
        hidden_size = layer.hidden_size

        # importance analysis
        local_norm = []
        local_norm = torch.tensor(local_norm, device=get_device())
        for g in glinear:
            local_norm = torch.concat((local_norm, g.weight.data), dim=0)
        local_norm = local_norm.abs().sum(1)
        sorted_idxs = torch.argsort(local_norm).tolist()

        # index mapping and avoid pruning one sublayer all
        pr_num = 0
        g_pr_num = [0] * layer.groups
        gidxs_list = [[] for _ in range(layer.groups)]
        for i in sorted_idxs:
            if pr_num == len(idxs):
                break
            for s in range(layer.groups):
                if i >= layer.gout_slice[s] and i < layer.gout_slice[s+1]:
                    if layer.gout_slice[s+1] - layer.gout_slice[s] - g_pr_num[s] > 1:
                        gidxs_list[s].append(i-layer.gout_slice[s])
                        g_pr_num[s] += 1
                        pr_num += 1
                    break

        for g_i, gidxs in enumerate(gidxs_list):
            keep_idxs = list(set(range(glinear[g_i].out_features)) - set(gidxs))
            keep_idxs.sort()
            glinear[g_i].weight = self._prune_parameter_and_grad(glinear[g_i].weight, keep_idxs, 0)
            if glinear[g_i].bias is not None:
                glinear[g_i].bias = self._prune_parameter_and_grad(glinear[g_i].bias, keep_idxs, 0)
            layer.gout_slice[g_i+1] = len(keep_idxs) + layer.gout_slice[g_i]
            glinear[g_i].out_features = len(keep_idxs)

        return layer
    
    def prune_in_channels(self, layer: GroupedLinear, idxs: Sequence[int]) -> nn.Module: 
        # because of shuffle, we don't use idxs but compute L1norm by itself
        glinear = layer.layers
        input_size = layer.input_size

        # importance analysis
        local_norm = []
        local_norm = torch.tensor(local_norm, device=get_device())
        for g in glinear:
            local_norm = torch.concat((local_norm, g.weight.data), dim=1)
        local_norm = local_norm.abs().sum(0)
        sorted_idxs = torch.argsort(local_norm).tolist()

        # index mapping and avoid pruning one sublayer all
        pr_num = 0
        g_pr_num = [0] * layer.groups
        gidxs_list = [[] for _ in range(layer.groups)]
        for i in sorted_idxs:
            if pr_num == len(idxs):
                break
            for s in range(layer.groups):
                if i >= layer.gin_slice[s] and i < layer.gin_slice[s+1]:
                    if layer.gin_slice[s+1] - layer.gin_slice[s] - g_pr_num[s] > 1:
                        gidxs_list[s].append(i-layer.gin_slice[s])
                        g_pr_num[s] += 1
                        pr_num += 1
                    break

        for g_i, gidxs in enumerate(gidxs_list):
            keep_idxs = list(set(range(glinear[g_i].in_features)) - set(gidxs))
            keep_idxs.sort()
            glinear[g_i].weight = self._prune_parameter_and_grad(glinear[g_i].weight, keep_idxs, 1)
            layer.gin_slice[g_i+1] = len(keep_idxs) + layer.gin_slice[g_i]
            glinear[g_i].in_features = len(keep_idxs)

        return layer

    def get_out_channels(self, layer: GroupedLinear):
        return layer.gout_slice[-1]
    
    def get_in_channels(self, layer: GroupedLinear):
        return layer.gin_slice[-1]


def get_device():
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:0")
    else:
        DEVICE = torch.device("cpu")
    return DEVICE 