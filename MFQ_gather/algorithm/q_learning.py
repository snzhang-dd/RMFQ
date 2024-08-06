"""
# Original Work Copyright © 2018-present, Mean Field Reinforcement Learning (https://github.com/mlii/mfrl). 
# All rights reserved.
# Modifications (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""


import os
import torch
import numpy as np
import torch.nn.functional as F
import math
from . import base_mfq, base_rmfq, base_mtmfq


def get_minimap(position, state):
    """对翻转的state进行minimap提取

    Args:
        position (tensor): 位置矩阵batch_size*81*2
        state (tensor): 状态矩阵batch_size*81*45*45*5

    Returns:
        minimap(tensor):minimap矩阵batch_size*81*13*13*2
    """
    batch_size = state.size(0)
    n = state.size(1)
    width = state.size(2)
    
    state = state.permute(0, 1, 4, 3, 2)
    state = state.flip(dims=[-1])
    unit_size = math.ceil(state.size(-1) / 13)
    padding_size = unit_size*13 - state.size(-1)
    width += padding_size
    padding = [0, padding_size, 0, padding_size] + [0]*((len(state.shape)-2)*2)
    state_padded = F.pad(state, padding, "constant", 0)
    red_n = state_padded[:,:,1].sum(dim=[2,3]).view(batch_size, n, 1, 1).view(-1, 1, 1, 1)
    blue_n = state_padded[:,:,3].sum(dim=[2,3]).view(batch_size, n, 1, 1).view(-1, 1, 1, 1)
    state_mini_red = state_padded[:,:,1].unsqueeze(2).view(-1, 1, width, width)
    state_mini_blue = state_padded[:,:,3].unsqueeze(2).view(-1, 1, width, width)

    minimap_red = F.avg_pool2d(state_mini_red, unit_size)*(unit_size*unit_size)/(red_n+0.001)
    minimap_red = minimap_red.view(batch_size, n, 13, 13)

    minimap_blue = F.avg_pool2d(state_mini_blue, unit_size)*(unit_size*unit_size)/(blue_n+0.001)
    minimap_blue = minimap_blue.view(batch_size, n, 13, 13)
    
    # 位置处+1
    position_flip = position
    position_flip[:,:,1] = 44 - position[:,:,1]
    position_flip = (position_flip / 4).floor().long()
    batch_indices = torch.arange(batch_size).unsqueeze(1).expand(batch_size, n)
    n_indices = torch.arange(n).unsqueeze(0).expand(batch_size, n)

    # 提取位置坐标
    x_positions = position_flip[:,:,0]
    y_positions = position_flip[:,:,1]

    # 更新值
    minimap_red[batch_indices, n_indices, x_positions, y_positions] += 1
    minimap_blue[batch_indices, n_indices, x_positions, y_positions] += 1
    return minimap_red, minimap_blue
    
    
    
class MFQ(base_mfq.MFQ_NET):
    def __init__(self, args):
        super().__init__(args=args)

        self.device = args.device
        self.train_ct = 0
        self.update_every = args.target_update_freq
        self.train_step = 0
        self.train_recl_freq = args.train_recl_freq
        mapping = {0: 0, 1: 3, 2: 2, 3: 1, 4: 8, 5: 7, 6: 6, 7:5, 8:4, 9: 11, 11: 9, 10: 10, 12: 12, 13: 15, 15:13, 17:16, 16: 17, 18: 20, 20:18, 14: 14, 19: 19}
        self.mapping_tensor = torch.tensor([mapping[i] for i in range(max(mapping.keys())+1)]).to(self.device)

    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)

    def train(self, replay_buffer):

        for _ in range(self.args.multi_train_steps):
            batch_data, max_episode_len = replay_buffer.sample_random(self.args.batch_size)
                 
            obs = batch_data['obs_n'].to(self.device)
            reward = batch_data['r'].to(self.device)
            act = batch_data['a_n'].to(self.device)
            batch_last_a = batch_data['last_onehot_a_n'].to(self.device)
            mask = batch_data['active'].to(self.device)
            done = batch_data['dw'].to(self.device)
            # batch_avail_a_n = batch_data['avail_a_n']
            # next_obs = obs[:, 1:]
            # next_mean_action = mean_action[:, 1:]
            next_obs = batch_data['obs_next'].to(self.device)
            if self.args.use_att:
                global_role = batch_data['global_role'].to(self.device)
                # last_obs = batch_data['last_obs'].to(self.device)
                last_act = batch_data['last_act'].to(self.device)
                # last_role = batch_data['last_role'].to(self.device)
                next_role = batch_data['next_role'].to(self.device)
                next_mask = batch_data['active_next'].to(self.device)
                _, mean_action = self.cal_att_weights_mean_acts(obs, global_role, last_act, mask, key='eval')
                _, next_mean_action_e = self.cal_att_weights_mean_acts(next_obs, next_role, act, next_mask, key='eval')
                _, next_mean_action_t = self.cal_att_weights_mean_acts(next_obs, next_role, act, next_mask,key='target')
                next_mean_action = {'eval': next_mean_action_e, 'target': next_mean_action_t}
                
            else:
                mean_action = batch_data['mean_act'].to(self.device)
                next_mean_action = batch_data['mean_act_next'].to(self.device)
            
                    
            loss = super().update_q_nornn(obs, act, next_obs, reward, done, mask, mean_action, next_mean_action, max_episode_len)
            self.soft_update()

        self.train_step += 1
        return loss
    def soft_update_all_parms(self, model, tau):
        """soft-update

        Args:
            model (_type_):源模型,从model上获取参数
            tau (_type_): 软更新的参数
        """
        if self.args.use_att:
            model_params = {'eval_q_net':model.eval_net, 'target_q_net':model.target_net,  'att_net':model.att_weights_eval, 'att_net_target':model.att_weights_target}
            self_params = {'eval_q_net':self.eval_net, 'target_q_net':self.target_net,  'att_net':self.att_weights_eval, 'att_net_target':self.att_weights_target}
        else:
            model_params = {'eval_q_net':model.eval_net, 'target_q_net':model.target_net}
            self_params = {'eval_q_net':self.eval_net, 'target_q_net':self.target_net}
        for target, source in zip(self_params.values(), model_params.values()):
            for target_param, source_param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)
    
    def save_model(self, dir_path, step=0):
        if self.args.use_att:
            name_obj_list = [
                ("eval_net", self.eval_net),
                ("target_net", self.target_net),
                ("att_net", self.att_weights_eval),
                ("att_net_target", self.att_weights_target),
            ]
        else:
            name_obj_list = [
                ("eval_net", self.eval_net),
                ("target_net", self.target_net),
            ]
        for name, obj in name_obj_list:

            file_path = f"{dir_path}/{name}"
            self.mkdir(file_path)
            file_path = file_path + f"/{step}.pth"
            torch.save(obj.state_dict(), file_path)
        print("[*] Model saved at: {}".format(file_path))

    def load_model(self, dir_path, step=0):
        def load_torch_file(model_or_optim, _path):
            state_dict = torch.load(_path, map_location=lambda storage, loc: storage)
            model_or_optim.load_state_dict(state_dict)
        if self.args.use_att:
            name_obj_list = [
                ("eval_net", self.eval_net),
                ("target_net", self.target_net),
                ("att_net", self.att_weights_eval),
                ("att_net_target", self.att_weights_target),
            ]
        else:
            name_obj_list = [
                ("eval_net", self.eval_net),
                ("target_net", self.target_net),
            ]
        for name, obj in name_obj_list:
            file_path = f"{dir_path}/{name}/{step}.pth"
            if os.path.isfile(file_path):
                load_torch_file(obj, file_path)
            # break
                print("[*] Loaded model from {}".format(file_path))
            else:
                raise FileNotFoundError(f"No model found at {file_path}")
                

    def mkdir(self, directory):  # 当前传入： model/name
        if not os.path.exists(directory):
            os.makedirs(directory)

class RMFQ(base_rmfq.RMFQ_NET):
    def __init__(self, args):
        super().__init__(args=args)

        self.device = args.device
        self.train_ct = 0
        self.update_every = args.target_update_freq
        self.train_step = 0
        self.train_recl_freq = args.train_recl_freq
        
        # mapping = {0: 0, 1: 3, 2: 2, 3: 1, 4: 8, 5: 7, 6: 6, 7:5, 8:4, 9: 11, 11: 9, 10: 10, 12: 12, 13: 15, 15:13, 17:16, 16: 17, 18: 20, 20:18, 14: 14, 19: 19}
        # self.mapping_tensor = torch.tensor([mapping[i] for i in range(max(mapping.keys())+1)]).to(self.device)

    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)

    def train(self, replay_buffer):
                    
        for _ in range(self.args.multi_train_steps):

            batch_data, max_episode_len = replay_buffer.sample_random(self.args.batch_size)

            obs = batch_data['obs_n'].to(self.device)
            reward = batch_data['r'].to(self.device)
            act = batch_data['a_n'].to(self.device)
            batch_last_a = batch_data['last_onehot_a_n'].to(self.device)
            mask = batch_data['active'].to(self.device)
            done = batch_data['dw'].to(self.device)
            next_obs = batch_data['obs_next'].to(self.device)
            if self.args.use_att:
                global_role = batch_data['global_role'].to(self.device)
                last_act = batch_data['last_act'].to(self.device)
                next_role = batch_data['next_role'].to(self.device)
                next_mask = batch_data['active_next'].to(self.device)
                _, mean_action = self.cal_att_weights_mean_acts(obs, global_role, last_act, mask, key='eval')
                _, next_mean_action_e = self.cal_att_weights_mean_acts(next_obs, next_role, act, next_mask, key='eval')
                _, next_mean_action_t = self.cal_att_weights_mean_acts(next_obs, next_role, act, next_mask,key='target')
                next_mean_action = {'eval': next_mean_action_e, 'target': next_mean_action_t}
                
            else:
                mean_action = batch_data['mean_act'].to(self.device)
                next_mean_action = batch_data['mean_act_next'].to(self.device)

            if self.args.use_recl:
                if self.train_step % self.train_recl_freq == 0:
                    self.update_recl_nornn(obs, batch_last_a, mask, max_episode_len)
                    self.RECL.recl_soft_update(self.args.role_tau)

            loss = super().update_q_nornn(obs, act, next_obs, reward, done, mask, mean_action, next_mean_action, batch_last_a)
            self.soft_update()
            self.RECL.recl_soft_update(self.args.role_tau)

        self.train_step += 1
        return loss
    def soft_update_all_parms(self, model, tau):
        """_summary_

        Args:
            model (_type_):源模型,从model上获取参数
            tau (_type_): 软更新的参数
        """
        if self.args.use_att:
            model_params = {'eval_q_net':model.eval_net, 'target_q_net':model.target_net, 'RECL':model.RECL, 'att_net':model.att_weights_eval, 'att_net_target':model.att_weights_target}
            self_params = {'eval_q_net':self.eval_net, 'target_q_net':self.target_net, 'RECL':self.RECL, 'att_net':self.att_weights_eval, 'att_net_target':self.att_weights_target}
        else:
            model_params = {'eval_q_net':model.eval_net, 'target_q_net':model.target_net, 'RECL':model.RECL}
            self_params = {'eval_q_net':self.eval_net, 'target_q_net':self.target_net, 'RECL':self.RECL}
        for target, source in zip(self_params.values(), model_params.values()):
            for target_param, source_param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)
                
    

    def save_model(self, dir_path, step=0):
        if self.args.use_att:
            name_obj_list = [
                ("eval_net", self.eval_net),
                ("target_net", self.target_net),
                ("RECL", self.RECL),
                ("att_net", self.att_weights_eval),
                ("att_net_target", self.att_weights_target),
            ]
        else:
            name_obj_list = [
                ("eval_net", self.eval_net),
                ("target_net", self.target_net),
                ("RECL", self.RECL),
            ]
        for name, obj in name_obj_list:

            file_path = f"{dir_path}/{name}"
            self.mkdir(file_path)
            file_path = file_path + f"/{step}.pth"
            torch.save(obj.state_dict(), file_path)
        print("[*] Model saved at: {}".format(file_path))

    def load_model(self, dir_path, step=0):
        def load_torch_file(model_or_optim, _path):
            state_dict = torch.load(_path, map_location=lambda storage, loc: storage)
            model_or_optim.load_state_dict(state_dict)
        if self.args.use_att:
            name_obj_list = [
                ("eval_net", self.eval_net),
                ("target_net", self.target_net),
                ("RECL", self.RECL),
                ("att_net", self.att_weights_eval),
                ("att_net_target", self.att_weights_target),
            ]
        else:
            name_obj_list = [
                ("eval_net", self.eval_net),
                ("target_net", self.target_net),
                ("RECL", self.RECL),
            ]
        for name, obj in name_obj_list:
            file_path = f"{dir_path}/{name}/{step}.pth"
            if os.path.isfile(file_path):
                load_torch_file(obj, file_path)
            # break
                print("[*] Loaded model from {}".format(file_path))
            else:
                raise FileNotFoundError(f"No model found at {file_path}")

    def mkdir(self, directory):  # 当前传入： model/name
        if not os.path.exists(directory):
            os.makedirs(directory)

class MTMFQ(base_mtmfq.MTMFQ_NET):
    def __init__(self, args):
        super().__init__(args=args)

        self.device = args.device
        self.train_ct = 0
        self.update_every = args.target_update_freq
        self.train_step = 0

    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)

    def train(self, replay_buffer):

        for _ in range(self.args.multi_train_steps):
            batch_data, max_episode_len = replay_buffer.sample_random(self.args.batch_size)
                 
            obs = batch_data['obs_n'].to(self.device)
            reward = batch_data['r'].to(self.device)
            act = batch_data['a_n'].to(self.device)
            batch_last_a = batch_data['last_onehot_a_n'].to(self.device)
            mask = batch_data['active'].to(self.device)
            done = batch_data['dw'].to(self.device)
            next_obs = batch_data['obs_next'].to(self.device)
            mean_action = batch_data['mean_act'].to(self.device)
            next_mean_action = batch_data['mean_act_next'].to(self.device)
            
                    
            loss = super().update_q_nornn(obs, act, next_obs, reward, done, mask, mean_action, next_mean_action, max_episode_len)
            self.soft_update()

        self.train_step += 1
        return loss
    def soft_update_all_parms(self, model, tau):
        """soft-update

        Args:
            model (_type_):源模型,从model上获取参数
            tau (_type_): 软更新的参数
        """
        if self.args.use_att:
            model_params = {'eval_q_net':model.eval_net, 'target_q_net':model.target_net,  'att_net':model.att_weights_eval, 'att_net_target':model.att_weights_target}
            self_params = {'eval_q_net':self.eval_net, 'target_q_net':self.target_net,  'att_net':self.att_weights_eval, 'att_net_target':self.att_weights_target}
        else:
            model_params = {'eval_q_net':model.eval_net, 'target_q_net':model.target_net}
            self_params = {'eval_q_net':self.eval_net, 'target_q_net':self.target_net}
        for target, source in zip(self_params.values(), model_params.values()):
            for target_param, source_param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)
                
                
    
    def save_model(self, dir_path, step=0):
        name_obj_list = [
            ("eval_net", self.eval_net),
            ("target_net", self.target_net),
        ]
        for name, obj in name_obj_list:

            file_path = f"{dir_path}/{name}"
            self.mkdir(file_path)
            file_path = file_path + f"/{step}.pth"
            torch.save(obj.state_dict(), file_path)
        print("[*] Model saved at: {}".format(file_path))

    def load_model(self, dir_path, step=0):
        def load_torch_file(model_or_optim, _path):
            state_dict = torch.load(_path, map_location=lambda storage, loc: storage)
            model_or_optim.load_state_dict(state_dict)
        name_obj_list = [
            ("eval_net", self.eval_net),
            ("target_net", self.target_net),
        ]
        for name, obj in name_obj_list:
            file_path = f"{dir_path}/{name}/{step}.pth"
            if os.path.isfile(file_path):
                load_torch_file(obj, file_path)
            # break
                print("[*] Loaded model from {}".format(file_path))
            else:
                raise FileNotFoundError(f"No model found at {file_path}")
                

    def mkdir(self, directory):  # 当前传入： model/name
        if not os.path.exists(directory):
            os.makedirs(directory)