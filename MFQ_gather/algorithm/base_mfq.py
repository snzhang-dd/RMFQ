import torch
from kmeans_gpu import KMeans
import copy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from util.attention import MultiHeadAttention

class Net_nornn(nn.Module):
    def __init__(self, args):
        super(Net_nornn, self).__init__()
        self.obs_dim = args.obs_dim
        self.act_dim = args.action_dim
        self.agent_embedding_dim = args.agent_embedding_dim
        self.role_embedding_dim = args.role_embedding_dim
        self.cluster_num = args.cluster_num
        self.desc_dim = args.desc_dim
        self.width = self.obs_dim[2]
        self.in_channels = self.obs_dim[0]
        self.padding = int((args.cnn_kernel_size-1) / 2)  # 卷积前后图片长宽不变
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=32, kernel_size=args.cnn_kernel_size, stride=1),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=args.cnn_kernel_size, stride=1),
            nn.ReLU(),
        )
        self.cnn_out_dim = (self.width-4) * (self.width-4) * 32
        
        args.obs_dim_cnn = self.cnn_out_dim

        self.obs_fc1 = nn.Linear(self.cnn_out_dim, self.agent_embedding_dim)
        
        # fc1,fc2平均动作经过线性层
        self.fc2 = nn.Linear(self.act_dim, 64).to(args.device)
        self.fc3 = nn.Linear(64, 32).to(args.device)
        self.hiden1 = nn.Linear(self.agent_embedding_dim+ 32, 128)
        self.hiden2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, self.act_dim)
        

    def cnn_forward(self, x):
        x = x.reshape(-1, self.in_channels, self.width, self.width)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(-1, self.cnn_out_dim)
        return x
    
    def agent_embedding_forward(self, obs):
        obs = self.cnn_forward(obs)
        fc1_out = torch.relu(self.obs_fc1(obs))
        return fc1_out
   
    def forward(self, x_agent, y):
        x_agent = x_agent.reshape(-1, self.agent_embedding_dim)
        x = x_agent
        y = y.reshape(-1, self.act_dim)
        y_out = F.relu(self.fc3(F.relu(self.fc2(y))))
        
        input = torch.cat([x, y_out], dim=1)
        xx = F.relu(self.hiden2(F.relu(self.hiden1(input))))
        actions_value = self.out(xx)
        return actions_value


class att_net_weights(nn.Module):
    def __init__(self, args, dim_q, dim_k, N):
        super(att_net_weights, self).__init__()
        self.att_dim = args.att_dim
        self.N = N
        self.fc_q = nn.Linear(dim_q, self.att_dim, bias=False)
        self.fc_k = nn.Linear(dim_k, self.att_dim, bias=False)
        
    def forward(self, x_q, x_k):
        
        x_q = x_q.reshape(-1, self.N, x_q.shape[-1])
        x_k = x_k.reshape(-1, self.N, x_k.shape[-1])
        q = self.fc_q(x_q)
        k = self.fc_k(x_k).transpose(1, 2)
        weights = torch.matmul(q, k)/np.sqrt(self.att_dim)
        
        return weights


class MFQ_NET():
    # 增加对比学习网络，Q网络结构更改为多角色的平均动作为输入
    def __init__(self, args):
        self.args = args
        self.device = args.device
        if type(args.N) is dict:
            if args.target == args.key[0]:
                self.N = args.N[args.key[0]]
            else:
                self.N = args.N[args.key[1]]
        else:
            self.N = args.N
        self.eval_net, self.target_net = Net_nornn(self.args).to(self.device), Net_nornn(args).to(self.device)
        if args.use_att:
            self.att_weights_eval = att_net_weights(args, args.agent_embedding_dim, args.agent_embedding_dim, self.N).to(self.device)
            self.att_weights_target = att_net_weights(args, args.agent_embedding_dim, args.agent_embedding_dim, self.N).to(self.device)
            self.att_parameters = list(self.att_weights_eval.parameters())
            self.att_optimizer = torch.optim.Adam(self.att_parameters, lr=args.lr)
            for target_param, param in zip(self.att_weights_target.parameters(), self.att_weights_eval.parameters()):
                target_param.data.copy_(param.data)
        self.q_net_optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=args.lr)

        for target_param, param in zip(self.target_net.parameters(), self.eval_net.parameters()):
            target_param.data.copy_(param.data)
        
        self.tau = args.tau
        self.lr = args.lr
        self.gamma = args.gamma
        
        if type(args.N) is dict:
            if args.target == args.key[0]:
                self.N = args.N[args.key[0]]
            else:
                self.N = args.N[args.key[1]]
        else:
            self.N = args.N
            
        self.role_embedding_dim = args.role_embedding_dim
        self.batch_size = args.batch_size
        
    def get_agent_embedding(self, obs):
        # 观测嵌入
        return self.eval_net.agent_embedding_forward(obs)
        
    
    def act(self, obs, mean_action, eps):
        # 动作选择
        # eps-greedy探索
        if np.random.uniform() < eps:
            return torch.randint(self.args.action_dim, (self.N,))
        else:
            mean_action = torch.FloatTensor(mean_action).to(self.device)
            mean_action = mean_action.reshape(-1, self.args.cluster_num, self.args.action_dim)
            q_values = self.eval_net(obs, mean_action)
            predict = F.softmax(q_values, dim=-1)
            actions = torch.argmax(predict, axis=1)
            return actions
    
    def calc_target_q_nornn(self, obs_next, rewards, dones, mean_act_next):
        # with torch.no_grad():
        obs_input_t = self.target_net.agent_embedding_forward(obs_next)
        obs_input_e = self.eval_net.agent_embedding_forward(obs_next)
        if self.args.use_att:
            t_q = self.target_net.forward(obs_input_t, mean_act_next['target'])
            e_q = self.eval_net.forward(obs_input_e, mean_act_next['eval'])
        else:
            t_q = self.target_net.forward(obs_input_t, mean_act_next)
            e_q = self.eval_net.forward(obs_input_e, mean_act_next)
        act_idx = torch.argmax(e_q, axis=1)
        # q_values = t_q[np.arange(len(t_q)), act_idx]  # torch.gather
        q_values = torch.gather(t_q, 1, act_idx.unsqueeze(1)).squeeze(1)
        target_q_value = rewards + (1. - dones) * q_values.reshape(-1) * self.gamma
        return target_q_value.detach()
    
    def soft_update(self):
        for target_param, param in zip(self.target_net.parameters(), self.eval_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        if self.args.use_att:
            for target_param, param in zip(self.att_weights_target.parameters(), self.att_weights_eval.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    
    def update_q_nornn(self, obs, act, next_obs, reward, done, mask, mean_action:torch.FloatTensor, next_mean_action, max_episode_len):
        # 更新Q网络
        loss = 0
        
        # act_one_hot = F.one_hot(act.to(torch.int64), num_classes=self.args.action_dim).reshape(-1, self.args.action_dim)
        
        obs = obs.reshape(-1, self.args.obs_dim[0], self.args.obs_dim[1], self.args.obs_dim[2])   
        next_obs = next_obs.reshape(-1, self.args.obs_dim[0], self.args.obs_dim[1], self.args.obs_dim[2])
        reward = reward.reshape(-1)
        done = done.reshape(-1)
        mask = mask.reshape(-1)
        act = act.reshape(-1)
        mean_action = mean_action.reshape(-1, self.args.cluster_num, self.args.action_dim)
        if not self.args.use_att:
            next_mean_action = next_mean_action.reshape(-1, self.args.cluster_num, self.args.action_dim)
        else:
            next_mean_action['eval'] = next_mean_action['eval'].reshape(-1, self.args.cluster_num, self.args.action_dim)
            next_mean_action['target'] = next_mean_action['target'].reshape(-1, self.args.cluster_num, self.args.action_dim)
        obs_embedding_es = self.eval_net.agent_embedding_forward(obs)
        
        q_targets = self.calc_target_q_nornn(next_obs, reward, done, next_mean_action)
        q_eval = self.eval_net(obs_embedding_es, mean_action)
        # q_eval_max = torch.mul(act_one_hot, q_eval).sum(axis=1)  # torch.gather
        q_eval_max = torch.gather(q_eval, 1, act.unsqueeze(1)).squeeze(1)
        loss = torch.sum(torch.square(q_targets - q_eval_max)*mask.reshape(-1))/torch.sum(mask)
        
        self.q_net_optimizer.zero_grad()
        
        if self.args.use_att:
            self.att_optimizer.zero_grad()
            
        loss.backward()
        
        self.q_net_optimizer.step()

        if self.args.use_att:
            self.att_optimizer.step()
        
        return loss.item()
    
    def cal_att_weights_mean_acts(self, obs, role_global, act_global, mask, key = 'eval'):
        if self.args.use_att:
            if key == 'eval':
                obs_embeddings = self.eval_net.agent_embedding_forward(obs)
                weights = self.att_weights_eval(obs_embeddings, obs_embeddings)
            elif key == 'target':
                obs_embeddings = self.target_net.agent_embedding_forward(obs).detach()
                weights = self.att_weights_target(obs_embeddings, obs_embeddings)
            mask_weights = weights.unsqueeze(1).repeat(1, self.args.cluster_num, 1, 1)
            for i in range(self.args.cluster_num):
                idx = torch.where(((role_global.to(torch.int64) == i) & mask.to(torch.int64) !=0), torch.tensor(1).to(self.device), torch.tensor(-1000).to(self.device)).unsqueeze(1).repeat(1, self.N, 1)
                mask_weights[:, i] = torch.mul(idx.float(), mask_weights[:, i])
            soft_max_weights = F.softmax(mask_weights, dim=-1)
            act = F.one_hot(act_global, self.args.action_dim).to(torch.float32).unsqueeze(1).expand(-1, self.args.cluster_num, -1, -1)
            mean_a = torch.matmul(soft_max_weights, act).transpose(1,2)
        else:
            mask_weights = torch.ones(len(obs), self.args.cluster_num, self.N, self.N).to(self.device)
            for i in range(self.args.cluster_num):
                idx = torch.where(((role_global.to(torch.int64) == i) & mask.to(torch.int64) !=0), torch.tensor(1).to(self.device), torch.tensor(-1000).to(self.device)).unsqueeze(1).repeat(1, self.N, 1)
                mask_weights[:, i] = torch.mul(idx.float(), mask_weights[:, i])
            soft_max_weights = F.softmax(mask_weights, dim=-1)
            act = F.one_hot(act_global, self.args.action_dim).to(torch.float32).unsqueeze(1).expand(-1, self.args.cluster_num, -1, -1)
            mean_a = torch.matmul(soft_max_weights, act).transpose(1,2)
        
        return soft_max_weights, mean_a