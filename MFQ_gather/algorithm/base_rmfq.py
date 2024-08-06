import torch
from kmeans_gpu import KMeans
import copy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from util.attention import MultiHeadAttention
from util.scores import compute_silhouette_score, max_silhouette_score

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
        self.obs_fc1 = nn.Linear(self.cnn_out_dim, self.agent_embedding_dim)
        
        # fc1,fc2平均动作经过线性层
        self.fc2 = nn.ModuleList([nn.Linear(self.act_dim, 64).to(args.device) for i in range(self.desc_dim)])
        self.fc3 = nn.ModuleList([nn.Linear(64, 32).to(args.device) for i in range(self.desc_dim)])

        self.hiden1 = nn.Linear(self.agent_embedding_dim+ self.role_embedding_dim+32*self.desc_dim, 128)
        self.hiden2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, self.act_dim)
        
        self.lamda = nn.Parameter(torch.rand(1)).to(args.device)
        self.gamma = nn.Parameter(torch.rand(1)).to(args.device)


    def cnn_forward(self, x):
        x = x.reshape(-1, self.in_channels, self.width, self.width)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(-1, self.cnn_out_dim)
        return x
    
    def agent_embedding_forward(self, obs, detach=False):
        obs = self.cnn_forward(obs)
        fc1_out = torch.relu(self.obs_fc1(obs))
        # fc2_out = self.obs_fc2(fc1_out)
        # if detach:
        #     fc2_out.detach()
        return fc1_out
   

    ## 不考虑顺序的输入
    ## 使用五数描述的输入
    def forward(self, x_agent, x_role, y):
        # 使用特征表示的平均动作是y.shape=(batch_size, desc_dim, action_dim)
        x_agent = x_agent.reshape(-1, self.agent_embedding_dim)
        x_role = x_role.reshape(-1, self.role_embedding_dim)
        x = torch.cat([x_agent, x_role], dim=1)
        y = y.reshape(-1, self.desc_dim, self.act_dim)
        y_out = torch.FloatTensor([]).to(x.device)
        for i in range(self.desc_dim):
            y_input = y[:,i]
            y_input = F.relu(self.fc3[i](F.relu(self.fc2[i](y_input))))
            y_out = torch.cat([y_out, y_input], dim=1)
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


class Agent_Embedding(nn.Module):
    def __init__(self, args):
        super(Agent_Embedding, self).__init__()
        self.use_last_action = args.use_action
        self.obs_dim = args.obs_dim
        self.act_dim = args.action_dim
        self.agent_embedding_dim = args.agent_embedding_dim
        self.role_embedding_dim = args.role_embedding_dim
        self.cluster_num = args.cluster_num
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
        self.act_net = nn.Linear(self.act_dim, 64)
        
        args.obs_dim_cnn = self.cnn_out_dim
        self.obs_fc1 = nn.Linear(self.cnn_out_dim, self.agent_embedding_dim)
        self.obs_fc2 = nn.Linear(self.agent_embedding_dim+64, self.agent_embedding_dim)

    def cnn_forward(self, x):
        x = x.reshape(-1, self.in_channels, self.width, self.width)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(-1, self.cnn_out_dim)
        return x
    
    def forward(self, obs, last_act, detach):
        obs = self.cnn_forward(obs)
        fc1_out = torch.relu(self.obs_fc1(obs))
        if self.use_last_action:
            act_out = torch.relu(self.act_net(last_act.reshape(-1, self.act_dim)))
            fc1_out = torch.cat([fc1_out, act_out], dim=1)
            fc2_out = self.obs_fc2(fc1_out)
            if detach:
                fc2_out.detach()
            return fc2_out
        else:
            if detach:
                fc1_out.detach()    
            return fc1_out

class Role_Embedding(nn.Module):
    def __init__(self, args):
        super(Role_Embedding, self).__init__()
        self.agent_embedding_dim = args.agent_embedding_dim
        self.role_embedding_dim = args.role_embedding_dim
        self.use_ln = args.use_ln

        if self.use_ln:     # 使用layer_norm
            self.role_embeding = nn.ModuleList([nn.Linear(self.agent_embedding_dim, self.role_embedding_dim),
                                                nn.LayerNorm(self.role_embedding_dim)])
        else:
            self.role_embeding = nn.Linear(self.agent_embedding_dim, self.role_embedding_dim)
    
    def forward(self, agent_embedding, detach=False):
        if self.use_ln:
            output = self.role_embeding[1](self.role_embeding[0](agent_embedding))
        else:
            output = self.role_embeding(agent_embedding)
        
        if detach:
            output.detach()
        output = torch.sigmoid(output)
        return output

class RECL_NET(nn.Module):
    def __init__(self, args):
        super(RECL_NET, self).__init__()

        if type(args.N) is dict:
            if args.target == args.key[0]:
                self.N = args.N[args.key[0]]
            else:
                self.N = args.N[args.key[1]]
        else:
            self.N = args.N
        self.agent_embedding_dim = args.agent_embedding_dim
        self.role_embedding_dim = args.role_embedding_dim
        self.action_dim = args.action_dim

        self.agent_embedding_net = Agent_Embedding(args)
        self.role_embedding_net = Role_Embedding(args)
        self.role_embedding_target_net = Role_Embedding(args)
        self.role_embedding_target_net.load_state_dict(self.role_embedding_net.state_dict())

        self.W = nn.Parameter(torch.rand(self.role_embedding_dim, self.role_embedding_dim))

    def forward(self, obs, last_a, detach=False):
        agent_embedding = self.agent_embedding_net(obs, last_a, detach=detach)
        role_embedding = self.role_embedding_net(agent_embedding)
        return agent_embedding, role_embedding

    def agent_embedding_forward(self, obs, act, detach=False):
        return self.agent_embedding_net(obs, act, detach)
    
    def role_embedding_forward(self, agent_embedding, detach=False, ema=False):
        if ema:
            output = self.role_embedding_target_net(agent_embedding, detach)
        else:
            output = self.role_embedding_net(agent_embedding)
        return output
    
    def batch_role_embed_forward(self, batch_o, net, max_episode_len, detach=False):
        net.rnn_hidden = None
        agent_embedding_net = net.agent_embedding_forward
        agent_embeddings = []
        for t in range(max_episode_len+1):    # t = 0,1,2...(max_episode_len-1), max_episode_len
            agent_embedding = self.agent_embedding_forward(batch_o[:, t],
                                          agent_embedding_net,
                                          detach=detach)  # agent_embedding.shape=(batch_size*N, agent_embed_dim)
            agent_embedding = agent_embedding.reshape(batch_o.shape[0], self.N, -1)  # shape=(batch_size,N, agent_embed_dim)
            agent_embeddings.append(agent_embedding.reshape(batch_o.shape[0],self.N, -1))
        # Stack them according to the time (dim=1)
        agent_embeddings = torch.stack(agent_embeddings, dim=1).reshape(-1,self.agent_embedding_dim) # agent_embeddings.shape=(batch_size*(max_episode_len+1)*N, agent_embed_dim)
        with torch.no_grad():
            role_embeddings = self.role_embedding_forward(agent_embeddings, detach=False, ema=False).reshape(-1, max_episode_len+1, self.N, self.role_embedding_dim)
        agent_embeddings = agent_embeddings.reshape(-1, max_episode_len+1, self.N, self.agent_embedding_dim)
        return agent_embeddings, role_embeddings
    
    def recl_soft_update(self, tau):
        for target_param, param in zip(self.role_embedding_target_net.parameters(), self.role_embedding_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class RMFQ_NET():
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
        # self.eval_net, self.target_net = Net(self.args).to(self.device), Net(args).to(self.device)
        self.eval_net, self.target_net = Net_nornn(self.args).to(self.device), Net_nornn(args).to(self.device)
        if args.use_att:
            self.att_weights_eval = att_net_weights(args, args.agent_embedding_dim, args.agent_embedding_dim, self.N).to(self.device)
            self.att_weights_target = att_net_weights(args, args.agent_embedding_dim, args.agent_embedding_dim, self.N).to(self.device)
            self.att_parameters = list(self.att_weights_eval.parameters())
            self.att_optimizer = torch.optim.Adam(self.att_parameters, lr=args.lr)
            for target_param, param in zip(self.att_weights_target.parameters(), self.att_weights_eval.parameters()):
                target_param.data.copy_(param.data)
        self.RECL = RECL_NET(self.args).to(self.device)
        self.RECL_optimizer = torch.optim.AdamW(self.RECL.parameters(), lr=args.lr, weight_decay=0.0001)  # 对比学习时更新role_embedding和w
        self.q_net_optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=args.lr)
        self.role_parameters = list(self.RECL.agent_embedding_net.parameters())+list(self.RECL.role_embedding_net.parameters())
        self.role_embedding_optimizer = torch.optim.AdamW(self.role_parameters, lr=args.lr, weight_decay=0.0001)  #eight_decay=0.001

        self.cluster_num = args.cluster_num
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
    
    def get_role_embedding(self, obs, last_a):
        # 角色嵌入
        obs_embedding, role_embedding = self.RECL(obs, last_a, detach=True)
        return obs_embedding, role_embedding
        
    def role_clustering(self, obs_embedding_n, act_embedding_n, last_center):
        # 角色聚类
        # if self.args.use_action:
        #     input_embedding_n = torch.cat([obs_embedding_n, act_embedding_n], dim=1)
            # input_embedding_n = input_embedding_n.unsqueeze(0)
        # else:
        input_embedding_n = obs_embedding_n
        input_embedding_n = input_embedding_n.unsqueeze(0)
        
        kmeans = KMeans(
            n_clusters=self.cluster_num,
            max_iter=100,
            tolerance=1e-4,
            distance='euclidean',
            sub_sampling=None,
            max_neighbors=15,
            )
        if not self.args.use_adaptive_k:
            centers, role_labels = kmeans(input_embedding_n, centroids=last_center)
        else:
            role_labels_all = []
            for k in range(2,10):
                kmeans.n_clusters = k
                centers, role_labels = kmeans(input_embedding_n, centroids=last_center)
                role_labels_all.append(role_labels.squeeze())
            role_labels_all = torch.stack(role_labels_all, dim=0)
            x = input_embedding_n.expand(role_labels_all.shape[0], -1, -1).detach()
            scores = compute_silhouette_score(x, role_labels_all)
            best_k = torch.argmax(scores)
            role_labels = role_labels_all[best_k]

        
        return role_labels
    
    def act(self, obs, role_embedding, mean_action, eps):
        # 动作选择
        # eps-greedy探索
        if np.random.uniform() < eps:
            return torch.randint(self.args.action_dim, (self.N,))
        else:
            cluster_num = 9 if self.args.use_adaptive_k else self.args.cluster_num
            mean_action = torch.FloatTensor(mean_action).to(self.device)
            mean_action = mean_action.reshape(-1, cluster_num, self.args.action_dim)
            _, mean_action_desc = get_mean_action_description(mean_action)
            q_values = self.eval_net(obs, role_embedding, mean_action_desc)
            predict = F.softmax(q_values, dim=-1)
            actions = torch.argmax(predict, axis=1)
        
            return actions
    
    
    def calc_target_q_nornn(self, obs_next, rewards, dones, mean_act_next, act):
        # with torch.no_grad():
        act_one_hot = F.one_hot(act.to(torch.int64), num_classes=self.args.action_dim).reshape(-1, self.args.action_dim)
        obs_input = self.RECL.agent_embedding_forward(obs_next, act_one_hot.float())
        if self.args.use_recl:
            role_input_t = self.RECL.role_embedding_forward(obs_input, ema=True)
            role_input_e = self.RECL.role_embedding_forward(obs_input)
        else:
            role_input_t = torch.zeros(len(obs_input), self.args.role_embedding_dim).to(self.device)
            role_input_e = torch.zeros(len(obs_input), self.args.role_embedding_dim).to(self.device)
        obs_input_t = self.target_net.agent_embedding_forward(obs_next)
        obs_input_e = self.eval_net.agent_embedding_forward(obs_next)
        if self.args.use_att:
            t_q = self.target_net.forward(obs_input_t, role_input_t, mean_act_next['target'])
            e_q = self.eval_net.forward(obs_input_e, role_input_e, mean_act_next['eval'])
        else:
            t_q = self.target_net.forward(obs_input_t, role_input_t, mean_act_next)
            e_q = self.eval_net.forward(obs_input_e, role_input_e, mean_act_next)
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
    
    
    def update_q_nornn(self, obs, act, next_obs, reward, done, mask, mean_action:torch.FloatTensor, next_mean_action, last_act):
        # 更新Q网络
        loss = 0
        
        # act_one_hot = F.one_hot(act.to(torch.int64), num_classes=self.args.action_dim).reshape(-1, self.args.action_dim)
        
        obs = obs.reshape(-1, self.args.obs_dim[0], self.args.obs_dim[1], self.args.obs_dim[2])   
        next_obs = next_obs.reshape(-1, self.args.obs_dim[0], self.args.obs_dim[1], self.args.obs_dim[2])
        reward = reward.reshape(-1)
        done = done.reshape(-1)
        mask = mask.reshape(-1)
        act = act.reshape(-1)
        last_act = last_act.reshape(-1)
        cluster_num = 9 if self.args.use_adaptive_k else self.args.cluster_num
        if not self.args.use_att:
            mean_action = mean_action.reshape(-1, cluster_num, self.args.action_dim)
            next_mean_action = next_mean_action.reshape(-1, cluster_num, self.args.action_dim)
            _, mean_action_desc = get_mean_action_description(mean_action)
            _, next_mean_action_desc = get_mean_action_description(next_mean_action)
        else:
            next_mean_action_desc = {}
            mean_action = mean_action.reshape(-1, cluster_num, self.args.action_dim)
            next_mean_action['eval'] = next_mean_action['eval'].reshape(-1, cluster_num, self.args.action_dim)
            next_mean_action['target'] = next_mean_action['target'].reshape(-1, cluster_num, self.args.action_dim)
            _, mean_action_desc = get_mean_action_description(mean_action)
            _, next_mean_action_desc['eval'] = get_mean_action_description(next_mean_action['eval'])
            _, next_mean_action_desc['target'] = get_mean_action_description(next_mean_action['target'])
        obs_embedding_es = self.eval_net.agent_embedding_forward(obs)
        
        if self.args.use_recl:
            obs_embedding_recl = self.RECL.agent_embedding_forward(obs, last_act, detach=False)
            # with torch.no_grad():
            role_embeddings = self.RECL.role_embedding_forward(obs_embedding_recl, detach=False, ema=False)
        else:
            role_embeddings = torch.zeros(len(obs), self.args.role_embedding_dim).to(self.device)
        
        q_targets = self.calc_target_q_nornn(next_obs, reward, done, next_mean_action_desc, act)
        q_eval = self.eval_net(obs_embedding_es, role_embeddings, mean_action_desc)
        # q_eval_max = torch.mul(act_one_hot, q_eval).sum(axis=1)  # torch.gather
        q_eval_max = torch.gather(q_eval, 1, act.unsqueeze(1)).squeeze(1)
        loss = torch.sum(torch.square(q_targets - q_eval_max)*mask.reshape(-1))/torch.sum(mask)
        
        self.q_net_optimizer.zero_grad()
        
        if self.args.use_recl:
            self.role_embedding_optimizer.zero_grad()
        if self.args.use_att:
            self.att_optimizer.zero_grad()
            
        loss.backward()
        if torch.isnan(loss).any():
            print('loss is nan')
        nn.utils.clip_grad_norm_(self.eval_net.parameters(), max_norm=10)
        self.q_net_optimizer.step()
        

        if self.args.use_recl:
            nn.utils.clip_grad_norm_(self.role_parameters, max_norm=10)
            self.role_embedding_optimizer.step()
        if self.args.use_att:
            nn.utils.clip_grad_norm_(self.att_parameters, max_norm=10)
            self.att_optimizer.step()

        return loss.item()
    
    def update_recl_nornn(self, batch_o, batch_act, batch_active, max_episode_len):
        """
        N = agent_num
        batch_o.shape = (batch_size, N,  obs_dim)
        batch_a.shape = (batch_size, N,  action_dim)
        batch_active = (batch_size, 1)
        """
        dk = self.args.role_embedding_dim
        loss = 0
        labels = np.zeros((batch_o.shape[0], self.N))  # (batch_size, N)
        kmeans = KMeans(
            n_clusters=self.cluster_num,
            max_iter=100,
            tolerance=1e-4,
            distance='euclidean',
            sub_sampling=None,
            max_neighbors=15,
            )
        with torch.no_grad():
            agent_embedding = self.RECL.agent_embedding_forward(batch_o, batch_act, detach=True)  # agent_embedding.shape=(batch_size, N, agent_embed_dim)
        role_embedding_qury = self.RECL.role_embedding_forward(agent_embedding,
                                                                detach=False,
                                                                ema=False).reshape(-1,self.N, self.role_embedding_dim)   # shape=(batch_size, N, role_embed_dim)
        role_embedding_key = self.RECL.role_embedding_forward(agent_embedding,
                                                                detach=True,
                                                                ema=True).reshape(-1,self.N, self.role_embedding_dim)
        logits0 = torch.bmm(role_embedding_qury, self.RECL.W.squeeze(0).expand((role_embedding_qury.shape[0],self.role_embedding_dim,self.role_embedding_dim)))
        logits1 = torch.bmm(logits0, role_embedding_key.transpose(1,2)) / dk   # (batch_size, N, N)
        logits = logits1 - torch.max(logits1, dim=-1)[0][:,:,None]
        exp_logits = torch.exp(logits) # (batch_size, N, N)
        agent_embedding = agent_embedding.reshape(batch_o.shape[0],self.N, -1)  # shape=(batch_size,N, agent_embed_dim)
        
        
        if not self.args.use_adaptive_k:
            mask = torch.sum(batch_active, dim=1) >= self.cluster_num
            if mask.sum() > 0:
                centroids, clusters_labels = kmeans(agent_embedding[mask])
                labels = copy.deepcopy(clusters_labels)
            else:
                clusters_labels = labels

            idx = 0
            exp_logits_copy = exp_logits.clone()
            for k in range(agent_embedding.shape[0]): 
                if mask[k]:
                    for j in range(self.cluster_num):
                        label_pos = ((clusters_labels[idx] == j) & (batch_active[idx]==1)).nonzero().reshape(-1)
                        active_pos = (batch_active[idx] == 0).nonzero().reshape(-1)
                        exp_logits_copy[idx,:,active_pos] = 0
                        anchor_exp_logits = exp_logits_copy[idx,label_pos]
                        
                        loss += -torch.log(torch.clip((anchor_exp_logits[:, label_pos].sum(dim=1) / anchor_exp_logits.sum(dim=1)), min=1e-30)).sum()
                        if torch.isinf(-torch.log(torch.clip((anchor_exp_logits[:, label_pos].sum(dim=1) / anchor_exp_logits.sum(dim=1)), min=1e-30)).sum()):
                            print('reloss is inf')
                    idx += 1
        else:
            labels = torch.zeros((batch_o.shape[0], 8, self.N), dtype=torch.long).to(self.device)
            mask = torch.sum(batch_active, dim=1) >= 9
            if mask.sum() > 0:
                for k in range(2,10):
                    kmeans.n_clusters = k
                    _, clusters_labels = kmeans(agent_embedding[mask])
                    labels[mask, k-2] = clusters_labels
                max_labels = max_silhouette_score(agent_embedding[mask].unsqueeze(1).expand(-1, labels.shape[1], -1, -1), labels[mask])   
            
            idx = 0
            exp_logits_copy = exp_logits.clone()
            for k in range(agent_embedding.shape[0]): 
                if mask[k]:
                    for j in range(torch.max(max_labels[idx])+1):
                        label_pos = ((max_labels[idx] == j) & (batch_active[idx]==1)).nonzero().reshape(-1)
                        active_pos = (batch_active[idx] == 0).nonzero().reshape(-1)
                        exp_logits_copy[idx,:,active_pos] = 0
                        anchor_exp_logits = exp_logits_copy[idx,label_pos]
                        
                        loss += -torch.log(torch.clip((anchor_exp_logits[:, label_pos].sum(dim=1) / anchor_exp_logits.sum(dim=1)), min=1e-30)).sum()
                        if torch.isinf(-torch.log(torch.clip((anchor_exp_logits[:, label_pos].sum(dim=1) / anchor_exp_logits.sum(dim=1)), min=1e-30)).sum()):
                            print('reloss is inf')
                    idx += 1            
        
        if torch.sum(batch_active) and mask.sum() and loss != 0:
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print('recl loss is nan')   
            loss /= torch.sum(batch_active)
            self.RECL_optimizer.zero_grad()
            loss.backward()
            
            nn.utils.clip_grad_norm_(self.RECL.parameters(), max_norm=10)
            self.RECL_optimizer.step()

        return loss
    
    def cal_att_weights_mean_acts(self, obs, role_global, act_global, mask, key = 'eval'):
        if self.args.use_adaptive_k:
            cluster_num = 9
        else:
            cluster_num = self.args.cluster_num
        if self.args.use_att:
            if key == 'eval':
                obs_embeddings = self.eval_net.agent_embedding_forward(obs)
                weights = self.att_weights_eval(obs_embeddings, obs_embeddings)
            elif key == 'target':
                obs_embeddings = self.target_net.agent_embedding_forward(obs).detach()
                weights = self.att_weights_target(obs_embeddings, obs_embeddings)
            mask_weights = weights.unsqueeze(1).repeat(1, cluster_num, 1, 1)
            for i in range(cluster_num):
                idx = torch.where(((role_global.to(torch.int64) == i) & mask.to(torch.int64) !=0), torch.tensor(1).to(self.device), torch.tensor(-1000).to(self.device)).unsqueeze(1).repeat(1, self.N, 1)
                mask_weights[:, i] = torch.mul(idx.float(), mask_weights[:, i])
            soft_max_weights = F.softmax(mask_weights, dim=-1)
            act = F.one_hot(act_global, self.args.action_dim).to(torch.float32).unsqueeze(1).expand(-1, cluster_num, -1, -1)
            mean_a = torch.matmul(soft_max_weights, act).transpose(1,2)
        else:
            mask_weights = torch.ones(len(obs), cluster_num, self.N, self.N).to(self.device)
            for i in range(cluster_num):
                idx = torch.where(((role_global.to(torch.int64) == i) & mask.to(torch.int64) !=0), torch.tensor(1).to(self.device), torch.tensor(-1000).to(self.device)).unsqueeze(1).repeat(1, self.N, 1)
                mask_weights[:, i] = torch.mul(idx.float(), mask_weights[:, i])
            soft_max_weights = F.softmax(mask_weights, dim=-1)
            act = F.one_hot(act_global, self.args.action_dim).to(torch.float32).unsqueeze(1).expand(-1, cluster_num, -1, -1)
            mean_a = torch.matmul(soft_max_weights, act).transpose(1,2)
        if self.args.use_adaptive_k:
            role_global_max = role_global.max(dim=1, keepdim=True)[0]+1
            comparison_tensor = torch.arange(9).view(1, 1, mean_a.shape[2], 1).expand(mean_a.shape[0], mean_a.shape[1], mean_a.shape[2], mean_a.shape[3]).to(role_global.device)
            mask_role = (comparison_tensor < role_global_max.view(mean_a.shape[0], 1, 1, 1)).float()
            mask_role2 = (comparison_tensor >= role_global_max.view(mean_a.shape[0], 1, 1, 1)).float()
            mean_a = torch.mul(mean_a, mask_role)-mask_role2
        return soft_max_weights, mean_a
    
def get_mean_action_description(one_hot_action):
    """获得平均动作描述量,初步考虑使用各维的五数描述+平均值+方差

    Args:
        one_hot_action (torch.tensor): 经过one-hot编码的平均动作; shape=(batch_size, cluster_num, action_dim)

    Returns:
        desc_dim(int): 描述量的维度
        mean_action_desc(torch.Tensor): 平均动作描述量; shape=(batch_size, desc_dim, action_dim)
    """
    min = torch.min(one_hot_action)
    if min > -1:
        # 计算五数概括
        min_val, _ = torch.min(one_hot_action, dim=1)
        q1 = torch.quantile(one_hot_action, 0.25, dim=1)
        median = torch.median(one_hot_action, dim=1).values
        q3 = torch.quantile(one_hot_action, 0.75, dim=1)
        max_val, _ = torch.max(one_hot_action, dim=1)

        # 计算平均值和方差
        mean = torch.mean(one_hot_action, dim=1)
        var = torch.var(one_hot_action, dim=1)
    else:
        device = one_hot_action.device
        mask = one_hot_action != -1
        data_masked = torch.where(mask, one_hot_action, torch.tensor(float('nan')).to(device))
        q1 = torch.nanquantile(data_masked, 0.25, dim=1, interpolation='linear', keepdim=True).squeeze(1)
        median = torch.nanquantile(data_masked, 0.5, dim=1, interpolation='linear', keepdim=True).squeeze(1)
        q3 = torch.nanquantile(data_masked, 0.75, dim=1, interpolation='linear', keepdim=True).squeeze(1)
        
        max_val, _ = torch.max(torch.where(mask, one_hot_action, torch.tensor(float('-inf')).to(device)), dim=1)
        min_val, _ = torch.min(torch.where(mask, one_hot_action, torch.tensor(float('inf')).to(device)), dim=1)
        
        # 计算平均数和方差
        mean = torch.sum(one_hot_action * mask.float(), dim=1) / torch.sum(mask, dim=1).float()
        var = torch.sum((one_hot_action - mean.unsqueeze(1))**2 * mask.float(), dim=1) / torch.sum(mask, dim=1).float()

    # 将所有描述符合并到一个张量中
    mean_action_desc = torch.stack([min_val, q1, median, q3, max_val, mean, var], dim=1)
    if torch.isnan(mean_action_desc).any():
        print("desc is nan")
    desc_dim = mean_action_desc.shape[1]
    
    return desc_dim, mean_action_desc