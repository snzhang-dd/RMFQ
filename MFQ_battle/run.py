import torch
import numpy as np
from algorithm.q_learning import MFQ, RMFQ, MTMFQ
from util.replay_buffer import ReplayBuffer
import seaborn as sns
import matplotlib.pyplot as plt
import time
import os
import sys
CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]  # 当前目录
config_path = CURRENT_DIR.rsplit('/', 1)[0]  # 上三级目录
sys.path.append(config_path)
from env.magent2.magent_env import MAgent_Env
import datetime
from tensorboardX import SummaryWriter
import time
import copy
import random
from kmeans_gpu import KMeans

class Runner:
    def __init__(self, args):
        self.args = args
        self.env_name = self.args.env_name
        self.seed = self.args.seed
        # Set random seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        # env_seed = random.randint(0, 100000)
        # Create env
        self.env = MAgent_Env(self.env_name, self.seed, max_cycles=args.max_step, map_size=args.map_size, minimap_mode=True)
        self.env.reset()
        self.key = self.env.side_names
        
        self.N_group = len(self.key)
        self.cluster_num = self.args.cluster_num
        self.agents = self.env.agents
        self.args.key = self.key
        self.args.N = {self.key[i]:self.env.n_agents[i] for i in range(self.N_group)}  # The number of agents
        obs_shape = self.env.observation_space(self.env.agents[0]).shape
        self.args.obs_dim = (7, obs_shape[0], obs_shape[1])
        self.args.state_dim = self.env.state_space.shape  # The dimensions of global state space
        # self.args.state_dim = self.args.state_dim[0] * self.args.state_dim[1] * self.args.state_dim[2]  # The dimensions of global state space
        self.args.action_dim = self.env.action_space(self.env.agents[0]).n  # The dimensions of an agent's action space
        self.args.episode_limit = self.env.max_cycles  # Maximum number of steps per episode
        print("number of agents={}".format(self.args.N))
        print("obs_dim={}".format(self.args.obs_dim))
        print("state_dim={}".format(self.args.state_dim))
        print("action_dim={}".format(self.args.action_dim))
        print("episode_limit={}".format(self.args.episode_limit))
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        
        self.save_path =  args.save_path + '/' + args.algorithm +'/'+ args.env_name + '/'
        #检查路径是否存在
        str_result = datetime.datetime.now().strftime('%m-%d-%H-%M')+'_cn'+str(self.args.cluster_num)+'_re'+str(self.args.use_recl)+'_att'+str(self.args.use_att)+'_seed'+str(self.seed)+'adap_k'+str(self.args.use_adaptive_k)+'l2=0.0001_64'+self.args.algorithm+'map_size'+str(self.args.map_size)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        self.model_path = {self.key[i]:self.save_path + '/model/' +str_result + '/'+ self.key[i] + '/' for i in range(self.N_group)}
        for i in range(self.N_group):
            if not os.path.exists(self.model_path[self.key[i]]):
                os.makedirs(self.model_path[self.key[i]])
            
        self.writer = SummaryWriter(self.save_path + '/log/' + str_result)

        # Create N agents
        self.agent_n = {}
        self.replay_buffer = {}
        if args.algorithm in ['MFQ']:
            for i in range(self.N_group):
                self.args.target = self.key[i]
                self.agent_n.update({self.key[i]:MFQ(self.args)})
        if args.algorithm in ['MTMFQ']:
            for i in range(self.N_group):
                self.args.target = self.key[i]
                self.agent_n.update({self.key[i]:MTMFQ(self.args)})
        elif args.algorithm == 'RMFQ':
            for i in range(self.N_group):
                self.args.target = self.key[i]
                self.agent_n.update({self.key[i]:RMFQ(self.args)})
            self.agent_temp = RMFQ(self.args)
            
        for i in range(self.N_group):
            self.args.target = self.key[i]
            self.replay_buffer.update({self.key[i]:ReplayBuffer(self.args, self.args.buffer_size)})

        if args.model_path:
            for i in range(self.N_group):
                self.agent_n[self.key[i]].load_model(args.model_path[0]+self.key[i], args.model_path[1])
        
        self.epsilon = self.args.epsilon  # Initialize the epsilon
        self.win_rates = {self.key[i]:[] for i in range(self.N_group)}  # Record the win rates
        self.evaluate_reward = {self.key[i]:[] for i in range(self.N_group)}
        self.total_steps = 0
        self.total_episodes = 0
        self.train_every_steps = self.args.train_every_steps
        self.train_steps = 0
        self.agent_embed_pretrain_epoch, self.recl_pretrain_epoch = 0, 0
        self.pretrain_agent_embed_loss, self.pretrain_recl_loss = {self.key[i]:[] for i in range(self.N_group)}, {self.key[i]:[] for i in range(self.N_group)}

        self.args.agent_embed_pretrain_epochs =0
        self.args.recl_pretrain_epochs = 0
        
        self.center = {self.key[i]:None for i in range(self.N_group)}
        

    def run(self, ):
        evaluate_num = -1  # Record the number of evaluations
        train_num = 0
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
                evaluate_num += 1
            # t1 = time.time()
            _, rew, episode_steps = self.run_episode_rmfq(evaluate=False)  # Run an episode

            for i in range(self.N_group):
                self.writer.add_scalar('episode_reward_'+self.key[i], rew[self.key[i]], self.total_steps)
                                                    
            self.total_steps += episode_steps
            self.total_episodes += 1
            self.train_steps += episode_steps
            if self.replay_buffer[self.key[0]].current_size*200 >= self.args.batch_size and self.train_steps >= self.train_every_steps:
                # loss = self.agent_n[self.key[1]].train(self.replay_buffer[self.key[0]])  # Training
                # t3 = time.time()
                loss = self.agent_n[self.key[0]].train(self.replay_buffer[self.key[0]])  # Training
                self.writer.add_scalar('mfq_loss_'+self.key[0], loss, self.total_steps)
                self.train_steps = 0
                train_num += 1
                # print("train_time", time.time()-t3)
            if rew[self.key[0]]>=rew[self.key[1]]:
                self.agent_n[self.key[1]].soft_update_all_parms(self.agent_n[self.key[0]], self.args.tau)

            if self.total_episodes % self.args.save_rate == 0:
                for i in range(self.N_group):
                    self.agent_n[self.key[i]].save_model(self.model_path[self.key[i]], self.total_steps)

        self.evaluate_policy()
        self.env.close()
        
    def evaluate_policy(self, ):
        win_times = {self.key[i]:0 for i in range(self.N_group)}
        evaluate_reward = {self.key[i]:0 for i in range(self.N_group)}
        for _ in range(self.args.evaluate_times):
            win_tag, episode_reward, _ = self.run_episode_rmfq(evaluate=True)
            win_times = {self.key[i]: win_times.get(self.key[i], 0) + 1 if win_tag[self.key[i]] else win_times.get(self.key[i], 0) for i in range(self.N_group)}
            evaluate_reward = {self.key[i]:evaluate_reward[self.key[i]]+episode_reward[self.key[i]] for i in range(self.N_group)}

        win_rate = {self.key[i]:win_times[self.key[i]] / self.args.evaluate_times for i in range(self.N_group)}
        for i in range(self.N_group):
            self.win_rates[self.key[i]].append(win_rate[self.key[i]])
            evaluate_reward[self.key[i]] = evaluate_reward[self.key[i]] / self.args.evaluate_times
            self.evaluate_reward[self.key[i]].append(evaluate_reward[self.key[i]])
        print("total_steps:{} \t win_rate:{} \t evaluate_reward:{}".format(self.total_steps, win_rate, evaluate_reward))

        for i in range(self.N_group):
            self.writer.add_scalar('win_rate_'+self.key[i], win_rate[self.key[i]], self.total_episodes)
            self.writer.add_scalar('evaluate_reward_'+self.key[i], evaluate_reward[self.key[i]], self.total_episodes)
        
        
    def run_episode_rmfq(self, evaluate=False):
        win_tag = {self.key[i]:False for i in range(self.N_group)}
        episode_reward = {self.key[i]:0 for i in range(self.N_group)}
        obs_n, info = self.env.reset()
        obs_n_dict = {self.key[i]:{key:value for key, value in obs_n.items() if self.key[i] in key} for i in range(self.N_group)}
        obs_n_array = {self.key[i]:np.array(list(obs_n_dict[self.key[i]].values())).transpose(0, 3, 1, 2)[:,0:7] for i in range(self.N_group)}
        state = self.env.state()
        position = self.get_agent_position(obs_n)
        index_n_temp = [0]
        for i in range(self.N_group):
            index_n_temp.append(index_n_temp[i]+self.args.N[self.key[i]])
        
        if self.flip_blue:
            obs_blue = self.flip_blue_obs(obs_n_array[self.key[1]], state, position[self.key[i]])
            obs_n_array[self.key[1]] = obs_blue
        
        last_onehot_a_n = {self.key[i]:np.zeros((self.args.N[self.key[i]], self.args.action_dim)) for i in range(self.N_group)}  # Last actions of N agents(one-hot)
        a_n={}
        role_lables_n = {self.key[i]:np.zeros((self.args.N[self.key[i]]), dtype=np.int32) for i in range(self.N_group)}
        agent_mask_n = self.env.get_agent_mask()
        agent_mask_n_dict = {self.key[i]:agent_mask_n[index_n_temp[i]:index_n_temp[i+1]] for i in range(self.N_group)}
        active_n = {self.key[i]:[1 if item else 0 for item in agent_mask_n_dict[self.key[i]]] for i in range(self.N_group)}
        cluster_num = 9 if self.args.use_adaptive_k else self.args.cluster_num
        max_cluster_num = 9 if self.args.use_adaptive_k else self.args.cluster_num*2
        former_act_prob = {self.key[i]:[np.zeros((self.args.action_dim), dtype=np.float32) for _ in range(cluster_num)] for i in range(self.N_group)}
        # if self.args.use_att:
        mean_act_list = copy.deepcopy(former_act_prob)
        center = self.center
        role_embedding_n = {}
        for i in range(self.N_group):
            if self.args.use_recl and self.args.algorithm=='RMFQ':
                obs_tensor = torch.FloatTensor(obs_n_array[self.key[i]]).to(self.device)
                obs_embedding_n, role_embedding = self.agent_n[self.key[i]].get_role_embedding(obs_tensor, torch.FloatTensor(last_onehot_a_n[self.key[i]]).to(self.device))
                if sum(agent_mask_n_dict[self.key[i]]) >= max_cluster_num:
                    role_lables = self.agent_n[self.key[i]].role_clustering(obs_embedding_n, act_embedding_n=None, last_center=center[self.key[i]])
                    role_lables = role_lables.squeeze()
                    role_lables_n.update({self.key[i]:role_lables.detach().cpu().numpy()})
                else:
                    role_lables = role_lables_n[self.key[i]]
                    role_lables_n.update({self.key[i]:role_lables})
                role_embedding_n.update({self.key[i]:role_embedding})
                
        if self.args.algorithm == 'MTMFQ':
            w = 5
            h = self.args.N
            mat = {self.key[i]:np.zeros((h[self.key[i]],w)) for i in range(self.N_group)}
            mat_next ={self.key[i]:np.zeros((h[self.key[i]],w)) for i in range(self.N_group)}
            kmeans = KMeans(
                n_clusters=self.cluster_num,
                max_iter=100,
                tolerance=1e-4,
                distance='euclidean',
                sub_sampling=None,
                max_neighbors=15,
                )
        
        for episode_step in range(self.args.episode_limit):
            epsilon = 0 if evaluate else self.epsilon
            
            if episode_step > 1:
                mean_act_list = copy.deepcopy(former_act_prob)
            else:
                mean_act_list = {self.key[i]:np.tile([former_act_prob[self.key[i]]], (len(obs_n_array[self.key[i]]), 1,1)) for i in range(self.N_group)}

            for i in range(self.N_group):
                # avail_a_list.update({self.key[i]:[avail_a_n[key] for key in avail_a_n.keys() if self.key[i] in key]})
                obs_tensor = torch.FloatTensor(obs_n_array[self.key[i]]).to(self.device)
                obs_embedding_eval = self.agent_n[self.key[i]].get_agent_embedding(obs_tensor)
                epsilon1 = epsilon
                if self.args.use_recl and self.args.algorithm=='RMFQ':
                    a = self.agent_n[self.key[i]].act(obs_embedding_eval, role_embedding_n[self.key[i]], mean_act_list[self.key[i]], eps=epsilon1)
                    
                elif self.args.algorithm in ['MFQ', 'MTMFQ']:
                    a = self.agent_n[self.key[i]].act(obs_embedding_eval, mean_action=mean_act_list[self.key[i]], eps=epsilon1)
                    
                a_n.update({self.key[i]:a.cpu().numpy()})
            
            if episode_step == 0 and self.total_episodes == 0 and evaluate==False:
                self.center = center

            action = np.concatenate([a_n[self.key[i]] for i in range(self.N_group)], axis=0)
            if episode_step == 0:
                action = np.ones_like(action) * 6
            all_action = {self.agents[i]: action[i] for i in range(len(self.agents))}
            obs_n_next, r, termination, truncations, info, _ = self.env.step(all_action)  # Take a step   

            agent_mask_n_next = self.env.get_agent_mask()  # 活着的智能体值为True
            agent_mask_n_dict_next = {self.key[i]:agent_mask_n_next[index_n_temp[i]:index_n_temp[i+1]] for i in range(self.N_group)}
            active_n_next = {self.key[i]:[1 if item else 0 for item in agent_mask_n_dict_next[self.key[i]]] for i in range(self.N_group)}
            
            obs_n_next_padding = {}
            r_padding = {}
            # 对mask为false的智能体obs和r填0
            for j in range(len(agent_mask_n_next)):
                if not agent_mask_n_next[j]:
                    obs_n_next_padding[self.agents[j]]=np.zeros([self.args.obs_dim[1], self.args.obs_dim[2], 9], dtype=np.float32)
                    r_padding[self.agents[j]] = 0
                else:
                    obs_n_next_padding[self.agents[j]] = obs_n_next[self.agents[j]]
                    r_padding[self.agents[j]] = r[self.agents[j]]          
            
            obs_n_dict_next = {self.key[i]:{key:value for key, value in obs_n_next_padding.items() if self.key[i] in key} for i in range(self.N_group)}
            obs_n_array_next = {self.key[i]:np.array(list(obs_n_dict_next[self.key[i]].values())).transpose(0, 3, 1, 2) for i in range(self.N_group)}
            obs_n_array_next = {self.key[i]:obs_n_array_next[self.key[i]][:,0:7] for i in range(self.N_group)}
            r_n = {self.key[i]:{key:value for key, value in r_padding.items() if self.key[i] in key} for i in range(self.N_group)}
            r_n_array = {self.key[i]:np.array(list(r_n[self.key[i]].values())) for i in range(self.N_group)}
            r_total = {self.key[i]:sum(r_n[self.key[i]].values()) for i in range(self.N_group)}

            next_position = self.get_agent_position(obs_n_next_padding)
            
            alive, win_tag = self.get_alive_wintag(agent_mask_n_dict_next)
            done = any(win_tag.values())

            episode_reward = {self.key[i]:episode_reward[self.key[i]]+r_total[self.key[i]] for i in range(self.N_group)}
            
            if not evaluate:

                if done or (episode_step + 1) == self.args.episode_limit:
                    dw = True
                else:
                    dw = False

                # Store the transition
                if episode_step:
                    for i in range(self.N_group):
                        self.replay_buffer[self.key[i]].store_transition(episode_step-1, obs_n_array[self.key[i]], last_onehot_a_n[self.key[i]], a_n[self.key[i]], r_n_array[self.key[i]], dw, active_n[self.key[i]], mean_act_list[self.key[i]], role_lables_n[self.key[i]], state, position[self.key[i]])
                last_onehot_a_n = {self.key[i]:np.eye(self.args.action_dim)[a_n[self.key[i]]] for i in range(self.N_group)}  # Convert actions to one-hot vectors
                # obs_a_n_buffer[episode_step] = obs_n
                # Decay the epsilon
                self.epsilon = self.epsilon - self.args.epsilon_decay if self.epsilon - self.args.epsilon_decay > self.args.epsilon_min else self.args.epsilon_min
            
            
            obs_n = obs_n_next
            obs_n_array = obs_n_array_next
            agent_mask_n = agent_mask_n_next
            agent_mask_n_dict = agent_mask_n_dict_next
            active_n = active_n_next
            
            
            if not done:
                next_state = self.env.state()
                state = next_state
            position = next_position
            
            
            for i in range(self.N_group):
                if self.args.use_recl and self.args.algorithm=='RMFQ':
                    obs_tensor = torch.FloatTensor(obs_n_array[self.key[i]]).to(self.device)
                    obs_embedding_n, role_embedding = self.agent_n[self.key[i]].get_role_embedding(obs_tensor, torch.FloatTensor(last_onehot_a_n[self.key[i]]).to(self.device))
                    if sum(agent_mask_n_dict[self.key[i]]) >= max_cluster_num and episode_step % 5 == 0:
                        role_lables = self.agent_n[self.key[i]].role_clustering(obs_embedding_n, act_embedding_n=None, last_center=center[self.key[i]])
                        role_lables = role_lables.squeeze()
                        role_lables_n.update({self.key[i]:role_lables.detach().cpu().numpy()})
                    else:
                        role_lables = role_lables_n[self.key[i]]
                        role_lables_n.update({self.key[i]:role_lables})
                    role_embedding_n.update({self.key[i]:role_embedding})
                if self.args.algorithm == 'MTMFQ':
                    mat_next[self.key[i]][:,0:-1] = mat[self.key[i]][:,1:] 
                    mat_next[self.key[i]][:,-1] = a_n[self.key[i]]
                    mat2 = mat_next[self.key[i]].copy()
                    mat2[~agent_mask_n_dict[self.key[i]]] = -1
                    _, lables = kmeans(torch.FloatTensor(mat2).unsqueeze(0).to(self.device))
                    role_lables_n.update({self.key[i]:lables.squeeze().detach().cpu().numpy()})
                    mat[self.key[i]] = mat_next[self.key[i]]
            # 更新各类的平均动作
            if episode_step > 0:
                former_act_prob = self.get_mean_act(former_act_prob, obs_n_array, role_lables_n, a_n, agent_mask_n_dict, state, position)
            
            
            if done:
                break
        self.writer.add_scalar('episode_step', episode_step+1, self.total_episodes)
        self.writer.add_scalar('episode_alives_'+self.key[0], alive[self.key[0]], self.total_episodes)
        self.writer.add_scalar('episode_alives_'+self.key[1], alive[self.key[1]], self.total_episodes)

        if not evaluate:
            mean_act_list = copy.deepcopy(former_act_prob)
            for i in range(self.N_group):
                self.replay_buffer[self.key[i]].store_last_step(episode_step, obs_n_array[self.key[i]], last_onehot_a_n[self.key[i]], mean_act_list[self.key[i]], active_n[self.key[i]], state, position[self.key[i]], role_lables_n[self.key[i]])
        
        
        return win_tag, episode_reward, episode_step+1
    
    def get_alive_wintag(self, agent_mask):
        alives = {self.key[i]:0 for i in range(self.N_group)}
        wintag = {self.key[i]:False for i in range(self.N_group)}
        for i in range(self.N_group):
            alives[self.key[i]] = sum([1 for item in agent_mask[self.key[i]] if item])
        if alives[self.key[0]] == 0:
            wintag[self.key[1]] = True
        elif alives[self.key[1]] == 0:
            wintag[self.key[0]] = True
        return alives, wintag

    def get_mean_act(self, former_act_prob, obs, role_global, act_global, mask, state, position):
        for i in range(self.N_group):

            obs_ = obs[self.key[i]]
            obs_tensor = torch.FloatTensor(obs_).unsqueeze(0).to(self.device)
            role_tensor = torch.FloatTensor(role_global[self.key[i]]).unsqueeze(0).to(self.device)
            act_tensor = torch.LongTensor(act_global[self.key[i]]).unsqueeze(0).to(self.device)
            if self.flip_blue and i ==1:
                act_tensor = self.flip_blue_act(act_tensor)
            mask_tensor = torch.FloatTensor(mask[self.key[i]]).unsqueeze(0).to(self.device)
            weights, mean_act = self.agent_n[self.key[i]].cal_att_weights_mean_acts(obs_tensor, role_tensor, act_tensor, mask_tensor, key = 'eval') # N*N权重矩阵
            mean_act_array = mean_act.detach().cpu().numpy()
            former_act_prob[self.key[i]] = mean_act_array
            
        return former_act_prob
    
    def get_agent_position(self, obs_n):
        position = {self.agents[i]:[0,0] for i in range(len(self.agents))}
        for agent in self.agents:
            position[agent] = [round(obs_n[agent][0,0,-1]*self.args.map_size), round(obs_n[agent][0,0,-2]*self.args.map_size)]
        position = {self.key[i]:np.array([position[self.agents[j]] for j in range(len(self.agents)) if self.key[i] in self.agents[j]]) for i in range(self.N_group)}
        return position
    

    