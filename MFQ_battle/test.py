import random
import numpy as np
import torch
from algorithm.q_learning import RMFQ, MFQ, get_minimap, MTMFQ
import os
import sys
import copy
from matplotlib import pyplot as plt
CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]  # 当前目录
config_path = CURRENT_DIR.rsplit('/', 1)[0]  # 上三级目录
sys.path.append(config_path)
from env.magent2.magent_env import MAgent_Env
from sklearn.cluster import KMeans as sk_KMeans
from sklearn.metrics import silhouette_score
from util.scores import compute_silhouette_score
from kmeans_gpu import KMeans


class Test_Runner:
    def __init__(self, test_args, model_args_recl, model_args_cluster_num, model_args_blue_flip, model_args_algm, model_args_use_att,model_args_use_adaptive_k):
        self.args = test_args
        self.env_name = self.args.env_name
        env_seed = random.randint(0, 100000)
        # Create env
        self.env = MAgent_Env(self.env_name, env_seed, max_cycles=test_args.max_step, map_size=test_args.map_size, minimap_mode=True)
        self.env.reset()
        self.key = self.env.side_names
        
        self.N_group = len(self.key)

        self.agents = self.env.agents
        self.args.key = self.key
        self.args.N = {self.key[i]:self.env.n_agents[i] for i in range(self.N_group)}  # The number of agents
        obs_shape = self.env.observation_space(self.env.agents[0]).shape
        self.args.obs_dim = (7, obs_shape[0], obs_shape[1])
        self.args.state_dim = self.env.state_space.shape  # The dimensions of global state space
        self.args.action_dim = self.env.action_space(self.env.agents[0]).n  # The dimensions of an agent's action space
        self.args.episode_limit = self.env.max_cycles  # Maximum number of steps per episode
        print("number of agents={}".format(self.args.N))
        print("obs_dim={}".format(self.args.obs_dim))
        print("state_dim={}".format(self.args.state_dim))
        print("action_dim={}".format(self.args.action_dim))
        print("episode_limit={}".format(self.args.episode_limit))
        self.device = torch.device(test_args.device if torch.cuda.is_available() else 'cpu')
        self.center = {self.key[i]:None for i in range(self.N_group)}
        self.test_model_path = self.args.test_model_path
        
        model_args1 = copy.deepcopy(self.args)
        model_args1.cluster_num=model_args_cluster_num['red']
        model_args1.use_recl=model_args_recl['red']
        model_args1.use_att=model_args_use_att['red']
        model_args1.target='red'
        model_args1.use_adaptive_k=model_args_use_adaptive_k['red']
        model_args1.algorithm = model_args_algm['red']
        model_args2 = copy.deepcopy(self.args)
        model_args2.cluster_num=model_args_cluster_num['blue']
        model_args2.use_recl=model_args_recl['blue']
        model_args2.use_att=model_args_use_att['blue']
        model_args2.target='blue'
        model_args2.use_adaptive_k=model_args_use_adaptive_k['blue']
        model_args2.algorithm = model_args_algm['blue']
        model_args = {'red':model_args1, 'blue':model_args2}
        self.model_args = model_args
        self.flip = model_args_blue_flip
        self.model_args_algm = model_args_algm
        # Create N agents
        self.agent_n = {}
        if model_args_algm['red'] == 'RMFQ':
            self.agent_n.update({'red':RMFQ(self.model_args['red'])})
        elif model_args_algm['red'] == 'MFQ':
            self.agent_n.update({'red':MFQ(self.model_args['red'])})
        elif model_args_algm['red'] == 'MTMFQ':
            self.agent_n.update({'red':MTMFQ(self.model_args['red'])})
        if model_args_algm['blue'] == 'RMFQ':
            self.agent_n.update({'blue':RMFQ(self.model_args['blue'])})
        elif model_args_algm['blue'] == 'MFQ':
            self.agent_n.update({'blue':MFQ(self.model_args['blue'])})
        elif model_args_algm['blue'] == 'MTMFQ':
            self.agent_n.update({'blue':MTMFQ(self.model_args['blue'])})
        
    
    def test(self, ):
        # load model
        for i in range(self.N_group):
            self.agent_n[self.key[i]].load_model(self.test_model_path[self.key[i]][0], self.test_model_path[self.key[i]][1])
        
        test_num = 50
        alive_sum = {self.key[i]:0 for i in range(self.N_group)}
        win_num = {self.key[i]:0 for i in range(self.N_group)}
        win_num2 = {self.key[i]:0 for i in range(self.N_group)}
        reward_sum = {self.key[i]:0 for i in range(self.N_group)}    
        for _ in range(test_num):
            win_tag, reward, _, alive = self.test_episode()
            alive_sum = {self.key[i]:alive_sum[self.key[i]]+alive[self.key[i]] for i in range(self.N_group)}
            print("alive:", alive)
            if alive[self.key[0]]>alive[self.key[1]]:
                win_num[self.key[0]] += 1
            elif alive[self.key[0]]<=alive[self.key[1]]:
                win_num[self.key[1]] += 1
            win_num2 = {self.key[i]:win_num2[self.key[i]]+1 if win_tag[self.key[i]] else win_num2[self.key[i]] for i in range(self.N_group)}
            reward_sum = {self.key[i]:reward_sum[self.key[i]]+reward[self.key[i]] for i in range(self.N_group)}
        print("alive_avg:", {self.key[i]:alive_sum[self.key[i]]/test_num for i in range(self.N_group)})
        print("win_num:", win_num)
        print("win_num2:", win_num2)
        print("reward_avg:", {self.key[i]:reward_sum[self.key[i]]/test_num for i in range(self.N_group)})
        
        
            
        self.env.close()
        return win_num, {self.key[i]:alive_sum[self.key[i]]/test_num for i in range(self.N_group)}, {self.key[i]:reward_sum[self.key[i]]/test_num for i in range(self.N_group)}
    
    def test_episode(self):
        win_tag = {self.key[i]:False for i in range(self.N_group)}
        episode_reward = {self.key[i]:0 for i in range(self.N_group)}
        obs_n, info = self.env.reset()
        obs_n_dict = {self.key[i]:{key:value for key, value in obs_n.items() if self.key[i] in key} for i in range(self.N_group)}
        obs_n_array = {self.key[i]:np.array(list(obs_n_dict[self.key[i]].values())).transpose(0, 3, 1, 2)[:,0:7] for i in range(self.N_group)}
        self.flip_red = False

        
        if self.flip:
            obs_blue = np.flip(obs_n_array[self.key[1]], axis=-1).copy()
            position = torch.tensor(self.get_agent_position(obs_n)[self.key[1]]).to(self.device)
            position = position.unsqueeze(0)
            state = torch.tensor(self.env.state()).to(self.device).unsqueeze(0)
            state = state.expand(1, self.args.N[self.key[0]], self.args.map_size,self.args.map_size, 5)
            # import pdb; pdb.set_trace()            
            minimap_red, minimap_blue = get_minimap(position, state)
            obs_blue[:,3] = minimap_blue.squeeze().cpu().numpy()
            obs_blue[:,6] = minimap_red.squeeze().cpu().numpy()
            obs_n_array[self.key[1]] = obs_blue
        if self.flip_red:
            obs_n_array.update({self.key[0]: np.flip(obs_n_array[self.key[0]], axis=-1).copy()})

        index_n_temp = [0]
        for i in range(self.N_group):
            index_n_temp.append(index_n_temp[i]+self.args.N[self.key[i]])
        
        a_n={}
        role_labels_n = {self.key[i]:np.zeros((self.args.N[self.key[i]]), dtype=np.int32) for i in range(self.N_group)}
        agent_mask_n = self.env.get_agent_mask()
        agent_mask_n_dict = {self.key[i]:agent_mask_n[index_n_temp[i]:index_n_temp[i+1]] for i in range(self.N_group)}
        active_n = {self.key[i]:[1 if item else 0 for item in agent_mask_n_dict[self.key[i]]] for i in range(self.N_group)}
        max_cluster_num = {self.key[i]:9 if self.model_args[self.key[i]].use_adaptive_k else self.model_args[self.key[i]].cluster_num for i in range(self.N_group)}
        former_act_prob = {self.key[i]:[np.zeros((self.args.action_dim), dtype=np.float32) for _ in range(max_cluster_num[self.key[i]])] for i in range(self.N_group)}
        mean_act_list = copy.deepcopy(former_act_prob)
        state = self.env.state()
        last_onehot_a_n = {self.key[i]:np.zeros((self.args.N[self.key[i]], self.args.action_dim)) for i in range(self.N_group)}

        role_lables_n = {self.key[i]:np.zeros((self.args.N[self.key[i]]), dtype=np.int32) for i in range(self.N_group)}       
        role_embedding_n = {}
        for i in range(self.N_group):
            if self.model_args[self.key[i]].use_recl and self.model_args[self.key[i]].algorithm=='RMFQ':
                obs_tensor = torch.FloatTensor(obs_n_array[self.key[i]]).to(self.device)
                obs_embedding_n, role_embedding = self.agent_n[self.key[i]].get_role_embedding(obs_tensor, torch.FloatTensor(last_onehot_a_n[self.key[i]]).to(self.device))
                if sum(agent_mask_n_dict[self.key[i]]) >= max_cluster_num[self.key[i]]:
                    role_lables = self.agent_n[self.key[i]].role_clustering(obs_embedding_n, act_embedding_n=None, last_center=None)
                    role_lables = role_lables.squeeze()
                    role_lables_n.update({self.key[i]:role_lables.detach().cpu().numpy()})
                else:
                    role_lables = role_lables_n[self.key[i]]
                    role_lables_n.update({self.key[i]:role_lables})
                role_embedding_n.update({self.key[i]:role_embedding})
        mat = {}
        mat_next = {}
        for i in range(self.N_group):
            if self.model_args_algm[self.key[i]] == 'MTMFQ':
                w = 5
                h = self.args.N
                mat.update({self.key[i]:np.zeros((h[self.key[i]],w))})
                mat_next.update({self.key[i]:np.zeros((h[self.key[i]],w))})
                kmeans = KMeans(
                    n_clusters=self.model_args[self.key[i]].cluster_num,
                    max_iter=100,
                    tolerance=1e-4,
                    distance='euclidean',
                    sub_sampling=None,
                    max_neighbors=15,
                    )
        
        for episode_step in range(self.args.episode_limit):
            # self.env.render()
            # import pdb; pdb.set_trace()              
            if episode_step > 1:
                mean_act_list = copy.deepcopy(former_act_prob)
            else:
                mean_act_list = {self.key[i]:np.tile([former_act_prob[self.key[i]]], (len(obs_n_array[self.key[i]]), 1,1)) for i in range(self.N_group)}
            e1 = [0,0]
            # obs_n_array['red'][:,0] = 0
            # obs_n_array['blue'][:,0] = 0
            obs_embedding_n_dict = {}
            role_embedding_n_dict = {}
            obs_embedding_eval_dict = {}
            
            for i in range(self.N_group):
                obs_tensor = torch.FloatTensor(obs_n_array[self.key[i]]).to(self.device)
                obs_embedding_eval = self.agent_n[self.key[i]].get_agent_embedding(obs_tensor)
                obs_embedding_eval_dict.update({self.key[i]:obs_embedding_eval})    
                if self.model_args[self.key[i]].use_recl and self.model_args_algm[self.key[i]] == 'RMFQ':
                    # import pdb; pdb.set_trace() 
                    a = self.agent_n[self.key[i]].act(obs_embedding_eval, role_embedding_n[self.key[i]], mean_act_list[self.key[i]], eps=0.1)

                elif self.model_args_algm[self.key[i]] in ['MFQ','MTMFQ']:
                    # import pdb; pdb.set_trace()
                    a = self.agent_n[self.key[i]].act(obs_embedding_eval, mean_action=mean_act_list[self.key[i]], eps=0.1)
                    
                
                a = a.cpu().numpy()
                if self.flip and i == 1:
                    mapping = {0: 0, 1: 3, 2: 2, 3: 1, 4: 8, 5: 7, 6: 6, 7:5, 8:4, 9: 11, 11: 9, 10: 10, 12: 12, 13: 15, 15:13, 17:16, 16: 17, 18: 20, 20:18, 14: 14, 19: 19}
                    for j in range(len(a)):
                        a[j] = mapping[a[j]]
                if self.flip_red and i == 0:
                    mapping = {0: 0, 1: 3, 2: 2, 3: 1, 4: 8, 5: 7, 6: 6, 7:5, 8:4, 9: 11, 11: 9, 10: 10, 12: 12, 13: 15, 15:13, 17:16, 16: 17, 18: 20, 20:18, 14: 14, 19: 19}
                    for j in range(len(a)):
                        a[j] = mapping[a[j]]        
                a_n.update({self.key[i]:a})
            
            action = np.concatenate([a_n[self.key[i]] for i in range(self.N_group)], axis=0)
            if episode_step == 0:
                action = np.ones_like(action)*6
            all_action = {self.agents[i]: action[i] for i in range(len(self.agents))}
            obs_n_next, r, termination, truncations, info,_ = self.env.step(all_action)  # Take a step   

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
            obs_n_array_next = {self.key[i]:np.array(list(obs_n_dict_next[self.key[i]].values())).transpose(0, 3, 1, 2)[:,0:7] for i in range(self.N_group)}
            
            
            r_n = {self.key[i]:{key:value for key, value in r.items() if self.key[i] in key} for i in range(self.N_group)}
            r_total = {self.key[i]:sum(r_n[self.key[i]].values()) for i in range(self.N_group)}

            alive, win_tag = self.get_alive_wintag(agent_mask_n_dict_next)
            done = any(win_tag.values())

            episode_reward = {self.key[i]:episode_reward[self.key[i]]+r_total[self.key[i]] for i in range(self.N_group)}

            if self.flip:
                # import pdb; pdb.set_trace()
                obs_next_blue = np.flip(obs_n_array_next[self.key[1]], axis=-1).copy()
                position = torch.tensor(self.get_agent_position(obs_n_next_padding)[self.key[1]]).to(self.device)
                position = position.unsqueeze(0)
                if not done:
                    state = self.env.state()
                    state = torch.tensor(state).to(self.device).unsqueeze(0)
                    state = state.expand(1, self.args.N[self.key[0]], self.args.map_size,self.args.map_size, 5)
                    minimap_red, minimap_blue = get_minimap(position, state)
                    obs_next_blue[:,3] = minimap_blue.squeeze().cpu().numpy()
                    obs_next_blue[:,6] = minimap_red.squeeze().cpu().numpy()
                    obs_n_array_next[self.key[1]] = obs_next_blue
            if self.flip_red:
                obs_n_array_next[self.key[0]] = np.flip(obs_n_array_next[self.key[0]], axis=-1).copy()

            
            # 更新各类的平均动作，labels每步都获取了，不一定稳定
            last_onehot_a_n = {self.key[i]:np.eye(self.args.action_dim)[a_n[self.key[i]]] for i in range(self.N_group)}  # Convert actions to one-hot vectors
            obs_n = obs_n_next
            obs_n_array = obs_n_array_next
            agent_mask_n = agent_mask_n_next
            agent_mask_n_dict = agent_mask_n_dict_next

            for i in range(self.N_group):
                if self.model_args[self.key[i]].use_recl and self.model_args[self.key[i]].algorithm=='RMFQ':   
                    obs_tensor = torch.FloatTensor(obs_n_array[self.key[i]]).to(self.device)
                    obs_embedding_n, role_embedding = self.agent_n[self.key[i]].get_role_embedding(obs_tensor, torch.FloatTensor(last_onehot_a_n[self.key[i]]).to(self.device))
                    if sum(agent_mask_n_dict[self.key[i]]) >= max_cluster_num[self.key[i]] and episode_step % 5 == 0:
                        role_lables = self.agent_n[self.key[i]].role_clustering(obs_embedding_n, act_embedding_n=None, last_center=None)
                        role_lables = role_lables.squeeze()
                        role_lables_n.update({self.key[i]:role_lables.detach().cpu().numpy()})
                    else:
                        role_lables = role_lables_n[self.key[i]]
                        role_lables_n.update({self.key[i]:role_lables})
                    role_embedding_n.update({self.key[i]:role_embedding})
                    # import pdb; pdb.set_trace()  
                if self.model_args[self.key[i]].algorithm=='MTMFQ':
                    mat_next[self.key[i]][:,0:-1] = mat[self.key[i]][:,1:] 
                    mat_next[self.key[i]][:,-1] = a_n[self.key[i]]
                    mat2 = mat_next[self.key[i]].copy()
                    mat2[~agent_mask_n_dict[self.key[i]]] = -1
                    _, lables = kmeans(torch.FloatTensor(mat2).unsqueeze(0).to(self.device))
                    role_lables_n.update({self.key[i]:lables.squeeze().detach().cpu().numpy()})
                    mat[self.key[i]] = mat_next[self.key[i]]
            # 更新各类的平均动作
            if episode_step > 0:
                former_act_prob = self.get_mean_act(former_act_prob, obs_n_array, role_lables_n, a_n, agent_mask_n_dict)
            if done:
                break      
        
        return win_tag, episode_reward, episode_step+1, alive
    
    def get_mean_act(self, former_act_prob, obs, role_global, act_global, mask):
        for i in range(self.N_group):
            if i == 1 and self.flip:
                a_blue = act_global['blue'].copy()
                mapping = {0: 0, 1: 3, 2: 2, 3: 1, 4: 8, 5: 7, 6: 6, 7:5, 8:4, 9: 11, 11: 9, 10: 10, 12: 12, 13: 15, 15:13, 17:16, 16: 17, 18: 20, 20:18, 14: 14, 19: 19}
                for j in range(len(a_blue)):
                    a_blue[j] = mapping[act_global['blue'][j]]
                act_global['blue'] = a_blue    

                
            if self.flip_red and i == 0:
                a_red = act_global['red'].copy()
                mapping = {0: 0, 1: 3, 2: 2, 3: 1, 4: 8, 5: 7, 6: 6, 7:5, 8:4, 9: 11, 11: 9, 10: 10, 12: 12, 13: 15, 15:13, 17:16, 16: 17, 18: 20, 20:18, 14: 14, 19: 19}
                for j in range(len(a_red)):
                    a_red[j] = mapping[act_global['red'][j]]
                act_global['red'] = a_red
            # import pdb; pdb.set_trace()          
            obs_tensor = torch.FloatTensor(obs[self.key[i]]).unsqueeze(0).to(self.device)
            role_tensor = torch.FloatTensor(role_global[self.key[i]]).unsqueeze(0).to(self.device)
            act_tensor = torch.LongTensor(act_global[self.key[i]]).unsqueeze(0).to(self.device)
            mask_tensor = torch.FloatTensor(mask[self.key[i]]).unsqueeze(0).to(self.device)
            weights, mean_act = self.agent_n[self.key[i]].cal_att_weights_mean_acts(obs_tensor, role_tensor, act_tensor, mask_tensor, key = 'eval') 
            mean_act_array = mean_act.detach().cpu().numpy()
            former_act_prob[self.key[i]] = mean_act_array
            
        return former_act_prob
    
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
    
    def plot_obs_embedding_n(self, obs_embedding_n_dict, label_n_dict, episode_step):
        """_summary_
        画降维后的观测空间图像
        Args:
            obs_embedding_n_dict (_type_): 字典类型{'red':obs_embedding_n, 'blue':obs_embedding_n}; obs_embedding_n: N*obs_embedding_dim
        """
        from umap import UMAP
        
        from sklearn.manifold import TSNE
        umap_model = UMAP(n_components=2, n_neighbors=15, min_dist=0.1)
        tsne_model = TSNE(n_components=2)
        for i in obs_embedding_n_dict.keys():
            obs_embedding_n = obs_embedding_n_dict[i].detach().cpu().numpy()
            label = label_n_dict[i]
            obs_e_reduced = umap_model.fit_transform(obs_embedding_n)
            # obs_e_reduced = tsne_model.fit_transform(obs_embedding_n)
            if obs_e_reduced.shape[1] == 2:
                plt.scatter(obs_embedding_n[:, 0], obs_embedding_n[:, 1], c=label, s=10)
                plt.title(i)
                plt.savefig('MFQ_magent2_attweights/fig/'+i+str(episode_step)+'obs_eval.png')
                plt.close()
            elif obs_e_reduced.shape[1] == 3:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(obs_embedding_n[:, 0], obs_embedding_n[:, 1], obs_embedding_n[:, 2], s=10)
                plt.title(i)
                plt.show()
            else:
                print("The dimension of the observation embedding is not 2 or 3")
                break

    def get_agent_position(self, obs_n):
        position = {self.agents[i]:[0,0] for i in range(len(self.agents))}
        for agent in self.agents:
            position[agent] = [round(obs_n[agent][0,0,-1]*self.args.map_size), round(obs_n[agent][0,0,-2]*self.args.map_size)]
        position = {self.key[i]:np.array([position[self.agents[j]] for j in range(len(self.agents)) if self.key[i] in self.agents[j]]) for i in range(self.N_group)}
        return position