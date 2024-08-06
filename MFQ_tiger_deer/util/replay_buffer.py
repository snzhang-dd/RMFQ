import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, args, buffer_size):
        if type(args.N) is dict:
            if args.target == args.key[0]:
                self.N = args.N[args.key[0]]
            else:
                self.N = args.N[args.key[1]]
        else:
            self.N = args.N
        self.obs_dim = args.obs_dim[args.target]
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim[args.target]
        self.episode_limit = args.episode_limit
        self.buffer_size = buffer_size
        self.episode_num = 0
        self.current_size = 0
        self.cluster_num = 9 if args.use_adaptive_k else args.cluster_num
        self.buffer = {'obs_n': np.zeros([self.buffer_size, self.episode_limit + 1, self.N, self.obs_dim[0], self.obs_dim[1], self.obs_dim[2]]),
                       's': np.zeros([self.buffer_size, self.episode_limit + 1, self.state_dim[0], self.state_dim[1], self.state_dim[2]]),
                    #    'avail_a_n': np.ones([self.buffer_size, self.episode_limit + 1, self.N, self.action_dim]),  # Note: We use 'np.ones' to initialize 'avail_a_n'
                       'last_onehot_a_n': np.zeros([self.buffer_size, self.episode_limit + 1, self.N, self.action_dim]),
                       'a_n': np.zeros([self.buffer_size, self.episode_limit, self.N]),
                       'r': np.zeros([self.buffer_size, self.episode_limit, self.N]),
                       'dw': np.ones([self.buffer_size, self.episode_limit, self.N]),  # Note: We use 'np.ones' to initialize 'dw'
                       'active': np.zeros([self.buffer_size, self.episode_limit+1, self.N]),
                       'mean_act': np.zeros([self.buffer_size, self.episode_limit+1, self.N, self.cluster_num,self.action_dim]),
                       'global_role': np.zeros([self.buffer_size, self.episode_limit+1, self.N]),
                       'position': np.zeros([self.buffer_size, self.episode_limit+1, self.N, 2]),
                       }
        self.episode_len = np.zeros(self.buffer_size)

    def store_transition(self, episode_step, obs_n, last_onehot_a_n, a_n, r, dw, active, mean_act_list, global_role, state, position):
        self.buffer['obs_n'][self.episode_num][episode_step] = obs_n
        # self.buffer['s'][self.episode_num][episode_step] = s
        # self.buffer['avail_a_n'][self.episode_num][episode_step] = avail_a_n
        self.buffer['last_onehot_a_n'][self.episode_num][episode_step + 1] = last_onehot_a_n
        self.buffer['a_n'][self.episode_num][episode_step] = a_n
        self.buffer['r'][self.episode_num][episode_step] = r
        self.buffer['dw'][self.episode_num][episode_step] = np.tile(dw, (self.N, 1)).T
        self.buffer['active'][self.episode_num][episode_step] = active
        self.buffer['mean_act'][self.episode_num][episode_step] = mean_act_list
        self.buffer['global_role'][self.episode_num][episode_step] = global_role
        self.buffer['s'][self.episode_num][episode_step] = state
        self.buffer['position'][self.episode_num][episode_step] = position

    def store_last_step(self, episode_step, obs_n, last_onehot_a_n, mean_act_list, mask, state, position, role):
        self.buffer['obs_n'][self.episode_num][episode_step] = obs_n
        # self.buffer['s'][self.episode_num][episode_step] = s
        # self.buffer['avail_a_n'][self.episode_num][episode_step] = avail_a_n
        self.buffer['last_onehot_a_n'][self.episode_num][episode_step] = last_onehot_a_n
        self.buffer['active'][self.episode_num][episode_step:] = np.array([0 for _ in range(self.N)])
        self.buffer['mean_act'][self.episode_num][episode_step] = mean_act_list
        self.buffer['active'][self.episode_num][episode_step] = mask
        self.buffer['s'][self.episode_num][episode_step] = state
        self.buffer['position'][self.episode_num][episode_step] = position
        self.buffer['global_role'][self.episode_num][episode_step] = role
        self.episode_len[self.episode_num] = episode_step  # Record the length of this episode
        self.episode_num = (self.episode_num + 1) % self.buffer_size
        self.current_size = min(self.current_size + 1, self.buffer_size)
        
    def sample(self, batch_size):
        # Randomly sampling
        index = np.random.choice(self.current_size, size=batch_size, replace=False)
        max_episode_len = int(np.max(self.episode_len[index]))
        batch = {}
        for key in self.buffer.keys():
            if key == 'obs_n' or key == 's' or key == 'avail_a_n' or key == 'last_onehot_a_n' or key == 'mean_act' or key == 'global_role':
                batch[key] = torch.FloatTensor(self.buffer[key][index, :max_episode_len + 1])
            elif key == 'a_n':
                batch[key] = torch.LongTensor(self.buffer[key][index, :max_episode_len])
            else:
                batch[key] = torch.FloatTensor(self.buffer[key][index, :max_episode_len])

        return batch, max_episode_len

    
    def sample_random(self, batch_size):
        idx_e = np.random.choice(self.current_size, size=int(batch_size), replace=True)
        idx_t = np.random.choice(self.episode_limit, size=int(batch_size), replace=True)
        idx_t = idx_t%self.episode_len[idx_e]
        idx_t = idx_t.astype(int)
        # idx_n = np.random.choice(self.N, size=int(batch_size*self.episode_limit), replace=True)
        # idx_n = np.random.choice(self.N, size=int(batch_size*self.episode_limit), replace=True)
        batch = {}
        for key in self.buffer.keys():
            if key == 'a_n':
                batch[key] = torch.LongTensor(self.buffer[key][idx_e, idx_t])
            elif key == 's':
                batch[key] = torch.FloatTensor(self.buffer[key][idx_e, idx_t]).unsqueeze(1).repeat(1, self.N, 1, 1, 1)
            else:
                batch[key] = torch.FloatTensor(self.buffer[key][idx_e, idx_t]) 
        batch['obs_next'] = torch.FloatTensor(self.buffer['obs_n'][idx_e, idx_t+1])
        batch['mean_act_next'] = torch.FloatTensor(self.buffer['mean_act'][idx_e, idx_t+1])
        batch['active_next'] = torch.FloatTensor(self.buffer['active'][idx_e, idx_t+1])
        # batch['last_obs'] = torch.FloatTensor(self.buffer['obs_n'][idx_e, np.maximum(idx_t-1,0)])
        # batch['last_role'] = torch.FloatTensor(self.buffer['global_role'][idx_e, np.maximum(idx_t-1,0)])
        batch['last_act'] = torch.LongTensor(np.where((idx_t-1>=0).reshape(-1,1), self.buffer['a_n'][idx_e, idx_t-1], np.zeros([batch_size, self.N])))
        batch['next_state'] = torch.FloatTensor(self.buffer['s'][idx_e, idx_t+1]).unsqueeze(1).repeat(1, self.N, 1, 1, 1)
        batch['next_position'] = torch.FloatTensor(self.buffer['position'][idx_e, idx_t+1])
        # batch['next_act'] = torch.LongTensor(self.buffer['a_n'][idx_e, idx_t+1])
        batch['next_role'] = torch.FloatTensor(self.buffer['global_role'][idx_e, idx_t+1])
        # batch['last_s'] = torch.FloatTensor(self.buffer['s'][idx_e, np.maximum(idx_t-1,0)]).unsqueeze(1).repeat(1, self.N, 1, 1, 1)
        # batch['last_position'] = torch.FloatTensor(self.buffer['position'][idx_e, np.maximum(idx_t-1,0)])
        return batch, self.episode_limit