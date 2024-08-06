import argparse
import torch
from run import Runner
from test import Test_Runner

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for QMIX,VDN and RECL_QMIX in SMAC environment")
    parser.add_argument("--max_train_steps", type=int, default=500000, help="Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=int, default=5000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=1, help="Evaluate times")  # 32
    # parser.add_argument("--save_freq", type=int, default=int(1e5), help="Save frequency")
    parser.add_argument("--algorithm", type=str, default="MTMFQ", help="QMIX or VDN")
    parser.add_argument("--epsilon", type=float, default=1, help="Initial epsilon")
    parser.add_argument("--epsilon_decay_steps", type=float, default=200000, help="How many steps before the epsilon decays to the minimum")
    parser.add_argument("--epsilon_min", type=float, default=0.1, help="Minimum epsilon")
    parser.add_argument("--buffer_size", type=int, default=40, help="The capacity of the replay buffer")  # 40*81*200
    parser.add_argument("--batch_size", type=int, default=83, help="Batch size (the number of episodes)")  # 83*81
    parser.add_argument("--multi_train_steps", type=int, default=32, help="Train frequency of one episode")
    
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--qmix_hidden_dim", type=int, default=32, help="The dimension of the hidden layer of the QMIX network")
    parser.add_argument("--hyper_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of the hyper-network")
    parser.add_argument("--hyper_layers_num", type=int, default=1, help="The number of layers of hyper-network")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of RNN")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of MLP")
    parser.add_argument("--add_last_action", type=bool, default=True, help="Whether to add last actions into the observation")
    parser.add_argument("--use_hard_update", type=bool, default=False, help="Whether to use hard update")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Whether to use learning rate decay")
    parser.add_argument("--lr_decay_steps", type=int, default=10000, help="every steps decay steps")  # 500
    parser.add_argument("--lr_decay_rate", type=float, default=0.98, help="learn decay rate")
    parser.add_argument("--target_update_freq", type=int, default=100, help="Update frequency of the target network")
    parser.add_argument("--tau", type=float, default=0.01, help="If use soft update")
    parser.add_argument("--seed", type=int, default=123, help="random seed")
    parser.add_argument('--device', type=str, default='cuda:6')
    parser.add_argument('--env_name', type=str, default='battle_v4')  #['3m', '8m', '2s3z']
    parser.add_argument('--train_every_steps', type=float, default=199)
    
    # RECL
    parser.add_argument("--agent_embedding_dim", type=int, default=256, help="The dimension of the agent embedding")
    parser.add_argument("--role_embedding_dim", type=int, default=64, help="The dimension of the role embedding")
    parser.add_argument("--use_ln", type=bool, default=False, help="Whether to use layer normalization")
    parser.add_argument("--cluster_num", type=int, default=int(1), help="the cluster number of knn")
    parser.add_argument("--recl_lr", type=float, default=8e-4, help="Learning rate")
    parser.add_argument("--agent_embedding_lr", type=float, default=1e-3, help="agent_embedding Learning rate")
    parser.add_argument("--train_recl_freq", type=int, default=20, help="Train frequency of the RECL network")
    parser.add_argument("--role_tau", type=float, default=0.005, help="If use soft update")
    parser.add_argument("--multi_steps", type=int, default=1, help="Train frequency of the RECL network")
    parser.add_argument("--role_mix_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of the QMIX network")
    parser.add_argument("--recl_pretrain_epochs", type=int, default=10, help="The number of pretrain epochs")
    parser.add_argument("--agent_embed_pretrain_epochs", type=int, default=10, help="The number of pretrain epochs")
    parser.add_argument("--cnn_kernel_size", type=int, default=3, help="The kernel size of the cnn")

    # attention
    parser.add_argument("--att_dim", type=int, default=128, help="The dimension of the attention net")
    parser.add_argument("--att_out_dim", type=int, default=64, help="The dimension of the attention net")
    parser.add_argument("--n_heads", type=int, default=2, help="multi-head attention")
    parser.add_argument("--soft_temperature", type=float, default=1.0, help="multi-head attention")
    parser.add_argument("--state_embed_dim", type=int, default=64, help="The dimension of the gru state net")

    # save path
    parser.add_argument('--save_path', type=str, default='MFQ_magent2_attweights/results_711')
    parser.add_argument('--model_path', type=tuple, default=None)
    parser.add_argument('--save_rate', type=int, default=100)
    
    parser.add_argument('--max_step', type=int, default=200)  # 500
    parser.add_argument('--map_size', type=int, default=45)
    
    parser.add_argument('--use_action', type=bool, default=True)  # 是否使用动作裁剪RODE部分
    parser.add_argument('--use_att', type=bool, default=False)  # 是否使用attention自适应权重
    parser.add_argument('--use_recl', type=bool, default=False)  # 是否使用对比学习，role_embedding
    parser.add_argument('--use_adaptive_k', type=bool, default=False)  # 是否使用自适应k
    
    parser.add_argument('--desc_dim', type=int, default=7)  # 使用描述统计 7
    
    parser.add_argument('--train_blue_flip', type=bool, default=False)  # 是否训练对称的blue
    
    parser.add_argument('--is_test', type=bool, default=True)  # 是否测试

    args = parser.parse_args()
    args.epsilon_decay = (args.epsilon - args.epsilon_min) / args.epsilon_decay_steps
    
    model_args_use_recl = {'blue':False, 'red':False}
    model_args_cluster_num = {'blue':1, 'red':1}
    model_args_algm = {'blue':'MFQ', 'red':'MFQ'}
    model_args_use_att = {'blue':False, 'red':True}
    model_args_use_adaptive_k = {'blue':False, 'red':False}
    # model_args_use_recl = {'blue':False, 'red':True}
    # model_args_cluster_num = {'blue':1, 'red':3}
    # model_args_algm = {'blue':'MFQ', 'red':'RMFQ'}
    # model_args_use_att = {'blue':True, 'red':True}
    # model_args_use_adaptive_k = {'blue':False, 'red':False}
    model_args_blue_flip = True
    if args.is_test:
        csv_path = 'test_battle40.csv'
        with open(csv_path, 'w') as f:
            f.write('red,blue,win,alive,rew\n')
        model_list = {
            "MFQ":{"path":('MFQ_magent2_attweights/results_711/MFQ/battle_v4/model/07-11-22-29_cn1_reFalse_attFalse_seed123adap_kFalse_L2recl/red', 293725),
                             "model_args_use_recl":False, "model_args_cluster_num":1, "model_args_algm":'MFQ', "model_args_use_att":False, "model_args_use_adaptive_k":False},
            "MTMFQ":{"path":("MFQ_magent2_attweights/results_711/MTMFQ/battle_v4/model/07-20-15-36_cn3_reFalse_attFalse_seed123adap_kFalsel2=0.0001_64MTMFQmap_size45/red",294567), "model_args_use_recl":False, "model_args_cluster_num":3, "model_args_algm":'MTMFQ', "model_args_use_att":False, "model_args_use_adaptive_k":False},          
                    "RMFQ2":{"path":("MFQ_magent2_attweights/results_711/RMFQ/battle_v4/model/07-16-22-26_cn3_reTrue_attTrue_seed123adap_kTruel2=0.0001_64/red",292046),
                        "model_args_use_recl":True, "model_args_cluster_num":3, "model_args_algm":'RMFQ', "model_args_use_att":True, "model_args_use_adaptive_k":True},

                      }

        import numpy as np
        import pandas as pd
        k = len(model_list.keys())
        win_rate = np.zeros((k,k))
        avg_alive = np.zeros((k,k))
        avg_rew = np.zeros((k,k))
        for index, key in enumerate(model_list.keys()): # 循环红方
            for index2, k2 in enumerate(model_list.keys()): # 循环蓝方
                if index == index2:
                    continue
                args.test_model_path = {'red':model_list[key]['path'], 'blue':model_list[k2]['path']}
                model_args_use_recl = {'red':model_list[key]['model_args_use_recl'], 'blue':model_list[k2]['model_args_use_recl']}
                model_args_algm = {'red':model_list[key]['model_args_algm'], 'blue':model_list[k2]['model_args_algm']}
                model_args_cluster_num = {'red':model_list[key]['model_args_cluster_num'], 'blue':model_list[k2]['model_args_cluster_num']}
                model_args_use_att = {'red':model_list[key]['model_args_use_att'], 'blue':model_list[k2]['model_args_use_att']}
                model_args_use_adaptive_k = {'red':model_list[key]['model_args_use_adaptive_k'], 'blue':model_list[k2]['model_args_use_adaptive_k']}
                runner = Test_Runner(args, model_args_use_recl, model_args_cluster_num, model_args_blue_flip, model_args_algm, model_args_use_att,model_args_use_adaptive_k)
                win, alive, rew=runner.test()
                with open(csv_path, 'a') as f:
                    f.write(key+','+k2+','+str(win)+','+str(alive)+','+str(rew)+'\n')
                win_rate[index, index2] += win['red']
                win_rate[index2, index] += win['blue']
                avg_alive[index, index2] += alive['red']
                avg_alive[index2, index] += alive['blue']
                avg_rew[index, index2] += rew['red']
                avg_rew[index2, index] += rew['blue']
        keys = list(model_list.keys())
        df_win_rate = pd.DataFrame(win_rate,index=keys, columns=keys)
        df_avg_alive = pd.DataFrame(avg_alive/2,index=keys, columns=keys)
        df_avg_rew = pd.DataFrame(avg_rew/2,index=keys, columns=keys)

        # 创建一个 ExcelWriter 对象
        with pd.ExcelWriter('battle_40.xlsx') as writer:
            # 将 DataFrame 写入 Excel 文件
            df_win_rate.to_excel(writer, sheet_name='Win Rate')
            df_avg_alive.to_excel(writer, sheet_name='Average Alive')
            df_avg_rew.to_excel(writer, sheet_name='Average Reward')
        
    else:
        runner = Runner(args)
        runner.run()
