import argparse
import torch
from run import Runner

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for QMIX,VDN and RECL_QMIX in SMAC environment")
    parser.add_argument("--max_train_steps", type=int, default=400000, help="Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=int, default=5000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=1, help="Evaluate times")  # 32
    # parser.add_argument("--save_freq", type=int, default=int(1e5), help="Save frequency")
    parser.add_argument("--algorithm", type=str, default="MFQ", help="QMIX or VDN")
    parser.add_argument("--epsilon", type=float, default=1, help="Initial epsilon")
    parser.add_argument("--epsilon_decay_steps", type=float, default=200000, help="How many steps before the epsilon decays to the minimum")
    parser.add_argument("--epsilon_min", type=float, default=0.1, help="Minimum epsilon")
    parser.add_argument("--buffer_size", type=int, default=20, help="The capacity of the replay buffer")  # 40*81*200
    parser.add_argument("--batch_size", type=int, default=40, help="Batch size (the number of episodes)")  # 83*81
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
    parser.add_argument('--device', type=str, default='cuda:7')
    parser.add_argument('--env_name', type=str, default='gather_v4')  #['3m', '8m', '2s3z']
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
    parser.add_argument('--save_path', type=str, default='MFQ_gather/results_711')
    # parser.add_argument('--model_path', type=tuple, default=("MFQ_magent2_attweights/results_711/RMFQ/battle_v4/model/07-14-20-51_cn3_reTrue_attTrue_seed123adap_kFalse_L2recl0.01/", 39969))
    parser.add_argument('--model_path', type=tuple, default=None)
    parser.add_argument('--save_rate', type=int, default=100)
    
    parser.add_argument('--max_step', type=int, default=500)  # 500
    parser.add_argument('--map_size', type=int, default=45)
    
    parser.add_argument('--use_action', type=bool, default=True)  # 是否使用动作裁剪RODE部分
    parser.add_argument('--use_att', type=bool, default=False)  # 是否使用attention自适应权重
    parser.add_argument('--use_recl', type=bool, default=False)  # 是否使用对比学习，role_embedding
    parser.add_argument('--use_adaptive_k', type=bool, default=False)  # 是否使用自适应k
    
    parser.add_argument('--desc_dim', type=int, default=7)  # 使用描述统计 7
    
    parser.add_argument('--train_blue_flip', type=bool, default=False)  # 是否训练对称的blue


    args = parser.parse_args()
    args.epsilon_decay = (args.epsilon - args.epsilon_min) / args.epsilon_decay_steps
    
    runner = Runner(args)
    runner.run()

