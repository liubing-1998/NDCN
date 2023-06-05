# coding:utf-8

import os
import argparse
import time
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import networkx as nx
import datetime
from utils_in_learn_dynamics import *
from neural_dynamics import *
import torchdiffeq as ode
import sys
import functools
print = functools.partial(print, flush=True)

parser = argparse.ArgumentParser('Biochemical Dynamics Case')
parser.add_argument('--method', type=str,
                    choices=['dopri5', 'adams', 'explicit_adams', 'fixed_adams','tsit5', 'euler', 'midpoint', 'rk4'],
                    default='euler')  # dopri5
parser.add_argument('--rtol', type=float, default=0.01,
                    help='optional float64 Tensor specifying an upper bound on relative error, per element of y')
parser.add_argument('--atol', type=float, default=0.001,
                    help='optional float64 Tensor specifying an upper bound on absolute error, per element of y')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-3,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--hidden', type=int, default=20,
                    help='Number of hidden units.')
parser.add_argument('--time_tick', type=int, default=100) # default=10)
parser.add_argument('--sampled_time', type=str,
                    choices=['irregular', 'equal'], default='irregular')

parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')

parser.add_argument('--n', type=int, default=400, help='Number of nodes')
parser.add_argument('--sparse', action='store_true')
parser.add_argument('--network', type=str,
                    choices=['grid', 'random', 'power_law', 'small_world', 'community'], default='grid')
parser.add_argument('--layout', type=str, choices=['community', 'degree'], default='community')
parser.add_argument('--seed', type=int, default=0, help='Random Seed')
parser.add_argument('--T', type=float, default=5., help='Terminal Time')
parser.add_argument('--T2', type=float, default=6., help=('Extrapolation terminal time'))
parser.add_argument('--T3', type=float, default=40., help=('long-time begin time'))
parser.add_argument('--T4', type=float, default=50., help=('long-time terminal time'))
parser.add_argument('--operator', type=str,
                    choices=['lap', 'norm_lap', 'kipf', 'norm_adj' ], default='norm_lap')

parser.add_argument('--baseline', type=str,
                    choices=['ndcn', 'no_embed', 'no_control', 'no_graph',
                             'lstm_gnn', 'rnn_gnn', 'gru_gnn'],
                    default='ndcn')
parser.add_argument('--dump', action='store_true', help='Save Results')

parser.add_argument('--count', type=str, default='0')  # 日志计数用的
# 只有BiochemicalDynamics上随机初始化
parser.add_argument('--init_random', action='store_true', help='if True then random init x value of t0')
# parser.add_argument('--power_law_param', type=int, default=5, help='power_law parameter default=5, in BirthDeath is 4')
# parser.add_argument('--dump_appendix', type=str, default='',
#                    help='dump_appendix to distinguish results file, e.g. same as baseline name')
# use args.baseline instead

args = parser.parse_args()
if args.gpu >= 0:
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')

# args.seed<=0,随机出一个种子，否则使用其值作为种子
if args.seed <= 0:
    seed = random.randint(0, 2022)
else:
    seed = args.seed

# 设置随机数种子
random.seed(seed)
np.random.seed(seed)
# 为CPU设置种子用于生成随机数，以使结果是确定的
torch.manual_seed(seed)
# 为当前GPU设置种子用于生成随机数，以使结果是确定的
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

if args.viz:
    # dirname = r'figure/biochemical/' + args.network
    dirname = r'figure/biochemical/' + args.network + '/' + args.count
    makedirs(dirname)
    fig_title = r'biochemical Dynamics'

if args.dump:
    # results_dir = r'results/biochemical/' + args.network
    results_dir = r'results/biochemical/' + args.network
    makedirs(results_dir)

# Build network # A: Adjacency matrix, L: Laplacian Matrix,  OM: Base Operator
n = args.n  # e.g nodes number 400
N = int(np.ceil(np.sqrt(n)))  # grid-layout pixels :20
# seed = args.seed
if args.network == 'grid':
    print("Choose graph: " + args.network)
    A = grid_8_neighbor_graph(N)
    G = nx.from_numpy_array(A.numpy())
elif args.network == 'random':
    print("Choose graph: " + args.network)
    G = nx.erdos_renyi_graph(n, 0.02, seed=seed)  # 0.1
    G = networkx_reorder_nodes(G, args.layout)
    A = torch.FloatTensor(nx.to_numpy_array(G))
elif args.network == 'power_law':
    print("Choose graph: " + args.network)
    G = nx.barabasi_albert_graph(n, 5, seed=seed)
    G = networkx_reorder_nodes(G,  args.layout)
    A = torch.FloatTensor(nx.to_numpy_array(G))
elif args.network == 'small_world':
    print("Choose graph: " + args.network)
    G = nx.newman_watts_strogatz_graph(n, 5, 0.5, seed=seed)
    G = networkx_reorder_nodes(G, args.layout)
    A = torch.FloatTensor(nx.to_numpy_array(G))
elif args.network == 'community':
    print("Choose graph: " + args.network)
    n1 = int(n/3)
    n2 = int(n/3)
    n3 = int(n/4)
    n4 = n - n1 - n2 -n3
    G = nx.random_partition_graph([n1, n2, n3, n4], .04, .005, seed=seed)  # .25, .01
    G = networkx_reorder_nodes(G, args.layout)
    A = torch.FloatTensor(nx.to_numpy_array(G))

if args.viz:
    # makedirs(r'figure/network/')
    path = r'figure/biochemical' + '/network' + args.count + '/'
    makedirs(path)
    visualize_graph_matrix(G, args.network, path)

D = torch.diag(A.sum(1))
L = (D - A)

# equally-sampled time
# sampled_time = 'irregular'
if args.sampled_time == 'equal':
    print('Build Equally-sampled -time dynamics')
    t = torch.linspace(0., args.T, args.time_tick)  # args.time_tick) # 100 vector
    # train_deli = 80
    id_train = list(range(int(args.time_tick * 0.8))) # first 80 % for train
    id_test = list(range(int(args.time_tick * 0.8), args.time_tick)) # last 20 % for test (extrapolation)
    t_train = t[id_train]
    t_test = t[id_test]
elif args.sampled_time == 'irregular':
    print('Build irregularly-sampled -time dynamics')
    t_train_inter = np.sort(np.random.uniform(0, args.T, 100))
    t_extra = np.sort(np.random.uniform(args.T, args.T2, 20))
    t_long_time = np.sort(np.random.uniform(args.T3, args.T4, 20))
    t = np.concatenate((t_train_inter, t_extra, t_long_time), axis=0)
    t[0] = 0
    t = torch.tensor(t)
    id_test = list(range(100, int(100 * 1.2)))  # [100-120) for test (extrapolation)
    id_test2 = np.random.permutation(range(1, 100))[
               :int(100 * 0.2)].tolist()  # 20 random in (0, 100) for test (interpolation)
    id_test2.sort()
    id_train = list(set(range(100)) - set(id_test2))  # other 80 in (0, 100) for train
    id_train.sort()
    id_test3 = list(range(120, 140))

    t_train = t[id_train]
    t_test = t[id_test]
    t_test2 = t[id_test2]
    t_test3 = t[id_test3]

if args.operator == 'lap':
    print('Graph Operator: Laplacian')
    OM = L
elif args.operator == 'kipf':
    print('Graph Operator: Kipf')
    OM = torch.FloatTensor(zipf_smoothing(A.numpy()))
elif args.operator == 'norm_adj':
    print('Graph Operator: Normalized Adjacency')
    OM = torch.FloatTensor(normalized_adj(A.numpy()))
else:
    print('Graph Operator[Default]: Normalized Laplacian')
    OM = torch.FloatTensor(normalized_laplacian(A.numpy()))  # L # normalized_adj


if args.baseline in ['lstm_gnn', 'rnn_gnn', 'gru_gnn']:
    print('For temporal-gnn model lstm_gnn, rnn_gnn, and gru_gnn'
          'Graph Operator Choose: Kipf in GCN')
    OM = torch.FloatTensor(zipf_smoothing(A.numpy()))


if args.sparse:
    # For small network, dense matrix is faster
    # For large network, sparse matrix cause less memory
    L = torch_sensor_to_torch_sparse_tensor(L)
    A = torch_sensor_to_torch_sparse_tensor(A)
    OM = torch_sensor_to_torch_sparse_tensor(OM)

# Initial Value
x0 = torch.zeros(N, N)
# 随机生成数据
if args.init_random:
    print("x0随机初始化（0-25均匀分布初始化）")
    x0 = 25*torch.rand(N, N)
else:
    print("x0固定区域指定值初始化")
    x0[int(0.05 * N):int(0.25 * N), int(0.05 * N):int(0.25 * N)] = 25  # x0[1:5, 1:5] = 25  for N = 20 or n= 400 case
    x0[int(0.45 * N):int(0.75 * N), int(0.45 * N):int(0.75 * N)] = 20  # x0[9:15, 9:15] = 20 for N = 20 or n= 400 case
    x0[int(0.05 * N):int(0.25 * N), int(0.35 * N):int(0.65 * N)] = 17  # x0[1:5, 7:13] = 17 for N = 20 or n= 400 case
x0 = x0.view(-1, 1).float()
energy = x0.sum()

class BiochemicalDiffusion(nn.Module):
    def __init__(self):
        super(BiochemicalDiffusion, self).__init__()
        self.A = A
        self.f = 1
        self.b = 0.1
        self.r = 0.01  # 暂时没有用上，没有乘
    def forward(self, t, x):
        """
        :param t:  time tick
        :param x:  initial value:  is 2d row vector feature, n * dim
        :return: dxi(t)/dt = f-b*xi - \sum_{j=1}^{N}Aij r*xi *xj
        If t is not used, then it is autonomous system, only the time difference matters in numerical computing
        """
        f = self.f - self.b * x
        outer = torch.mm(self.A, self.r * torch.mm(x, x.t()))
        outer = torch.diag(outer).view(-1, 1)
        f -= outer
        return f

with torch.no_grad():
    solution_numerical = ode.odeint(BiochemicalDiffusion(), x0, t, method='dopri5')  # shape: 1000 * 1 * 2
    # print(solution_numerical.shape)


now = datetime.datetime.now()
appendix = now.strftime("%m%d-%H%M%S")
zmin = solution_numerical.min()
zmax = solution_numerical.max()
for ii, xt in enumerate(solution_numerical, start=1):
    if args.viz and (ii % 10 == 1):
        # print(xt.shape)
        # visualize(N, x0, xt, '{:03d}-tru'.format(ii)+appendix, fig_title, dirname, zmin, zmax)
        visualize(N, x0, xt, appendix + '-true-{:03d}'.format(ii), fig_title, dirname, zmin, zmax)

true_y = solution_numerical.squeeze().t().to(device)  # 120 * 1 * 400  --squeeze--> 120 * 400 -t-> 400 * 120
true_y0 = x0.to(device)  # 400 * 1
true_y_train = true_y[:, id_train].to(device)  # 400*80  for train
true_y_test = true_y[:, id_test].to(device)  # 400*20  for extrapolation prediction
if args.sampled_time == 'irregular':
    true_y_test2 = true_y[:, id_test2].to(device)  # 400*20  for interpolation prediction
    true_y_test3 = true_y[:, id_test3].to(device)  # long-time
L = L.to(device)  # 400 * 400
OM = OM.to(device)  # 400 * 400
A = A.to(device)

# Build model
input_size = true_y0.shape[1]   # y0: 400*1 ,  input_size:1
hidden_size = args.hidden  # args.hidden  # 20 default  # [400 * 1 ] * [1 * 20] = 400 * 20
dropout = args.dropout  # 0 default, not stochastic ODE
num_classes = 1  # 1 for regression
# Params for discrete models
input_n_graph= true_y0.shape[0]
hidden_size_gnn = 5
hidden_size_rnn = 10


flag_model_type = ""  # "continuous" "discrete"  input, model, output format are little different
# Continuous time network dynamic models
if args.baseline == 'ndcn':
    print('Choose model:' + args.baseline)
    flag_model_type = "continuous"
    model = NDCN(input_size=input_size, hidden_size=hidden_size, A=OM, num_classes=num_classes,
                 dropout=dropout, no_embed=False, no_graph=False, no_control=False,
                 rtol=args.rtol, atol=args.atol, method=args.method)
elif args.baseline == 'no_embed':
    print('Choose model:' + args.baseline)
    flag_model_type = "continuous"
    model = NDCN(input_size=input_size, hidden_size=input_size, A=OM, num_classes=num_classes,
                 dropout=dropout, no_embed=True, no_graph=False, no_control=False,
                 rtol=args.rtol, atol=args.atol, method=args.method)
elif args.baseline == 'no_control':
    print('Choose model:' + args.baseline)
    flag_model_type = "continuous"
    model = NDCN(input_size=input_size, hidden_size=hidden_size, A=OM, num_classes=num_classes,
                 dropout=dropout, no_embed=False, no_graph=False, no_control=True,
                 rtol=args.rtol, atol=args.atol, method=args.method)
elif args.baseline == 'no_graph':
    print('Choose model:' + args.baseline)
    flag_model_type = "continuous"
    model = NDCN(input_size=input_size, hidden_size=hidden_size, A=OM, num_classes=num_classes,
                 dropout=dropout, no_embed=False, no_graph=True, no_control=False,
                 rtol=args.rtol, atol=args.atol, method=args.method)
# Discrete time or Sequential network dynamic models
elif args.baseline == 'lstm_gnn':
    print('Choose model:' + args.baseline)
    flag_model_type = "discrete"
    # print('Graph Operator: Kipf') # Using GCN as graph embedding layer
    # OM = torch.FloatTensor(zipf_smoothing(A.numpy()))
    # OM = OM.to(device)
    model = TemporalGCN(input_size, hidden_size_gnn, input_n_graph, hidden_size_rnn, OM, dropout=dropout, rnn_type='lstm')
elif args.baseline == 'gru_gnn':
    print('Choose model:' + args.baseline)
    flag_model_type = "discrete"
    model = TemporalGCN(input_size, hidden_size_gnn, input_n_graph, hidden_size_rnn, OM, dropout=dropout, rnn_type='gru')
elif args.baseline == 'rnn_gnn':
    print('Choose model:' + args.baseline)
    flag_model_type = "discrete"
    model = TemporalGCN(input_size, hidden_size_gnn, input_n_graph, hidden_size_rnn, OM, dropout=dropout, rnn_type='rnn')


model = model.to(device)
# model = nn.Sequential(*embedding_layer, *neural_dynamic_layer, *semantic_layer).to(device)

num_paras = get_parameter_number(model)

if __name__ == '__main__':
    # 保存日志信息
    makedirs(r'log/')
    log_filename = r'log/biochemical' + "_" + args.network + "_NDCN_" + args.count + ".txt"
    logging.basicConfig(filename=log_filename, level=logging.DEBUG, format='%(asctime)s %(message)s',
                        datefmt='%Y/%m/%d %I:%M:%S %p')
    # logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.info("*********************************" + args.count + '_' + appendix + "*********************************")
    # 记录下参数和seed值
    logging.info(args)
    logging.info("seed: " + str(seed))

    logging.info('id_train' + str(id_train))
    logging.info('id_test' + str(id_test))
    logging.info('id_test2' + str(id_test2))
    logging.info('id_test3' + str(id_test3))

    # 保存邻接矩阵相关信息
    print(A.sum())
    logging.info(A.sum())
    print(A.sum(dim=-1))
    logging.info(A.sum(dim=-1))

    t_start = time.time()
    params = model.parameters()
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = F.l1_loss  # F.mse_loss(pred_y, true_y)
    # criterion = F.mse_loss  # F.mse_loss(pred_y, true_y)
    # time_meter = RunningAverageMeter(0.97)
    # loss_meter = RunningAverageMeter(0.97)
    if args.dump:
        results_dict = {'args': args.__dict__, 'v_iter': [], 'abs_error': [], 'rel_error': [],
                        'true_y': [solution_numerical.squeeze().t()], 'predict_y': [], 'abs_error2': [],
                        'rel_error2': [], 'predict_y2': [], 'abs_error3': [], 'rel_error3': [], 'predict_y3': [],
                        'model_state_dict': [], 'total_time': [], 'A': [], 'id_train': [], 'id_test': [],
                        'id_test2': [], 'id_test3': [], 'seed': [], 'num_paras': []}
        results_dict['A'].append(A)
        results_dict['id_train'].append(id_train)
        results_dict['id_test'].append(id_test)
        results_dict['id_test2'].append(id_test2)
        results_dict['id_test3'].append(id_test3)
        results_dict['seed'].append(seed)
        results_dict['num_paras'].append(num_paras)

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()

        if flag_model_type == "continuous":
            pred_y = model(t_train, true_y0)  # 80 * 400 * 1 should be 400 * 80
            pred_y = pred_y.squeeze().t()
            loss_train = criterion(pred_y, true_y_train) # true_y)  # 400 * 20 (time_tick)
            # torch.mean(torch.abs(pred_y - batch_y))
            relative_loss_train = criterion(pred_y, true_y_train) / true_y_train.mean()
        elif flag_model_type == "discrete":
            # true_y_train = true_y[:, id_train]  # 400*80  for train
            pred_y = model(true_y_train[:, :-1])  # true_y_train 400*80 true_y_train[:, :-1] 400*79
            # pred_y = pred_y.squeeze().t()
            loss_train = criterion(pred_y, true_y_train[:, 1:])  # true_y)  # 400 * 20 (time_tick)
            # torch.mean(torch.abs(pred_y - batch_y))
            relative_loss_train = criterion(pred_y, true_y_train[:, 1:]) / true_y_train[:, 1:].mean()
        else:
            print("flag_model_type NOT DEFINED!")
            exit(-1)

        loss_train.backward()
        optimizer.step()

        # time_meter.update(time.time() - t_start)
        # loss_meter.update(loss.item())
        if itr % args.test_freq == 0:
            with torch.no_grad():
                if flag_model_type == "continuous":
                    # pred_y = model(true_y0).squeeze().t() # odeint(model, true_y0, t)
                    # loss = criterion(pred_y, true_y)
                    # relative_loss = criterion(pred_y, true_y) / true_y.mean()
                    pred_y = model(t, true_y0).squeeze().t()  # odeint(model, true_y0, t)
                    loss = criterion(pred_y[:, id_test], true_y_test)
                    relative_loss = criterion(pred_y[:, id_test], true_y_test) / true_y_test.mean()
                    if args.sampled_time == 'irregular': # for interpolation results
                        loss2 = criterion(pred_y[:, id_test2], true_y_test2)
                        relative_loss2 = criterion(pred_y[:, id_test2], true_y_test2) / true_y_test2.mean()
                        loss3 = criterion(pred_y[:, id_test3], true_y_test3)
                        relative_loss3 = criterion(pred_y[:, id_test3], true_y_test3) / true_y_test3.mean()
                elif flag_model_type == "discrete":
                    pred_y = model(true_y_train, future=len(id_test)) #400*100
                    # pred_y = pred_y.squeeze().t()
                    loss = criterion(pred_y[:, id_test], true_y_test) #pred_y[:, id_test] 400*20
                    # torch.mean(torch.abs(pred_y - batch_y))
                    relative_loss = criterion(pred_y[:, id_test], true_y_test) / true_y_test.mean()

                if args.dump:
                    # Info to dump
                    results_dict['v_iter'].append(itr)
                    results_dict['abs_error'].append(loss.item())    # {'abs_error': [], 'rel_error': [], 'X_t': []}
                    results_dict['rel_error'].append(relative_loss.item())
                    results_dict['predict_y'].append(pred_y[:, id_test])
                    results_dict['model_state_dict'].append(model.state_dict())
                    if args.sampled_time == 'irregular':  # for interpolation results
                        results_dict['abs_error2'].append(loss2.item())  # {'abs_error': [], 'rel_error': [], 'X_t': []}
                        results_dict['rel_error2'].append(relative_loss2.item())
                        results_dict['predict_y2'].append(pred_y[:, id_test2])
                        results_dict['abs_error3'].append(loss3.item())  # {'abs_error': [], 'rel_error': [], 'X_t': []}
                        results_dict['rel_error3'].append(relative_loss3.item())
                        results_dict['predict_y3'].append(pred_y[:, id_test3])
                    # now = datetime.datetime.now()
                    # appendix = now.strftime("%m%d-%H%M%S")
                    # results_dict_path = results_dir + r'/result_' + appendix + '.' + args.dump_appendix
                    # torch.save(results_dict, results_dict_path)
                    # print('Dump results as: ' + results_dict_path)
                if args.sampled_time == 'irregular':
                    print(
                        'Iter {:04d}| Train Loss {:.6f}({:.6f} Relative) | Test Loss {:.6f}({:.6f} Relative) | Test Loss2 {:.6f}({:.6f} Relative) | Test Loss3 {:.6f}({:.6f} Relative) | Time {:.4f}'
                            .format(itr, loss_train.item(), relative_loss_train.item(), loss.item(),
                                    relative_loss.item(),
                                    loss2.item(), relative_loss2.item(), loss3.item(), relative_loss3.item(),
                                    time.time() - t_start))
                    logging.info(
                        'Iter {:04d}| Train Loss {:.6f}({:.6f} Relative) | Test Loss {:.6f}({:.6f} Relative) | Test Loss2 {:.6f}({:.6f} Relative) | Test Loss3 {:.6f}({:.6f} Relative) | Time {:.4f}'
                            .format(itr, loss_train.item(), relative_loss_train.item(), loss.item(),
                                    relative_loss.item(),
                                    loss2.item(), relative_loss2.item(), loss3.item(), relative_loss3.item(),
                                    time.time() - t_start))
                else:
                    print('Iter {:04d}| Train Loss {:.6f}({:.6f} Relative) '
                          '| Test Loss {:.6f}({:.6f} Relative) '
                          '| Time {:.4f}'
                          .format(itr, loss_train.item(), relative_loss_train.item(),
                                  loss.item(), relative_loss.item(),
                                  time.time() - t_start))
                    logging.info('Iter {:04d}| Train Loss {:.6f}({:.6f} Relative) '
                                 '| Test Loss {:.6f}({:.6f} Relative) '
                                 '| Time {:.4f}'
                                 .format(itr, loss_train.item(), relative_loss_train.item(),
                                         loss.item(), relative_loss.item(),
                                         time.time() - t_start))

    now = datetime.datetime.now()
    appendix = now.strftime("%m%d-%H%M%S")
    logging.info(appendix)
    with torch.no_grad():
        if flag_model_type == "continuous":
            pred_y = model(t, true_y0).squeeze().t()  # odeint(model, true_y0, t)
            loss = criterion(pred_y[:, id_test], true_y_test)
            relative_loss = criterion(pred_y[:, id_test], true_y_test) / true_y_test.mean()
            if args.sampled_time == 'irregular':  # for interpolation results
                loss2 = criterion(pred_y[:, id_test2], true_y_test2)
                relative_loss2 = criterion(pred_y[:, id_test2], true_y_test2) / true_y_test2.mean()
                loss3 = criterion(pred_y[:, id_test3], true_y_test3)
                relative_loss3 = criterion(pred_y[:, id_test3], true_y_test3) / true_y_test3.mean()
        elif flag_model_type == "discrete":
            pred_y = model(true_y_train, future=len(id_test))  # 400*100
            loss = criterion(pred_y[:, id_test], true_y_test)  # pred_y[:, id_test] 400*20
            relative_loss = criterion(pred_y[:, id_test], true_y_test) / true_y_test.mean()

        if args.sampled_time == 'irregular':
            print(
                'RESULT {:04d}| Train Loss {:.6f}({:.6f} Relative) | Test Loss {:.6f}({:.6f} Relative) | Test Loss2 {:.6f}({:.6f} Relative) | Test Loss3 {:.6f}({:.6f} Relative) | Time {:.4f}'
                    .format(itr, loss_train.item(), relative_loss_train.item(), loss.item(), relative_loss.item(),
                            loss2.item(), relative_loss2.item(), loss3.item(), relative_loss3.item(),
                            time.time() - t_start))
            logging.info(
                'RESULT {:04d}| Train Loss {:.6f}({:.6f} Relative) | Test Loss {:.6f}({:.6f} Relative) | Test Loss2 {:.6f}({:.6f} Relative) | Test Loss3 {:.6f}({:.6f} Relative) | Time {:.4f}'
                    .format(itr, loss_train.item(), relative_loss_train.item(), loss.item(), relative_loss.item(),
                            loss2.item(), relative_loss2.item(), loss3.item(), relative_loss3.item(),
                            time.time() - t_start))
        else:
            print('RESULT {:04d}| Train Loss {:.6f}({:.6f} Relative) '
                  '| Test Loss {:.6f}({:.6f} Relative) '
                  '| Time {:.4f}'
                  .format(itr, loss_train.item(), relative_loss_train.item(),
                          loss.item(), relative_loss.item(),
                          time.time() - t_start))
            logging.info('RESULT {:04d}| Train Loss {:.6f}({:.6f} Relative) '
                          '| Test Loss {:.6f}({:.6f} Relative) '
                          '| Time {:.4f}'
                          .format(itr, loss_train.item(), relative_loss_train.item(),
                                  loss.item(), relative_loss.item(),
                                  time.time() - t_start))

        if args.viz:
            for ii in range(pred_y.shape[1]):
                if ii % 10 == 0:
                    xt_pred = pred_y[:, ii].cpu()
                    # print(xt_pred.shape)
                    # visualize(N, x0, xt_pred, '{:03d}-{:s}-'.format(ii+1, args.baseline)+appendix, fig_title, dirname, zmin, zmax)
                    visualize(N, x0, xt_pred, appendix + '-{:s}-{:03d}'.format(args.baseline, ii + 1), fig_title, dirname, zmin, zmax)

        t_total = time.time() - t_start
        print('Total Time {:.4f}'.format(t_total))
        logging.info('Total Time {:.4f}'.format(t_total))
        num_paras = get_parameter_number(model)
        if args.dump:
            # 把训练好的模型的最后一次输出结果和时间点t都保存下来
            results_dict['pred_result'] = pred_y
            results_dict['t'] = t

            results_dict['total_time'] = t_total
            # results_dict_path = results_dir + r'/result_' + appendix + '.' + args.baseline  #args.dump_appendix
            results_dict_path = results_dir + r'/result_' + args.baseline + '_' + args.network + '_' + appendix + '_' + args.count + '.pth'  # args.dump_appendix
            torch.save(results_dict, results_dict_path)
            print('Dump results as: ' + results_dict_path)

            # Test dumped results:
            rr = torch.load(results_dict_path)
            fig, ax = plt.subplots()
            ax.plot(rr['v_iter'], rr['abs_error'], '-', label='Absolute Error')
            ax.plot(rr['v_iter'], rr['rel_error'], '--', label='Relative Error')
            legend = ax.legend( fontsize='x-large') # loc='upper right', shadow=True,
            # legend.get_frame().set_facecolor('C0')
            fig.savefig(results_dict_path + ".png", transparent=True)
            fig.savefig(results_dict_path + ".pdf", transparent=True)
            # plt.show()
            # plt.pause(0.001)
            plt.close(fig)
    logging.info("----------------------------------------------------------------------------------------------------")
    logging.info('\n')

# --time_tick 20 --niters 2500 --network grid --dump --dump_appendix differential_gcn --baseline differential_gcn  --viz
# python heat_dynamics.py  --time_tick 20 --niters 2500 --network grid --dump --dump_appendix differential_gcn --baseline differential_gcn  --viz

# python biochemical_dynamics.py --network grid --count 0

'''
公式系数等相同设置下，在ndcn上跑五次平均结果做对比
python biochemical_dynamics.py --network grid --init_random True --viz --dump --count test20_0 --seed 1480
python biochemical_dynamics.py --network grid --init_random True --viz --dump --count test20_1 --seed 558
python biochemical_dynamics.py --network grid --init_random True --viz --dump --count test20_2 --seed 1574
python biochemical_dynamics.py --network grid --init_random True --viz --dump --count test20_3 --seed 1313
python biochemical_dynamics.py --network grid --init_random True --viz --dump --count test20_4 --seed 977

python biochemical_dynamics.py --network random --init_random True --viz --dump --count test20_0 --seed 1537
python biochemical_dynamics.py --network random --init_random True --viz --dump --count test20_1 --seed 766
python biochemical_dynamics.py --network random --init_random True --viz --dump --count test20_2 --seed 318
python biochemical_dynamics.py --network random --init_random True --viz --dump --count test20_3 --seed 734
python biochemical_dynamics.py --network random --init_random True --viz --dump --count test20_4 --seed 1057

python biochemical_dynamics.py --network power_law --init_random True --viz --dump --count test20_0 --seed 1308
python biochemical_dynamics.py --network power_law --init_random True --viz --dump --count test20_1 --seed 1359
python biochemical_dynamics.py --network power_law --init_random True --viz --dump --count test20_2 --seed 1103
python biochemical_dynamics.py --network power_law --init_random True --viz --dump --count test20_3 --seed 370
python biochemical_dynamics.py --network power_law --init_random True --viz --dump --count test20_4 --seed 1263

python biochemical_dynamics.py --network small_world --init_random True --viz --dump --count test20_0 --seed 1043
python biochemical_dynamics.py --network small_world --init_random True --viz --dump --count test20_1 --seed 968
python biochemical_dynamics.py --network small_world --init_random True --viz --dump --count test20_2 --seed 1820
python biochemical_dynamics.py --network small_world --init_random True --viz --dump --count test20_3 --seed 1386
python biochemical_dynamics.py --network small_world --init_random True --viz --dump --count test20_4 --seed 179

python biochemical_dynamics.py --network community --init_random True --viz --dump --count test20_0 --seed 1933
python biochemical_dynamics.py --network community --init_random True --viz --dump --count test20_1 --seed 1661
python biochemical_dynamics.py --network community --init_random True --viz --dump --count test20_2 --seed 1665
python biochemical_dynamics.py --network community --init_random True --viz --dump --count test20_3 --seed 714
python biochemical_dynamics.py --network community --init_random True --viz --dump --count test20_4 --seed 193
'''

