# coding:utf-8

import os
import sys
import logging
import random
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch

from neural_dynamics import *
from utils_in_learn_dynamics import *
import torchdiffeq as ode

import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def different_initial_interpolation_extrapolation_longtime(result_filename, initial_filename, init_random=True):
    logging.info("============================================")
    pic_path = result_filename.split('\\')[-1][:-4] + "__with_initial_from__" + initial_filename.split('\\')[-1][:-4]
    logging.info(pic_path)

    # 取模型参数
    results_model = torch.load(result_filename)
    results_initial = torch.load(initial_filename)

    seed = results_initial['seed'][0]
    # args.seed<=0,随机出一个种子，否则使用其值作为种子
    if seed <= 0:
        seed = random.randint(0, 2022)
    # 设置随机数种子
    random.seed(seed)
    np.random.seed(seed)
    # 为CPU设置种子用于生成随机数，以使结果是确定的
    torch.manual_seed(seed)
    # 为当前GPU设置种子用于生成随机数，以使结果是确定的
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    logging.info("seed=" + str(seed))
    logging.info("init_random="+str(init_random))

    # 生成新的网络和初始化
    # Build network # A: Adjacency matrix, L: Laplacian Matrix,  OM: Base Operator
    n = 400  # e.g nodes number 400
    N = int(np.ceil(np.sqrt(n)))  # grid-layout pixels :20
    # Initial Value
    x0 = torch.zeros(N, N)
    # 随机生成数据
    if init_random:
        x0 = 25 * torch.rand(N, N)  # 种子固定以后，随机生成的初始化和从文件中读出来的x0值是一样的
        # x0 = results_initial['true_y'][0][:, 0].unsqueeze(dim=-1)
        logging.info("x0从对应文件中读出")
    else:
        x0[int(0.05 * N):int(0.25 * N), int(0.05 * N):int(0.25 * N)] = 25  # x0[1:5, 1:5] = 25  for N = 20 or n= 400 case
        x0[int(0.45 * N):int(0.75 * N), int(0.45 * N):int(0.75 * N)] = 20  # x0[9:15, 9:15] = 20 for N = 20 or n= 400 case
        x0[int(0.05 * N):int(0.25 * N), int(0.35 * N):int(0.65 * N)] = 17  # x0[1:5, 7:13] = 17 for N = 20 or n= 400 case
    x0 = x0.view(-1, 1).float()
    x0 = x0.to(device)  # torch.Size([400, 1])

    # print(x0.equal(results_initial['true_y'][0][:, 0].unsqueeze(dim=-1).to(device)))
    # sys.exit()

    # 模型参数初始化
    A = results_model['A'][0]
    D = torch.diag(A.sum(1))
    L = (D - A)
    L = L.to(device)
    input_size = 1
    hidden_size = results_model['args']['hidden']
    operator = results_model['args']['operator']
    num_classes = 1  # 1 for regression
    dropout = results_model['args']['dropout']
    rtol = results_model['args']['rtol']
    atol = results_model['args']['atol']
    method = results_model['args']['method']

    if operator == 'lap':
        print('Graph Operator: Laplacian')
        OM = L
    elif operator == 'kipf':
        print('Graph Operator: Kipf')
        OM = torch.FloatTensor(zipf_smoothing(A.numpy()))
    elif operator == 'norm_adj':
        print('Graph Operator: Normalized Adjacency')
        OM = torch.FloatTensor(normalized_adj(A.numpy()))
    else:
        print('Graph Operator[Default]: Normalized Laplacian')
        OM = torch.FloatTensor(normalized_laplacian(A.cpu().numpy()))  # L # normalized_adj

    OM = OM.to(device)
    model = NDCN(input_size=input_size, hidden_size=hidden_size, A=OM, num_classes=num_classes,
                 dropout=dropout, no_embed=False, no_graph=False, no_control=False,
                 rtol=rtol, atol=atol, method=method)
    model.load_state_dict(results_model['model_state_dict'][-1])
    model.to(device)

    # 时间从对应初始化的文件中读出
    t = results_initial['t']
    logging.info("t从对应文件中读出")

    class BirthDeathDynamics(nn.Module):  # PD second
        def __init__(self, A):
            super(BirthDeathDynamics, self).__init__()
            self.A = A
            self.B = 1
            self.R = 0.1
            self.b = 1
            self.a = 1

        def forward(self, t, x):
            """
            :param t:  time tick
            :param x:  initial value:  is 2d row vector feature, n * dim
            :return: dxi(t)/dt = -B*xi^b - \sum_{j=1}^{N}Aij R*xi^a
            If t is not used, then it is autonomous system, only the time difference matters in numerical computing
            """
            f = -self.B * (x ** self.b) + torch.mm(self.A, self.R * (x ** self.a))
            return f

    with torch.no_grad():
        solution_numerical = ode.odeint(BirthDeathDynamics(A), x0, t, method='dopri5') # shape: 1000 * 1 * 2
        print("choice BirthDeathDynamics")
        print(solution_numerical.shape)
        logging.info("choice BirthDeathDynamics")
    true_y = solution_numerical.squeeze().t().to(device)  # 120 * 1 * 400  --squeeze--> 120 * 400 -t-> 400 * 120
    true_y0 = x0.to(device)  # 400 * 1

    # 训练集测试集划分从对应初始化的文件中读出
    id_test = results_initial['id_test'][0]
    id_test2 = results_initial['id_test2'][0]
    id_test3 = results_initial['id_test3'][0]
    logging.info("id_test, id_test2, id_test3从对应文件中读出")

    criterion = F.l1_loss
    pred_y = model(t, true_y0).squeeze().t()  # odeint(model, true_y0, t)
    loss1 = criterion(pred_y[:, id_test], true_y[:, id_test])
    relative_loss1 = criterion(pred_y[:, id_test], true_y[:, id_test]) / true_y[:, id_test].mean()
    loss2 = criterion(pred_y[:, id_test2], true_y[:, id_test2])
    relative_loss2 = criterion(pred_y[:, id_test2], true_y[:, id_test2]) / true_y[:, id_test2].mean()
    loss3 = criterion(pred_y[:, id_test3], true_y[:, id_test3])
    relative_loss3 = criterion(pred_y[:, id_test3], true_y[:, id_test3]) / true_y[:, id_test3].mean()

    print(
        'RESULT Test Loss1 {:.6f}({:.6f} Relative) | Test Loss2 {:.6f}({:.6f} Relative) | Test Loss3 {:.6f}({:.6f} Relative))'
        .format(loss1.item(), relative_loss1.item(), loss2.item(), relative_loss2.item(), loss3.item(),
                relative_loss3.item()))
    logging.info(
        'RESULT Test Loss1 {:.6f}({:.6f} Relative) | Test Loss2 {:.6f}({:.6f} Relative) | Test Loss3 {:.6f}({:.6f} Relative))'
        .format(loss1.item(), relative_loss1.item(), loss2.item(), relative_loss2.item(), loss3.item(),
                relative_loss3.item()))


def read_file_name(file_dir):
    root_path = ""
    files = []
    for root_path, dirs, files in os.walk(file_dir):
        # print(root)  # 当前目录路径
        # print(dirs)  # 当前路径下所有子目录
        # print(files)  # 当前路径下所有非目录子文件
        pass
    return root_path, files

if __name__ == '__main__':
    if (not os.path.exists(r'.\different_initial_interpolation_extrapolation_longtime')):
        makedirs(r'.\different_initial_interpolation_extrapolation_longtime')

    log_filename = r'different_initial_interpolation_extrapolation_longtime/different_initial_birthdeath.txt'
    logging.basicConfig(filename=log_filename, level=logging.DEBUG, format='%(asctime)s %(message)s',
                        datefmt='%Y/%m/%d %I:%M:%S %p')
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    dynamic = 'heat'
    network_list = ['grid', 'random', 'power_law', 'small_world', 'community']
    for network in network_list:
        file_dir = os.path.join(r'.\results', dynamic, network)
        root, file_name_lists = read_file_name(file_dir)
        file_name_list = []
        for item in file_name_lists:
            if ".pdf" in item:
                pass
            elif ".png" in item:
                pass
            elif '_' not in item:
                pass
            else:
                file_name_list.append(os.path.join(root, item))
        for item in file_name_list[1:]:
            different_initial_interpolation_extrapolation_longtime(file_name_list[0], item, True)