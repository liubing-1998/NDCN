# coding:utf-8

import os
import pandas as pd
import sys
import random
import logging
import argparse
from copy import deepcopy

import scipy.io
from sklearn.decomposition import PCA
import plotly.express as px
import plotly

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits import axisartist

from neural_dynamics import *
from utils_in_learn_dynamics import *
import torchdiffeq as ode

import torch.nn.functional as F

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

def create_mat_data_figure_biochemical_result(filename, init_random=True, T=60, time_tick=1200):
    logging.info("============================================")
    pic_path = ''.join(filename.split('\\')[-1]).split('.')[0] + "_mat_" + str(T) + "_" + str(time_tick) + ".mat"
    logging.info(pic_path)
    save_mat_path = os.path.join('create_mat_data/biochemical', pic_path)

    results = torch.load(filename)
    seed = results['seed'][0]

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
    logging.info("init_random=" + str(init_random))

    n = 400  # e.g nodes number 400
    N = int(np.ceil(np.sqrt(n)))  # grid-layout pixels :20
    # Initial Value
    x0 = torch.zeros(N, N)
    x0[int(0.05 * N):int(0.25 * N), int(0.05 * N):int(0.25 * N)] = 25  # x0[1:5, 1:5] = 25  for N = 20 or n= 400 case
    x0[int(0.45 * N):int(0.75 * N), int(0.45 * N):int(0.75 * N)] = 20  # x0[9:15, 9:15] = 20 for N = 20 or n= 400 case
    x0[int(0.05 * N):int(0.25 * N), int(0.35 * N):int(0.65 * N)] = 17  # x0[1:5, 7:13] = 17 for N = 20 or n= 400 case
    # 随机生成数据
    if init_random:
        x0 = 25 * torch.rand(N, N)
        print("x0随机初始化（0-25均匀分布初始化）")
    x0 = x0.view(-1, 1).float()
    x0 = x0.to(device)

    # 取参数重定义模型
    results = torch.load(filename)
    A = results['A'][0].to(device)
    '''n = 400  # e.g nodes number 400
    N = int(np.ceil(np.sqrt(n)))  # grid-layout pixels :20'''
    D = torch.diag(A.sum(1))
    L = (D - A)
    L = L.to(device)
    input_size = 1
    hidden_size = results['args']['hidden']
    operator = results['args']['operator']
    num_classes = 1  # 1 for regression
    dropout = results['args']['dropout']
    rtol = results['args']['rtol']
    atol = results['args']['atol']
    method = results['args']['method']

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
    model.load_state_dict(results['model_state_dict'][-1])
    model.to(device)

    # 生成时间刻list，从0-T时间刻中选择time_tick*1.2个时间刻
    logging.info("T=" + str(T))
    logging.info("time_tick=" + str(time_tick))
    sampled_time = 'irregular'
    if sampled_time == 'equal':
        print('Build Equally-sampled -time dynamics')
        t = torch.linspace(0., T, time_tick)
    elif sampled_time == 'irregular':
        print('Build irregularly-sampled -time dynamics')
        # irregular time sequence
        sparse_scale = 10
        t = torch.linspace(0., T, time_tick * sparse_scale)  # 100 * 10 = 1000 equally-sampled tick
        t = np.random.permutation(t)[:int(time_tick * 1.2)]
        t = torch.tensor(np.sort(t))
        t[0] = 0

    class BiochemicalDiffusion(nn.Module):
        def __init__(self, A):
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
        solution_numerical = ode.odeint(BiochemicalDiffusion(A), x0, t, method='dopri5')  # shape: 1000 * 1 * 2
        print("choice BiochemicalDynamics")
        print(solution_numerical.shape)
        logging.info("choice BiochemicalDynamics")
    true_y = solution_numerical.squeeze().t().to(device)  # 120 * 1 * 400  --squeeze--> 120 * 400 -t-> 400 * 120
    true_y0 = x0.to(device)  # 400 * 1

    # 已训练模型输出结果
    Xht_input = model.input_layer(true_y0)
    hvx = model.neural_dynamic_layer(t, Xht_input)
    # print(hvx.shape)  # torch.Size([1440, 400, 20])
    output = model.output_layer(hvx)  # torch.Size([1440, 400, 1])

    prefix_name = pic_path.split('_')[4]
    scipy.io.savemat(save_mat_path,
                     mdict={prefix_name + '_true_data': true_y.t().cpu().detach().numpy(),
                            prefix_name + '_ndcn_data': output.squeeze().cpu().detach().numpy()})

def read_file_name(file_dir):
    root = ""
    files = ""
    for root, dirs, files in os.walk(file_dir):
        # print(root)  # 当前目录路径
        # print(dirs)  # 当前路径下所有子目录
        # print(files)  # 当前路径下所有非目录子文件
        pass
    return root, files

if __name__ == '__main__':
    if (not os.path.exists(r'.\create_mat_data\biochemical')):
        makedirs(r'.\create_mat_data\biochemical')

    # 遍历所有结果
    dynamic = 'biochemical'
    network = 'grid'
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

    log_filename = r'create_mat_data/biochemical/create_mat_data_of_biochemical.txt'
    logging.basicConfig(filename=log_filename, level=logging.DEBUG, format='%(asctime)s %(message)s',
                        datefmt='%Y/%m/%d %I:%M:%S %p')
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    for filepath in file_name_list:
        # filename = filepath.split('\\')[-1]
        create_mat_data_figure_biochemical_result(filepath, init_random=True, T=60, time_tick=1200)

