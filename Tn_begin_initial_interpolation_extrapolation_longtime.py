# coding:utf-8

import os
import pandas as pd
import sys
import random
import logging
import argparse
from copy import deepcopy
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

def Tn_begin_initial_interpolation_extrapolation_longtime_heat_result(filename, seed=876, init_random=True, T=60, time_tick=1200):
    logging.info("============================================")
    pic_path = '_'.join(strtmp for strtmp in filename.split('/')[1:]).split('.')[0] + "_delta"
    logging.info(pic_path)

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

    sample = results['t'][results['id_train'][0]]  # 训练集时间点

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

    # print(x0)
    # print(results['seed'])
    # sys.exit()

    '''# 生成时间刻list，从0-T时间刻中选择time_tick*1.2个时间刻
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
        t[0] = 0'''

    class HeatDiffusion(nn.Module):
        # In this code, row vector: y'^T = y^T A^T      textbook format: column vector y' = A y
        def __init__(self, L, k=0.1):
            super(HeatDiffusion, self).__init__()
            self.L = -L  # Diffusion operator
            self.k = k  # heat capacity
            # print('k=', self.k)

        def forward(self, t, x):
            """
            :param t:  time tick
            :param x:  initial value:  is 2d row vector feature, n * dim
            :return: dX(t)/dt = -k * L *X
            If t is not used, then it is autonomous system, only the time difference matters in numerical computing
            """
            if hasattr(self.L, 'is_sparse') and self.L.is_sparse:
                f = torch.sparse.mm(self.L, x)
            else:
                f = torch.mm(self.L, x)
            return self.k * f

    t = results['t'][41:]
    results_true_y = results['true_y'][0]
    results_true_y = results_true_y[:, 41:]
    t = t - np.ones_like(t.shape) * 2
    results_true_y0 = results_true_y[:, 0].unsqueeze(dim=-1)

    with torch.no_grad():
        solution_numerical = ode.odeint(HeatDiffusion(L), results_true_y0, t, method='dopri5') # shape: 1000 * 1 * 2
        print("choice HeatDynamics")
        # print(solution_numerical.shape)
        logging.info("choice HeatDynamics")
    true_y = solution_numerical.squeeze().t().to(device)  # 120 * 1 * 400  --squeeze--> 120 * 400 -t-> 400 * 120
    # true_y0 = x0.to(device)  # 400 * 1
    true_y0 = results_true_y0.to(device)

    # 已训练模型输出结果
    Xht_input = model.input_layer(true_y0)
    integration_time_vector = t.type_as(Xht_input)
    Xht_output = ode.odeint(model.neural_dynamic_layer.odefunc, Xht_input, integration_time_vector, rtol=0.01, atol=0.001, method='euler')  # torch.Size([100, 400, 20])
    hvx = model.neural_dynamic_layer(t, Xht_input)
    # print(hvx.shape)  # torch.Size([1440, 400, 20])
    output = model.output_layer(hvx)  # torch.Size([1440, 400, 1])
    # print(output[:, 1].shape)  # torch.Size([1440, 1])
    # print(output[:, 1, :].shape)  # torch.Size([1440, 1])
    # print(output[:, 1][:, 0].shape)  # torch.Size([1440])
    Xht_output_train = Xht_output[20, 0:400, :]  # torch.Size([1440, 400, 20])

    sample = t + np.ones_like(t.shape) * 2
    # 绘制真实的状态时间图
    x_t_true_figure(t=t + np.ones_like(t.shape) * 2, true_y=true_y, sample=sample, T=T, time_tick=time_tick)

    # 绘制模型得状态时间图
    x_t_model_figure(t + np.ones_like(t.shape) * 2, output, sample, T, time_tick)

    # 绘制真实的和模型的状态差值时间图
    x_t_difference_figure(t + np.ones_like(t.shape) * 2, output, true_y, sample, T, time_tick)


def x_t_true_figure(t, true_y, sample, T, time_tick):
    print("绘制x-t真实matplotlib线图用于保存")
    fig = plt.figure()
    ax = axisartist.Subplot(fig, 111)
    fig.add_axes(ax)
    ax.axis[:].set_visible(False)
    ax.axis["x"] = ax.new_floating_axis(0, 0)
    ax.axis["y"] = ax.new_floating_axis(1, 0)
    ax.axis["x"].set_axis_direction('bottom')
    ax.axis["y"].set_axis_direction('left')
    ax.axis["x"].set_axisline_style("->", size=1.0)
    ax.axis["y"].set_axisline_style("->", size=1.0)
    # # 设置坐标轴范围
    ax.set_xlim([0, 10])
    ax.set_xticks([i for i in np.arange(0, 11, 1)])
    ax.set_ylim([0, 26])
    for i in range(0, 100, 10):
        ax.plot(t, true_y[i, :].cpu().detach().numpy(), alpha=0.7, label='$True \enspace of \enspace x_{%d}$' % i)
        ax.plot(sample, [1] * len(sample), '|', linewidth=0.001, color='k')
    plt.axvline(x=5, color="black", linestyle="dashed", )
    plt.axvspan(0, 5, color='lightskyblue', alpha=0.3, lw=0)
    ax.axis["x"].label.set_text(r"$t$")
    ax.axis["y"].label.set_text(r"$x_{i}$")
    ax.axis["x"].label.set_size(14)
    ax.axis["y"].label.set_size(14)
    # ax.legend(loc="best")
    # pic_path = r'.\\Tn_begin_initial_interpolation_extrapolation_longtime\\' + '_'.join(strtmp for strtmp in filename.split('/')[1:]).split('.')[0] \
    #            + "_extra_true_withlegend_" + str(T) + "_" + str(time_tick) + ".png"
    pic_path = r'.\\Tn_begin_initial_interpolation_extrapolation_longtime\\' + '_'.join(strtmp for strtmp in filename.split('/')[1:]).split('.')[0] \
               + "_extra_true_10_" + str(T) + "_" + str(time_tick)
    plt.savefig(pic_path + ".png", bbox_inches='tight', dpi=1000)  # dpi=1000可调分辨率参数
    plt.savefig(pic_path + ".pdf", bbox_inches='tight', dpi=1000)
    plt.close(fig)


def x_t_model_figure(t, output, sample, T, time_tick):
    print("绘制模型x-t matplotlib线图")
    fig = plt.figure()
    ax = axisartist.Subplot(fig, 111)
    fig.add_axes(ax)
    ax.axis[:].set_visible(False)
    ax.axis["x"] = ax.new_floating_axis(0, 0)
    ax.axis["y"] = ax.new_floating_axis(1, 0)
    ax.axis["x"].set_axis_direction('bottom')
    ax.axis["y"].set_axis_direction('left')
    ax.axis["x"].set_axisline_style("->", size=1.0)
    ax.axis["y"].set_axisline_style("->", size=1.0)
    # # 设置坐标轴范围
    ax.set_xlim([0, 10])
    ax.set_xticks([i for i in np.arange(0, 11, 1)])
    ax.set_ylim([0, 26])
    for i in range(0, 100, 10):
        ax.plot(t, output[:, i].squeeze().cpu().detach().numpy(), alpha=0.7, label='$NDCN \enspace of \enspace x_{%d}$' % i)
        ax.plot(sample, [1] * len(sample), '|', linewidth=0.001, color='k')
    plt.axvline(x=5, color="black", linestyle="dashed", )
    plt.axvspan(0, 5, color='lightskyblue', alpha=0.3, lw=0)
    ax.axis["x"].label.set_text(r"$t$")
    ax.axis["y"].label.set_text(r"$\hat{x_{i}}$")
    ax.axis["x"].label.set_size(14)
    ax.axis["y"].label.set_size(14)
    # ax.legend(loc="best")
    # pic_path = r'.\\Tn_begin_initial_interpolation_extrapolation_longtime\\' + '_'.join(strtmp for strtmp in filename.split('/')[1:]).split('.')[0] \
    #            + "_extra_NDCN_withlegend_" + str(T) + "_" + str(time_tick) + ".png"
    pic_path = r'.\\Tn_begin_initial_interpolation_extrapolation_longtime\\' + '_'.join(strtmp for strtmp in filename.split('/')[1:]).split('.')[0] \
               + "_extra_NDCN_10_" + str(T) + "_" + str(time_tick)
    plt.savefig(pic_path + ".png", bbox_inches='tight', dpi=1000)  # dpi=1000可调分辨率参数
    plt.savefig(pic_path + ".pdf", bbox_inches='tight', dpi=1000)
    plt.close(fig)

def x_t_difference_figure(t, output, true_y, sample, T, time_tick):
    print("绘制模型与真实值的差值x-t matplotlib线图")
    fig = plt.figure()
    ax = axisartist.Subplot(fig, 111)
    fig.add_axes(ax)
    ax.axis[:].set_visible(False)
    ax.axis["x"] = ax.new_floating_axis(0, -10)
    ax.axis["y"] = ax.new_floating_axis(1, 0)
    ax.axis["x"].set_axis_direction('bottom')
    ax.axis["y"].set_axis_direction('left')
    ax.axis["x"].set_axisline_style("->", size=1.0)
    ax.axis["y"].set_axisline_style("->", size=1.0)
    # # 设置坐标轴范围
    ax.set_xlim([0, 10])
    ax.set_xticks([i for i in np.arange(0, 11, 1)])
    ax.set_ylim([-10, 10])
    for i in range(0, 100, 10):
        ax.plot(t, output[:, i].squeeze().cpu().detach().numpy() - true_y[i, :].cpu().detach().numpy(), alpha=0.7,
                label='$difference \enspace of \enspace x_{%d}$' % i)
        ax.plot(sample, [-9] * len(sample), '|', linewidth=0.001, color='k')
    plt.axvline(x=5, color="black", linestyle="dashed", )
    plt.axvspan(0, 5, color='lightskyblue', alpha=0.3, lw=0)
    ax.axis["x"].label.set_text(r"$t$")
    ax.axis["y"].label.set_text(r"$error \enspace of \enspace x_{i}$")
    ax.axis["x"].label.set_size(14)
    ax.axis["y"].label.set_size(14)
    # ax.legend(loc="best")
    # pic_path = r'.\\Tn_begin_initial_interpolation_extrapolation_longtime\\' + '_'.join(strtmp for strtmp in filename.split('/')[1:]).split('.')[0] \
    #            + "_extra_NDCN_withlegend_" + str(T) + "_" + str(time_tick) + ".png"
    pic_path = r'.\\Tn_begin_initial_interpolation_extrapolation_longtime\\' + '_'.join(strtmp for strtmp in filename.split('/')[1:]).split('.')[0] \
               + "_difference_10_" + str(T) + "_" + str(time_tick)
    plt.savefig(pic_path + ".png", bbox_inches='tight', dpi=1000)  # dpi=1000可调分辨率参数
    plt.savefig(pic_path + ".pdf", bbox_inches='tight', dpi=1000)
    plt.close(fig)


if __name__ == '__main__':
    if(not os.path.exists(r'.\Tn_begin_initial_interpolation_extrapolation_longtime')):
        makedirs(r'.\Tn_begin_initial_interpolation_extrapolation_longtime')

    log_filename = r'Tn_begin_initial_interpolation_extrapolation_longtime/Tn_begin_initial_heat.txt'
    logging.basicConfig(filename=log_filename, level=logging.DEBUG, format='%(asctime)s %(message)s',
                        datefmt='%Y/%m/%d %I:%M:%S %p')
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    filename = r'results/heat/grid/result_ndcn_grid_1217-194321_D1.pth'
    Tn_begin_initial_interpolation_extrapolation_longtime_heat_result(filename, seed=876, init_random=True, T=60, time_tick=1200)

