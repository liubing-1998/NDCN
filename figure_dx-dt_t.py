# coding:utf-8

import os
import sys
import random
import logging
import argparse
from copy import deepcopy
import plotly.express as px
import plotly

import pandas as pd
from sklearn.decomposition import PCA

import umap

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


class HeatDiffusion(nn.Module):
    # In this code, row vector: y'^T = y^T A^T      textbook format: column vector y' = A y
    def __init__(self, L, k=0.1):
        super(HeatDiffusion, self).__init__()
        self.L = -L  # Diffusion operator
        self.k = k  # heat capacity

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

def dx_dt_figure_heat_result(filename, seed=876, init_random=True, T=60, time_tick=1200):
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

    results = torch.load(filename)
    # print(results['seed'])
    A = results['A'][0]
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
    # true_y = results['true_y'][0]

    '''# Initial Value
    x0 = torch.zeros(N, N)
    x0[int(0.05 * N):int(0.25 * N), int(0.05 * N):int(0.25 * N)] = 25  # x0[1:5, 1:5] = 25  for N = 20 or n= 400 case
    x0[int(0.45 * N):int(0.75 * N), int(0.45 * N):int(0.75 * N)] = 20  # x0[9:15, 9:15] = 20 for N = 20 or n= 400 case
    x0[int(0.05 * N):int(0.25 * N), int(0.35 * N):int(0.65 * N)] = 17  # x0[1:5, 7:13] = 17 for N = 20 or n= 400 case
    # 随机生成数据
    if init_random:
        x0 = 25 * torch.rand(N, N)
        print("x0随机初始化（0-25均匀分布初始化）")
    x0 = x0.view(-1, 1).float()
    x0 = x0.to(device)'''

    logging.info("T=" + str(T))
    logging.info("time_tick=" + str(time_tick))
    sampled_time = 'irregular'
    if sampled_time == 'equal':
        print('Build Equally-sampled -time dynamics')
        t = torch.linspace(0., T, time_tick)  # time_tick) # 100 vector
    elif sampled_time == 'irregular':
        print('Build irregularly-sampled -time dynamics')
        # irregular time sequence
        sparse_scale = 10
        t = torch.linspace(0., T, time_tick * sparse_scale)  # 100 * 10 = 1000 equally-sampled tick
        t = np.random.permutation(t)[:int(time_tick * 1.2)]
        t = torch.tensor(np.sort(t))
        t[0] = 0

    with torch.no_grad():
        solution_numerical = ode.odeint(HeatDiffusion(L), x0, t, method='dopri5') # shape: 1000 * 1 * 2
        print("choice HeatDynamics")
        print(solution_numerical.shape)
        logging.info("choice HeatDynamics")
    true_y = solution_numerical.squeeze().t().to(device)  # 120 * 1 * 400  --squeeze--> 120 * 400 -t-> 400 * 120
    true_y0 = x0.to(device)


    # criterion = F.l1_loss
    # pred_y = model(t, true_y0).squeeze().t()  # odeint(model, true_y0, t)

    # 生成绘图需要的数据
    print("绘图数据生成（模型）")
    Xht_input = model.input_layer(true_y0)
    integration_time_vector = t.type_as(Xht_input)
    Xht_output = ode.odeint(model.neural_dynamic_layer.odefunc, Xht_input, integration_time_vector, rtol=0.01, atol=0.001, method='euler')  # torch.Size([100, 400, 20])
    # print(Xht_output.shape)  # torch.Size([1440, 400, 20])
    # hvx = model.neural_dynamic_layer(t, Xht_input)
    # print(hvx.shape)  # torch.Size([1440, 400, 20])
    # output = model.output_layer(hvx)
    # print(output.shape)  # torch.Size([1440, 400, 1])
    hvx = model.neural_dynamic_layer(t, Xht_input)
    # print(hvx.shape)  # torch.Size([1440, 400, 20])
    output = model.output_layer(hvx)  # torch.Size([1440, 400, 1])

    ##########绘制dx_dt-x导数图
    ######绘制模型的
    hx_dt = torch.zeros(Xht_output.shape)
    for i in range(Xht_output.shape[0]):
        hx_output = model.neural_dynamic_layer.odefunc(t, Xht_output[i, :, :])
        hx_dt[i, :, :] = hx_output
    # 计算出来的导数值
    hx_dt = hx_dt.cpu().detach().numpy()  # (1440, 400, 20)
    # 过 model.output_layer
    dx_dt = model.output_layer(torch.tensor(hx_dt).to(device))  # torch.Size([1440, 400, 1])

    sample = results['t'][results['id_train'][0]]  # 训练集时间点
    # 绘制模型打印的导数图
    dx_dt_t_model_figure(output, dx_dt, t, sample, T, time_tick)

    # 绘制真实的导数图
    dx_dt_t_true_figure(true_y, L, t, sample, T, time_tick)

    # 绘制模型和真实的导数差值图
    dx_dt_t_difference_figure(true_y, dx_dt, L, t, sample, T, time_tick)


def dx_dt_t_model_figure(output, fx_dt, t, sample, T, time_tick):
    print("绘制fx-dt模型matplotlib线图用于保存")
    fig = plt.figure()
    ax = axisartist.Subplot(fig, 111)
    fig.add_axes(ax)
    ax.axis[:].set_visible(False)
    ax.axis["x"] = ax.new_floating_axis(0, -8)
    ax.axis["y"] = ax.new_floating_axis(1, 0)
    ax.axis["x"].set_axis_direction('bottom')
    ax.axis["y"].set_axis_direction('left')
    ax.axis["x"].set_axisline_style("->", size=0.3)
    ax.axis["y"].set_axisline_style("->", size=0.3)
    # # 设置坐标轴范围
    ax.set_xlim([0, 10])
    ax.set_xticks([i for i in np.arange(0, 11, 1)])
    ax.set_ylim([-8, 8])
    # 计算出来的导数值
    # print(fx_dt_true.shape)  # (400, 1440)
    # num = 240  # 控制取得时间长度, 默认应该是len(t) = 1440
    num = 1440
    for i in range(0, 100, 10):
        # ax.plot(output[:num, i].squeeze().cpu().detach().numpy(), fx_dt[:num, i].squeeze().cpu().detach().numpy(),
        #         label='$NDCN \enspace of \enspace dx{%d}/dt$' % i)
        ax.plot(t.cpu().detach().numpy(), fx_dt[:, i].squeeze().cpu().detach().numpy(), alpha=0.7, label='$NDCN \enspace of \enspace dx{%d}/dt$' % i)
        # print(fx_dt[:num, i])  # 真实大小应该达到10e8数量级
        ax.plot(sample, [-7] * len(sample), '|', linewidth=0.001, color='k')
    plt.axvline(x=5, color="black", linestyle="dashed", )
    plt.axvspan(0, 5, color='lightskyblue', alpha=0.3, lw=0)
    ax.axis["x"].label.set_text(r"$t$")
    ax.axis["y"].label.set_text(r"$\hat{dx/dt}$")
    ax.axis["x"].label.set_size(14)
    ax.axis["y"].label.set_size(14)
    # ax.legend(loc="best")
    # pic_path = r'.\\figure_dx-dt\\' + '_'.join(strtmp for strtmp in filename.split('/')[1:]).split('.')[0] \
    #            + "_derivative_true_withlegend_" + str(T) + "_" + str(time_tick) + ".png"
    pic_path = r'.\\figure_dx-dt_t\\' + '_'.join(strtmp for strtmp in filename.split('/')[1:]).split('.')[0] \
               + "_derivative_t_NDCN_10_" + str(T) + "_" + str(time_tick)
    plt.savefig(pic_path + ".png", bbox_inches='tight', dpi=1000)  # dpi=1000可调分辨率参数
    plt.savefig(pic_path + ".pdf", bbox_inches='tight', dpi=1000)
    plt.close(fig)

def dx_dt_t_true_figure(true_y, L, t, sample, T, time_tick):
    print("绘制fx-dt真实matplotlib线图用于保存")
    fig = plt.figure()
    ax = axisartist.Subplot(fig, 111)
    fig.add_axes(ax)
    ax.axis[:].set_visible(False)
    ax.axis["x"] = ax.new_floating_axis(0, -8)
    ax.axis["y"] = ax.new_floating_axis(1, 0)
    ax.axis["x"].set_axis_direction('bottom')
    ax.axis["y"].set_axis_direction('left')
    ax.axis["x"].set_axisline_style("->", size=0.3)
    ax.axis["y"].set_axisline_style("->", size=0.3)
    # # 设置坐标轴范围
    ax.set_xlim([0, 10])
    ax.set_xticks([i for i in np.arange(0, 11, 1)])
    # ax.set_ylim([-25, 26])
    ax.set_ylim([-8, 8])
    dx_dt_true = np.zeros(true_y.shape)
    # print(true_y.shape)  # torch.Size([400, 1440])
    for i in range(true_y.shape[1]):
        hx_output = HeatDiffusion(L)(t, true_y[:, i].unsqueeze(0).t())
        # print(hx_output.shape)  # torch.Size([400, 1])
        dx_dt_true[:, i] = hx_output.cpu().detach().numpy().squeeze()
    # 计算出来的导数值
    # print(fx_dt_true.shape)  # (400, 1440)
    # print(true_y)
    # print(dx_dt_true)
    for i in range(0, 100, 10):
        # ax.plot(true_y[i, :].cpu().detach().numpy(), dx_dt_true[i, :], label='$True \enspace of \enspace dx{%d}/df$' % i)
        ax.plot(t.cpu().detach().numpy(), dx_dt_true[i, :], alpha=0.7, label='$True \enspace of \enspace dx{%d}/df$' % i)
        ax.plot(sample, [-7] * len(sample), '|', linewidth=0.001, color='k')
    plt.axvline(x=5, color="black", linestyle="dashed", )
    plt.axvspan(0, 5, color='lightskyblue', alpha=0.3, lw=0)
    ax.axis["x"].label.set_text(r"$t$")
    ax.axis["y"].label.set_text(r"$dx/dt$")
    ax.axis["x"].label.set_size(14)
    ax.axis["y"].label.set_size(14)
    # ax.legend(loc="best")
    # pic_path = r'.\\figure_dx-dt\\' + '_'.join(strtmp for strtmp in filename.split('/')[1:]).split('.')[0] \
    #            + "_derivative_true_withlegend_" + str(T) + "_" + str(time_tick) + ".png"
    pic_path = r'.\\figure_dx-dt_t\\' + '_'.join(strtmp for strtmp in filename.split('/')[1:]).split('.')[0] \
               + "_derivative_t_true_10_" + str(T) + "_" + str(time_tick)
    plt.savefig(pic_path + ".png", bbox_inches='tight', dpi=1000)  # dpi=1000可调分辨率参数
    plt.savefig(pic_path + ".pdf", bbox_inches='tight', dpi=1000)
    plt.close(fig)

def dx_dt_t_difference_figure(true_y, fx_dt, L, t, sample, T, time_tick):
    print("绘制fx-dt真实matplotlib线图用于保存")
    fig = plt.figure()
    ax = axisartist.Subplot(fig, 111)
    fig.add_axes(ax)
    ax.axis[:].set_visible(False)
    ax.axis["x"] = ax.new_floating_axis(0, -8)
    ax.axis["y"] = ax.new_floating_axis(1, 0)
    ax.axis["x"].set_axis_direction('bottom')
    ax.axis["y"].set_axis_direction('left')
    ax.axis["x"].set_axisline_style("->", size=0.3)
    ax.axis["y"].set_axisline_style("->", size=0.3)
    # # 设置坐标轴范围
    ax.set_xlim([0, 10])
    ax.set_xticks([i for i in np.arange(0, 11, 1)])
    # ax.set_ylim([-25, 26])
    ax.set_ylim([-8, 8])
    dx_dt_true = np.zeros(true_y.shape)
    # print(true_y.shape)  # torch.Size([400, 1440])
    for i in range(true_y.shape[1]):
        hx_output = HeatDiffusion(L)(t, true_y[:, i].unsqueeze(0).t())
        # print(hx_output.shape)  # torch.Size([400, 1])
        dx_dt_true[:, i] = hx_output.cpu().detach().numpy().squeeze()
    # 计算出来的导数值
    # print(fx_dt_true.shape)  # (400, 1440)
    # print(true_y)
    # print(dx_dt_true)
    num = 1440
    for i in range(0, 100, 10):
        # ax.plot(true_y[i, :].cpu().detach().numpy(), dx_dt_true[i, :], label='$True \enspace of \enspace dx{%d}/df$' % i)
        ax.plot(t.cpu().detach().numpy(), fx_dt[:, i].squeeze().cpu().detach().numpy()-dx_dt_true[i, :], alpha=0.7, label='$True \enspace of \enspace dx{%d}/df$' % i)
        ax.plot(sample, [-7] * len(sample), '|', linewidth=0.001, color='k')
    plt.axvline(x=5, color="black", linestyle="dashed", )
    plt.axvspan(0, 5, color='lightskyblue', alpha=0.3, lw=0)
    ax.axis["x"].label.set_text(r"$t$")
    ax.axis["y"].label.set_text(r"$error \enspace of \enspace dx/dt$")
    ax.axis["x"].label.set_size(14)
    ax.axis["y"].label.set_size(14)
    # ax.legend(loc="best")
    # pic_path = r'.\\figure_dx-dt\\' + '_'.join(strtmp for strtmp in filename.split('/')[1:]).split('.')[0] \
    #            + "_derivative_true_withlegend_" + str(T) + "_" + str(time_tick) + ".png"
    pic_path = r'.\\figure_dx-dt_t\\' + '_'.join(strtmp for strtmp in filename.split('/')[1:]).split('.')[0] \
               + "_different_t_true_10_" + str(T) + "_" + str(time_tick)
    plt.savefig(pic_path + ".png", bbox_inches='tight', dpi=1000)  # dpi=1000可调分辨率参数
    plt.savefig(pic_path + ".pdf", bbox_inches='tight', dpi=1000)
    plt.close(fig)

if __name__ == '__main__':
    if(not os.path.exists(r'.\figure_dx-dt_t')):
        makedirs(r'.\figure_dx-dt_t')

    log_filename = r'figure_dx-dt_t/heat_dx-dt_t.txt'
    logging.basicConfig(filename=log_filename, level=logging.DEBUG, format='%(asctime)s %(message)s',
                        datefmt='%Y/%m/%d %I:%M:%S %p')
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    filename = r'results/heat/grid/result_ndcn_grid_1217-194321_D1.pth'
    dx_dt_figure_heat_result(filename, seed=876, init_random=True, T=60, time_tick=1200)
