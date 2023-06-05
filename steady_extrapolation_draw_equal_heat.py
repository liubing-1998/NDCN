# coding:utf-8

import os
import sys
import random
import logging
import argparse
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits import axisartist

from neural_dynamics import *
from utils_in_learn_dynamics import *
import torchdiffeq as ode

import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def steady_extrapolation_draw_equal_heat(filename, T, time_tick, n, init_random):
    logging.info("============================================")
    pic_path = '_'.join(strtmp for strtmp in filename.split('/')[1:]).split('.')[0] + "_extra_" + str(T) + "_" + str(time_tick)
    logging.info(pic_path)

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
    logging.info("init_random="+str(init_random))

    # 生成新的网络和初始化
    # Build network # A: Adjacency matrix, L: Laplacian Matrix,  OM: Base Operator
    n = n  # e.g nodes number 400
    N = int(np.ceil(np.sqrt(n)))  # grid-layout pixels :20
    # Initial Value
    # 随机生成数据
    if init_random:
        x0 = 25 * torch.rand(N, N)
        print("x0随机初始化（0-25均匀分布初始化）")
    else:
        x0 = torch.zeros(N, N)
        x0[int(0.05 * N):int(0.25 * N), int(0.05 * N):int(0.25 * N)] = 25  # x0[1:5, 1:5] = 25  for N = 20 or n= 400 case
        x0[int(0.45 * N):int(0.75 * N), int(0.45 * N):int(0.75 * N)] = 20  # x0[9:15, 9:15] = 20 for N = 20 or n= 400 case
        x0[int(0.05 * N):int(0.25 * N), int(0.35 * N):int(0.65 * N)] = 17  # x0[1:5, 7:13] = 17 for N = 20 or n= 400 case

    x0 = x0.view(-1, 1).float()
    x0 = x0.to(device)

    # A = torch.tensor([])
    # if network == 'grid':
    #     print("Choose graph: " + network)
    #     A = grid_8_neighbor_graph(N)
    #     G = nx.from_numpy_array(A.numpy())
    # elif network == 'random':
    #     print("Choose graph: " + network)
    #     G = nx.erdos_renyi_graph(n, 0.02, seed=seed)  # 0.1  0.02
    #     G = networkx_reorder_nodes(G, layout)
    #     A = torch.FloatTensor(nx.to_numpy_array(G))
    # elif network == 'power_law':
    #     print("Choose graph: " + network)
    #     G = nx.barabasi_albert_graph(n, 5, seed=seed)  # 5
    #     G = networkx_reorder_nodes(G, layout)
    #     A = torch.FloatTensor(nx.to_numpy_array(G))
    # elif network == 'small_world':
    #     print("Choose graph: " + network)
    #     G = nx.newman_watts_strogatz_graph(n, 5, 0.5, seed=seed)
    #     G = networkx_reorder_nodes(G, layout)
    #     A = torch.FloatTensor(nx.to_numpy_array(G))
    # elif network == 'community':
    #     print("Choose graph: " + network)
    #     n1 = int(n / 3)
    #     n2 = int(n / 3)
    #     n3 = int(n / 4)
    #     n4 = n - n1 - n2 - n3
    #     G = nx.random_partition_graph([n1, n2, n3, n4], .04, .005, seed=seed)  # .25, .01
    #     G = networkx_reorder_nodes(G, layout)
    #     A = torch.FloatTensor(nx.to_numpy_array(G))
    # A = A.to(device)
    results = torch.load(filename)
    A = results['A'][0]
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

    # print(input_size, hidden_size, operator, num_classes, dropout, rtol, atol, method)
    # logging.info(str(input_size)+str(hidden_size)+operator+str(num_classes)+str(dropout)+str(rtol)+str(atol)+method)
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

    # equally-sampled time
    # sampled_time = 'irregular'
    sampled_time = 'equal'
    logging.info("T=" + str(T))
    logging.info("time_tick=" + str(time_tick))
    if sampled_time == 'equal':
        print('Build Equally-sampled -time dynamics')
        t = torch.linspace(0., T, time_tick+1)  # time_tick) # 100 vector
        # print(t)
        # print(t.shape)
        # sys.exit()

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


    with torch.no_grad():
        solution_numerical = ode.odeint(HeatDiffusion(L), x0, t, method='dopri5') # shape: 1000 * 1 * 2
        print("choice HeatDynamics")
        print(solution_numerical.shape)
        logging.info("choice HeatDynamics")

    true_y = solution_numerical.squeeze().t().to(device)  # 120 * 1 * 400  --squeeze--> 120 * 400 -t-> 400 * 120
    true_y0 = x0.to(device)  # 400 * 1

    #####################################
    criterion = F.l1_loss
    pred_y = model(t, true_y0).squeeze().t()  # odeint(model, true_y0, t)
    loss = criterion(pred_y, true_y)
    relative_loss = criterion(pred_y, true_y) / true_y.mean()
    print('RESULT Test Loss {:.6f}({:.6f} Relative)'.format(loss.item(), relative_loss.item()))
    logging.info('RESULT Test Loss {:.6f}({:.6f} Relative)'.format(loss.item(), relative_loss.item()))


    # 绘制曲线图
    # 单个节点画图
    x = t
    predY = pred_y[1].cpu().detach().numpy()
    trueY = true_y[1].cpu().detach().numpy()

    np.savez(r'.\\steady_extrapolation_draw_equal_heat\\' + 'NDCN_true_pred_t_5_100', ndcn_predY=predY, ndcn_trueY=trueY, ndcn_t=t.numpy())

    # 设置全局字体，字体大小（好像只对text生效）
    plt.rc('font', family='Times New Roman', size=16)
    plt.rc('lines', linewidth=2)  # 设置全局线宽

    fig = plt.figure()
    ax = axisartist.Subplot(fig, 111)
    fig.add_axes(ax)
    ax.axis[:].set_visible(False)
    # ax.axis["x"] = ax.new_floating_axis(0, -1.5)
    ax.axis["x"] = ax.new_floating_axis(0, 0)
    ax.axis["y"] = ax.new_floating_axis(1, 0)
    ax.axis["x"].set_axis_direction('bottom')
    ax.axis["y"].set_axis_direction('left')
    ax.axis["x"].set_axisline_style("->", size=0.3)
    ax.axis["y"].set_axisline_style("->", size=0.3)
    # 设置坐标轴范围
    # ax.set_xlim([0, 5.2])
    # ax.set_ylim([0, 9.2])
    ax.plot(x, trueY, color="red", linestyle="solid", label='$True \enspace of \enspace x_{i}$')
    ax.plot(x, predY, color="blue", linestyle="dashed", label='$Predict \enspace of \enspace x_{i}$')
    ax.axis["x"].label.set_text(r"$t$")
    ax.axis["y"].label.set_text(r"$x_{i}$")
    ax.axis["x"].label.set_size(16)
    ax.axis["y"].label.set_size(16)
    ax.legend(loc="best")
    pic_path = r'.\\steady_extrapolation_draw_equal_heat\\' + '_'.join(strtmp for strtmp in filename.split('/')[1:]).split('.')[0]\
               + "_extra_" + str(T) + "_" + str(time_tick) + "_one_node"
    plt.savefig(pic_path+".png", bbox_inches='tight', dpi=1000)  # dpi=1000可调分辨率参数
    plt.savefig(pic_path+".pdf", bbox_inches='tight', dpi=1000)
    plt.close(fig)

    # 多个节点绘制立体图
    # print(t.shape)  # torch.Size([120])
    # print(pred_y.shape)  # torch.Size([400, 120])
    # print(true_y.shape)  # torch.Size([400, 120])
    true_y_all = true_y[0:50].cpu().detach().numpy()
    pred_y_all = pred_y[0:50].cpu().detach().numpy()
    # 真实的和预测的画在两张图上
    fig = plt.figure()
    ax0 = fig.add_subplot(111, projection='3d')
    zmin = true_y[:, 0].min()
    zmax = true_y[:, 0].max()
    X = t
    Y = np.arange(50)
    X, Y = np.meshgrid(X, Y)
    surf = ax0.plot_surface(X, Y, true_y_all, cmap='rainbow', linewidth=0, antialiased=False, vmin=zmin, vmax=zmax)
    ax0.set_xlabel("$t$")
    ax0.set_ylabel("$i$")
    ax0.set_zlabel("$x_{i}(t)$")
    pic_path = r'.\\steady_extrapolation_draw_equal_heat\\' + '_'.join(strtmp for strtmp in filename.split('/')[1:]).split('.')[
        0] + "_extra_" + str(T) + "_" + str(time_tick) + "_50_node_true"
    plt.savefig(pic_path+".png", bbox_inches='tight', dpi=1000)
    plt.savefig(pic_path+".pdf", bbox_inches='tight', dpi=1000)
    plt.close()

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    zmin = pred_y_all[:, 0].min()
    zmax = pred_y_all[:, 0].max() * 1.1
    surf2 = ax1.plot_surface(X, Y, pred_y_all, cmap='rainbow', linewidth=0, antialiased=False,  vmin=zmin, vmax=zmax)  #
    ax1.set_xlabel("$t$")
    ax1.set_ylabel("$i$")
    ax1.set_zlabel("$x_{i}(t)$")
    pic_path = r'.\\steady_extrapolation_draw_equal_heat\\' + '_'.join(strtmp for strtmp in filename.split('/')[1:]).split('.')[
        0] + "_extra_" + str(T) + "_" + str(time_tick) + "_50_node_pred"
    plt.savefig(pic_path+".png", bbox_inches='tight', dpi=1000)
    plt.savefig(pic_path+".pdf", bbox_inches='tight', dpi=1000)
    plt.close()

    # 绘制差值二维图像,纵坐标为节点个数，横坐标为时间轴，值为预测值和真实值之差
    y = np.arange(0, 400, 1)
    x = t.numpy()
    X, Y = np.meshgrid(x, y)
    Z = true_y.cpu().detach().numpy() - pred_y.cpu().detach().numpy()
    fig, ax = plt.subplots()
    levels = np.arange(Z.min(), Z.max(), (Z.max() - Z.min()) / 200)
    ctf = ax.contourf(X, Y, Z, levels=levels, cmap=plt.cm.bwr)
    ax.set_ylabel("$x_{i}$", fontsize=14)
    ax.set_xlabel("$t$", fontsize=14)
    cb = fig.colorbar(ctf, ax=ax)  # 绘制颜色条
    pic_path = r'.\\steady_extrapolation_draw_equal_heat\\' + '_'.join(strtmp for strtmp in filename.split('/')[1:]).split('.')[
        0] + "_extra_" + str(T) + "_" + str(time_tick) + "_difference_value"
    plt.savefig(pic_path+".png", bbox_inches='tight', dpi=1000)
    plt.savefig(pic_path+".pdf", bbox_inches='tight', dpi=1000)
    plt.close()


if __name__ == '__main__':
    if(not os.path.exists(r'.\steady_extrapolation_draw_equal_heat')):
        makedirs(r'.\steady_extrapolation_draw_equal_heat')

    log_filename = r'steady_extrapolation_draw_equal_heat/heat_extrapolation.txt'
    logging.basicConfig(filename=log_filename, level=logging.DEBUG, format='%(asctime)s %(message)s',
                        datefmt='%Y/%m/%d %I:%M:%S %p')
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    filename = r'results/heat/grid/result_ndcn_grid_1217-194050_D0.pth'
    steady_extrapolation_draw_equal_heat(filename, T=50, time_tick=100, n=400, init_random=False)

    # network = ['grid', 'random', 'power_law', 'small_world', 'community']
    # for filename in filepath:
    #     for net in network:
    #         transfer_heat_result(filename, n=400, network=net, seed=2022, layout='community', init_random=False)
    # n目前得选成一个数得平方值，不然报错，数据维度对不上，因为程序画图的时候，画的是方格，对节点数开根号了
