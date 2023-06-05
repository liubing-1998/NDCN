# coding:utf-8

import os
import sys
import math
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import mpl_toolkits.axisartist as axisartist

weight_list = [2, 3, 4, 5, 10, 100, 500]
font = {'size': 20}
matplotlib.rc('font', **font)
# plt.style.use("ggplot")
if(not os.path.exists(r'.\figure_weight') ):
    os.makedirs(r'.\figure_weight')
def hyperparameter_T_of_weight():
    epoch = np.arange(1, 2000, 0.1)
    y = []
    for i in epoch:
        if(i<100):
            y.append(weight_list[0])
        elif(100<=i<200):
            y.append(weight_list[1])
        elif (200 <= i < 300):
            y.append(weight_list[2])
        elif (300 <= i < 400):
            y.append(weight_list[3])
        elif (400 <= i < 800):
            y.append(weight_list[4])
        elif (800 <= i < 1200):
            y.append(weight_list[5])
        elif (1200 <= i):
            y.append(weight_list[6])

    # 设置全局字体，字体大小（好像只对text生效）
    plt.rc('font', family='Times New Roman', size=24)
    plt.rc('lines', linewidth=2)  # 设置全局线宽

    fig = plt.figure()
    ax = axisartist.Subplot(fig, 111)
    fig.add_axes(ax)
    ax.axis[:].set_visible(False)

    ax.axis["x"] = ax.new_floating_axis(0, 0)
    ax.axis["y"] = ax.new_floating_axis(1, 0)
    ax.axis["x"].set_axis_direction('bottom')
    ax.axis["y"].set_axis_direction('left')
    ax.axis["x"].set_axisline_style("->", size=0.3)
    ax.axis["y"].set_axisline_style("->", size=0.3)

    ax.set_xlim([0, 2100])
    ax.set_ylim([0, 520])
    ax.set_xticks([0, 500, 1000, 1500, 2000])
    ax.set_yticks([100, 200, 300, 400, 500])

    ax.plot(epoch, y)
    # ax.set_title("$the \enspace hyperparameter \enspace \u03C4 \enspace of \enspace weight$")
    ax.axis["x"].label.set_text("$epochs$")
    ax.axis["y"].label.set_text("$\u03C4$")

    # ax.set_xlabel('epochs')
    # ax.set_ylabel('\u03C4')

    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(r'.\figure_weight\the hyperparameter T of weight.png', bbox_inches='tight', dpi=1000)  # dpi=1000可调分辨率参数
    plt.savefig(r'.\figure_weight\the hyperparameter T of weight.pdf', transparent=True, dpi=1000)


def weight_with_time():
    t = np.arange(0, 5, 0.01)
    weight_result = []
    for i in range(len(weight_list)):
        weight_tmp = []
        for item in t:
            weight_tmp.append(math.exp(-1*item/weight_list[i]))
        weight_result.append(weight_tmp)

    # 设置全局字体，字体大小（好像只对text生效）
    plt.rc('font', family='Times New Roman', size=30)
    plt.rc('lines', linewidth=2)  # 设置全局线宽

    fig = plt.figure()
    ax = axisartist.Subplot(fig, 111)
    fig.add_axes(ax)
    ax.axis[:].set_visible(False)

    ax.axis["x"] = ax.new_floating_axis(0, 0)
    ax.axis["y"] = ax.new_floating_axis(1, 0)
    ax.axis["x"].set_axis_direction('bottom')
    ax.axis["y"].set_axis_direction('left')
    ax.axis["x"].set_axisline_style("->", size=0.3)
    ax.axis["y"].set_axisline_style("->", size=0.3)

    ax.set_xlim([0, 5.15])
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_ylim((0, 1.04))

    # fig, ax = plt.subplots()
    ax.plot(t, weight_result[0], color="#FFFF00", label=r'$\tau$='+str(weight_list[0]))   # (255, 255, 0)
    ax.plot(t, weight_result[1], color="#FFDB00", label=r'$\tau$=' + str(weight_list[1]))  # (255, 219, 0)
    ax.plot(t, weight_result[2], color="#FFB600", label=r'$\tau$=' + str(weight_list[2]))  # (255, 182, 0)
    ax.plot(t, weight_result[3], color="#FF9200", label=r'$\tau$=' + str(weight_list[3]))  # (255, 146, 0)
    ax.plot(t, weight_result[4], color="#FF6D00", label=r'$\tau$=' + str(weight_list[4]))  # (255, 109, 0)
    ax.plot(t, weight_result[5], color="#FF4900", label=r'$\tau$=' + str(weight_list[5]))  # (255, 73, 0)
    ax.plot(t, weight_result[6], color="#FF2400", label=r'$\tau$=' + str(weight_list[6]))  # (255, 36, 0)
    # ax.set_title("weight with time")  # 添加标题，调整字符大小  , fontdict={"fontsize": 10}
    # ax.set_xlabel("t")  # 添加横轴标签
    # ax.set_ylabel("weight_value")  # 添加纵轴标签
    ax.axis["x"].label.set_text("$t$")
    # ax.axis["y"].label.set_text("$weight \enspace value$")
    ax.axis["y"].label.set_text("$w$")
    # ax.legend(loc="upper left", prop={'size': 20}, mode="expand", ncol=4)  # 展示图例

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 22})

    # plt.show()
    plt.tight_layout()
    plt.savefig(r'.\figure_weight\weight with time.png', bbox_inches='tight', dpi=1000)  # dpi=1000可调分辨率参数
    plt.savefig(r'.\figure_weight\weight with time.pdf', transparent=True, dpi=1000)


if __name__ == '__main__':
    hyperparameter_T_of_weight()
    weight_with_time()
    pass
