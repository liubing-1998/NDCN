# coding:utf-8

import os
import sys
import re
import csv
import torch
import numpy as np

def read_result():
    '''
    'args': args.__dict__,
    'v_iter': [],
    'abs_error': [],
    'rel_error': [],
    'true_y': [solution_numerical.squeeze().t()],
    'predict_y': [],
    'abs_error2': [],
    'rel_error2': [],
    'predict_y2': [],
    'abs_error3': [],
    'rel_error3': [],
    'predict_y3': [],
    'model_state_dict': [],
    'total_time': [],
    # add
    'A':[]
    'id_train':[]
    'id_test':[]
    'id_test2':[]
    'id_test3':[]
    'seed':[]
    'num_paras':[]    # 参数量
    'pred_result':
    't':
    '''
    path = r'results/HeatDynamics/grid/result_HeatDynamics_grid_1216-224712_D0.pth'
    results = torch.load(path)
    # print(results['true_y'][0].shape)
    # print(results['true_y'][0])
    # print(results['id_test'][0])
    # print(results['true_y'][0][:, 0].shape)
    # print(results['predict_y'][0].shape)
    # print(results['predict_y2'][0].shape)
    # print(results['rel_error'])
    # print(results['seed'][0])
    # print(results['t'])
    # print(results['model_state_dict'])
    # print(results['A'])
    # print(results['A'][0].shape)
    # for key in results.keys():
    #     print(key)


def read_file_name(file_dir):
    root = ""
    files = ""
    for root, dirs, files in os.walk(file_dir):
        # print(root)  # 当前目录路径
        # print(dirs)  # 当前路径下所有子目录
        # print(files)  # 当前路径下所有非目录子文件
        pass
    return root, files

def read_log_RESULT(root, file_name_list, count):
    '''

    Args:
        root: log文件存放目录
        file_name_list: log文件名list
        count: 同一网络动态下重复实验次数，每count次计算一次均值方差

    Returns:

    '''
    file_result_list = []
    file_result_list_count = []
    for i in range(len(file_name_list)):
        log_path = os.path.join(root, file_name_list[i])
        with open(log_path, "r", encoding='utf-8') as f:
            file = f.read()  # RESULT
            batch = re.split('RESULT ', file)
            batch = re.split('Time', batch[1])
            data = re.findall(r'2000\| Train Loss (.*?)\((.*?) Relative\) \| Test Loss (.*?)\((.*?) Relative\) \| Test Loss2 (.*?)\((.*?) Relative\) \| Test Loss3 (.*?)\((.*?) Relative.*?', batch[0], re.S)
            temp_list = [file_name_list[i]]
            temp_list.extend(list(data[0]))
            file_result_list_count.append(temp_list)

        if (i+1) % count == 0:  # 计算file_result_list_count里的方差均值，合并到file_result_list中并清空file_result_list_count
            '''
            新增加上将均值方差保留三位小数，并写成mean±std的形式，追加到file_result_list
            '''
            result_array = np.zeros((count, 8))  # count行，8列结果
            for ii in range(count):
                for jj in range(8):
                    result_array[ii][jj] = file_result_list_count[ii][jj+1]
            # 获取当前日志名字
            log_mean_name = [file_name_list[i-1][0:-6] + "_mean"]
            log_var_name = [file_name_list[i-1][0:-6] + "_std"]
            log_mean_var_name = [file_name_list[i - 1][0:-6] + "_mean_std"]
            log_mean = log_mean_name + np.mean(result_array, axis=0).tolist()
            # log_var = log_var_name + np.var(result_array, axis=0).tolist()  # 方差
            log_std = log_var_name + np.std(result_array, axis=0).tolist()  # 标准差
            # log_mean_three和log_var_three保留三位小数
            log_mean_three = log_mean_name + np.around(np.mean(result_array, axis=0), 3).tolist()
            log_var_three = log_var_name + np.around(np.std(result_array, axis=0), 3).tolist()
            log_mean_var_three = log_mean_var_name + [(str(log_mean_three[i]) + '+' + str(log_var_three[i])) for i in range(1, 9)]
            # log_mean_var_three_percent = [(str(log_mean_three[i]*100) + "±" + str(log_var_three[i]*100)) for i in range(1, 7)]
            file_result_list_count.append(log_mean)
            file_result_list_count.append(log_std)
            file_result_list_count.append(log_mean_three)
            file_result_list_count.append(log_var_three)
            file_result_list_count.append(log_mean_var_three)
            # file_result_list_count.append(log_mean_var_three_percent)
            file_result_list += file_result_list_count
            file_result_list_count = []
            file_result_list.append([])
    # print(file_result_list)
    with open('Dlog_all_ndcn_result_add.csv', 'a', encoding='utf-8', newline='') as f:
        # 2. 基于文件对象构建 csv写入对象
        csv_writer = csv.writer(f)
        # 3. 构建列表头
        csv_writer.writerow(['log', 'TrainLoss', 'TrainLossRelative', 'TestLoss', 'TestLossRelative', 'TestLoss2', 'TestLoss2Relative', 'TestLoss3', 'TestLoss3Relative'])
        # 4. 写入csv文件内容
        for i in range(len(file_result_list)):
            csv_writer.writerow(file_result_list[i])
    pass

if __name__ == '__main__':
    # file_dir = r'E:\code_and_data_package\PDE-FIND-master-MLP-3-test\log'
    # file_dir = r'.\server_result\log'
    file_dir = r'.\log'
    root, file_name_list = read_file_name(file_dir)
    read_log_RESULT(root, file_name_list, 10)

    # read_result()

