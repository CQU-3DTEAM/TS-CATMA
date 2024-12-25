import glob
import os
import numpy as np
import pandas as pd
import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from matplotlib import pyplot as plt
import pywt
from scipy import stats
from scipy.signal import medfilt
from torch.autograd import Variable


def normalize(data):
    # 对每个维度分别进行归一化
    normalized_data = stats.zscore(data, axis=0)
    return normalized_data


def preprocess(data):
    filt = np.empty_like(data)
    for i in range(len(data)):
        filt[i] = medfilt(data[i], 9)
    processed_data = filt
    return processed_data


def load_lc_data_1D(path, indices=None, split=False):
    if indices is None:
        indices = [0]
    txt_Paths = glob.glob(os.path.join(path, '*.txt'))
    txt_Paths.sort()
    sensor_data = []
    sensor_label = []
    for txt_item in txt_Paths:
        label = txt_item.split('-')[1]
        data_z = pd.read_csv(txt_item, delimiter=',', header=None)
        data_z = np.asarray(data_z)
        data = data_z.reshape((22, 360))
        # data = preprocess(data)
        # sensor array optimization
        # if indices:
        if len(indices) > 0:
            data = data[indices]
            data = data.reshape(len(indices), 360)

        # print(data.shape, type(data))
        if (label == 'health'):
            label_c = 0
        else:
            label_c = 1

        sensor_data.append(data)
        sensor_label.append(label_c)

    _data = np.array(sensor_data, dtype="float")
    _label = np.array(sensor_label, dtype="float")
    print(_data.shape)

    if split == True:
        # train,val-array,list
        from sklearn.model_selection import train_test_split
        x_train, x_val, y_train, y_val = train_test_split(
            _data, _label,
            test_size=0.1,
            random_state=20,  # results can re
            shuffle=True,
            stratify=_label
        )
        return x_train, x_val, y_train, y_val
    else:
        return _data, _label


def lc_data_loader():
    root = "data_dir"
    train_dir = root + "sensor_data/lungcancer/afterprocess_360_txt1/train"
    test_dir = root + "sensor_data/lungcancer/afterprocess_360_txt1/test"

    indices = np.arange(22, dtype=np.int8)
    # train_data, train_label = load_lc_data_1D(train_dir, indices)
    train_data, val_data, train_label, val_label = load_lc_data_1D(train_dir, indices, split=True)
    test_data, test_label = load_lc_data_1D(test_dir, indices)


    num_classes = 2
    train_data = torch.FloatTensor(train_data)
    val_data = torch.FloatTensor(val_data)
    test_data = torch.FloatTensor(test_data)
    # train_label = torch.FloatTensor(torch.eye(num_classes)[train_label])
    # test_label = torch.FloatTensor(torch.eye(num_classes)[test_label])

    wavelet = 'sym4'  # 使用Daubechies小波
    levels = 4  # 分解层数
    noise_std_by_level = [10,20,30,40]

    # 对数据进行滤波
    # train_wavelet_data = wavelet_filter(train_data, wavelet, levels)
    # test_wavelet_data = wavelet_filter(test_data, wavelet, levels)
    train_wavelet_data_noise = add_noise(train_data , 'gaus', 50)
    val_wavelet_data_noise = add_noise(val_data , 'gaus', 50)
    test_wavelet_data_noise = add_noise(test_data, 'gaus', 50)
    # train_wavelet_data = wavelet_add_noise_by_level(train_data,noise_std_by_level=noise_std_by_level)
    # test_wavelet_data = wavelet_add_noise_by_level(test_data,noise_std_by_level=noise_std_by_level)

    # train_wavelet_data = torch.FloatTensor(train_wavelet_data)
    # test_wavelet_data = torch.FloatTensor(test_wavelet_data)
    train_wavelet_data_noise = torch.FloatTensor(train_wavelet_data_noise)
    val_wavelet_data_noise = torch.FloatTensor(val_wavelet_data_noise)
    test_wavelet_data_noise = torch.FloatTensor(test_wavelet_data_noise)

    # 定义替换的值
    replace_value = 1
    # 将独热编码后的标签中的1替换为replace_value，0替换为1-replace_value
    train_label = torch.LongTensor(train_label)
    eye_tensor = torch.eye(num_classes)
    eye_tensor = eye_tensor * replace_value + (1 - eye_tensor) * (1 - replace_value)
    train_label = eye_tensor[train_label]

    val_label = torch.LongTensor(val_label)
    eye_tensor = torch.eye(num_classes)
    eye_tensor = eye_tensor * replace_value + (1 - eye_tensor) * (1 - replace_value)
    val_label = eye_tensor[val_label]

    test_label = torch.LongTensor(test_label)
    eye_tensor = torch.eye(num_classes)
    eye_tensor = eye_tensor * replace_value + (1 - eye_tensor) * (1 - replace_value)
    test_label = eye_tensor[test_label]

    # Define Broken Data and Target Data for Reconstruction Task
    train_broken_data, train_target_data = train_wavelet_data_noise, train_data
    val_broken_data, val_target_data = val_wavelet_data_noise, val_data
    test_broken_data, test_target_data = test_wavelet_data_noise, test_data

    # train_broken_data = torch.cat([train_broken_data, val_broken_data, test_broken_data])
    # train_target_data = torch.cat([train_target_data, val_target_data, test_target_data])
    # train_label = torch.cat([train_label, val_label, test_label])
    # train_broken_data = torch.cat([train_broken_data, val_broken_data])
    # train_target_data = torch.cat([train_target_data, val_target_data])
    # train_label = torch.cat([train_label, val_label])

    train_dataset = Data.TensorDataset(train_broken_data, train_target_data,
                                       train_label)  ## Convert to PyTorch's unique data format
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=68, shuffle=True)

    val_dataset = Data.TensorDataset(val_broken_data, val_target_data,
                                     val_label)  ## Convert to PyTorch's unique data format
    val_loader = Data.DataLoader(dataset=val_dataset, batch_size=8, shuffle=True)
    # val_dataset = Data.TensorDataset(train_broken_data, train_target_data,
    #                                  train_label)  ## Convert to PyTorch's unique data format
    # val_loader = Data.DataLoader(dataset=val_dataset, batch_size=76, shuffle=True)

    test_dataset = Data.TensorDataset(test_broken_data, test_target_data, test_label)
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=9, shuffle=False)
    return train_loader, val_loader, test_loader


def exponential_moving_average_filter(data, alpha):
    smoothed_data = np.zeros_like(data)
    for i in range(data.shape[0]):  # 遍历每个样本
        smoothed_data[i, :, 0] = data[i, :, 0]  # 初始化每个样本的第一个数据点
        for j in range(1, data.shape[2]):  # 遍历每个样本的时间序列长度
            smoothed_data[i, :, j] = alpha * data[i, :, j] + (1 - alpha) * smoothed_data[i, :, j - 1]
    return smoothed_data


def wavelet_filter(data, wavelet, levels):
    filtered_datas = []
    for i in range(data.shape[0]):
        filtered_data = []
        for j in range(data.shape[1]):
            coeffs = pywt.wavedec(data[i, j, :], wavelet, level=levels)

            # 将高频系数置零进行滤波
            for k in range(1, len(coeffs)):
                coeffs[k] = np.zeros_like(coeffs[k])
            # coeffs[0] = np.zeros_like(coeffs[0])

            # 重构信号
            filtered_signal = pywt.waverec(coeffs, wavelet)
            filtered_data.append(filtered_signal)
        filtered_datas.append(filtered_data)
    # 转换为numpy数组方便后续处理
    filtered_datas = np.array(filtered_datas)
    return filtered_datas


def wavelet_add_noise_by_level(x, wavelet='db4', noise_std_by_level=None):
    """
    在小波域上对输入数据的小波系数按级别添加高斯噪声
    :param x: 输入数据 (num_channels, seq_len)
    :param wavelet: 小波基函数
    :param noise_std_by_level: 各级别噪声的标准差列表,从最粗级到最细级
    :return: 处理后的数据
    """
    if noise_std_by_level is None:
        noise_std_by_level = [25] * 5  # 默认每级噪声标准差为0.1

    num_samples,num_channels, seq_len = len(x),22, 360
    x_noisy = np.zeros_like(x)

    # 对每个通道进行小波变换
    for sample in range(num_samples):
        for i in range(num_channels):
            coeffs = pywt.wavedec(x[sample][i], wavelet, mode='periodic')

            # 对每个级别的小波系数添加高斯噪声
            coeffs_noisy = []
            for j, c in enumerate(coeffs):
                noise_std = noise_std_by_level[j] if j < len(noise_std_by_level) else 0.0
                noise = np.random.normal(0, noise_std, size=c.shape)
                c_noisy = c + noise
                coeffs_noisy.append(c_noisy)

            # 重构处理后的信号
            x_noisy[sample][i] = pywt.waverec(coeffs_noisy, wavelet, mode='periodic')

    return x_noisy


def add_noise(data, type='gaus', noise_factor=25):
    noisy_data = None
    # 添加高斯噪声
    if type == 'gaus':
        gaussian_noise = noise_factor * np.random.normal(loc=0.0, scale=1.0, size=data.shape)
        noisy_data = data + gaussian_noise

    # 添加均匀噪声
    if type == 'uni':
        uniform_noise = noise_factor * np.random.uniform(low=-1, high=1, size=data.shape)
        noisy_data = data + uniform_noise

    # 添加脉冲噪声
    if type == 'sp':
        sp_ratio = 0.05  # 噪声比率
        sp_noise = np.copy(data)
        num_salt = np.ceil(sp_ratio * data.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in data.shape]
        sp_noise[coords[0], coords[1]] = 1

        num_pepper = np.ceil(sp_ratio * data.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in data.shape]
        sp_noise[coords[0], coords[1]] = 0
        noisy_data = sp_noise
    noisy_data = np.array(noisy_data)
    return noisy_data


train_loader, val_loader, test_loader = lc_data_loader()
# for batch_idx, (sdata, tdata) in enumerate(zip(train_loader, test_loader)):
#     print(batch_idx, sdata[1].__len__(), tdata[1].__len__())


if __name__ == "__main__":
    root = "data_dir"
    train_dir = root + "sensor_data/lungcancer/afterprocess_360_txt1/train"
    test_dir = root + "sensor_data/lungcancer/afterprocess_360_txt1/test"

    indices = np.arange(22, dtype=np.int8)
    train_data, train_label = load_lc_data_1D(train_dir, indices)
    test_data, test_label = load_lc_data_1D(test_dir, indices)

    # 设置指数加权移动平均滤波的参数
    alpha = 0.2

    wavelet = 'sym4'  # 使用Daubechies小波
    levels = 4  # 分解层数
    # 对数据进行滤波
    smoothed_data = exponential_moving_average_filter(train_data, alpha)
    # wavelet_data = wavelet_filter(train_data, wavelet, levels)
    wavelet_data = wavelet_add_noise_by_level(train_data)
    noisy_data = add_noise(wavelet_data, 'gaus', 25)
    print("wavelet_data:", wavelet_data.shape)
    # 随机选择一个样本的一个时间序列进行可视化比较
    sample_index = np.random.randint(0, 100)
    sequence_index = np.random.randint(0, 22)
    plt.figure(figsize=(10, 5))
    plt.plot(train_data[sample_index, sequence_index], label='Original Data', marker='o', linestyle='--')
    plt.plot(smoothed_data[sample_index, sequence_index], label='Smoothed Data', marker='x', linestyle='-')
    plt.plot(wavelet_data[sample_index, sequence_index], label='Mean Data', marker='*', linestyle='-')
    plt.plot(noisy_data[sample_index, sequence_index], label='Noisy Data', marker='.', linestyle='-')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(f'Comparison of Original and Smoothed Data for Sample {sample_index}, Sequence {sequence_index}')
    plt.legend()
    plt.grid(True)
    plt.show()
