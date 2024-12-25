import argparse
import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.manifold import TSNE
from data_loader import train_loader, test_loader
from main import parse_option
from model import build_model
from tool import load_model
import warnings
from proplot import rc
warnings.filterwarnings("ignore",
                        message="Support for FigureCanvases without a required_interactive_framework attribute was deprecated in Matplotlib 3.6 and will be removed two minor releases later.")
rc["font.family"] = "Times New Roman"
# 统一设置轴刻度标签的字体大小
rc['tick.labelsize'] = 10
# 统一设置xy轴名称的字体大小
rc["axes.labelsize"] = 20
# 统一设置轴刻度标签的字体粗细
rc["axes.labelweight"] = "light"
# 统一设置xy轴名称的字体粗细
rc["tick.labelweight"] = "bold"

def plot_tsne(dataloader, title, model=None, func_name=None, perplexity=5, save_root=None):
    # 将数据集和标签分开
    data_list = []
    labels_list = []

    for batch_idx, data in enumerate(dataloader):
        data_lq, data_x, data_cl = data
        if model is None:
            flattened_data = data_x.view(data_x.shape[0], -1).numpy()
        else:
            feature_data = model_func(model, data_x, func_name)
            flattened_data = feature_data.view(feature_data.shape[0], -1).detach().numpy()

        data_list.append(flattened_data)
        labels_list.append(np.argmax(data_cl.numpy(), axis=1))  # 将独热编码标签转换为类别标签

    # 将数据转换为 numpy 数组
    data_array = np.concatenate(data_list, axis=0)
    labels_array = np.concatenate(labels_list, axis=0)

    # 使用 t-SNE 进行降维
    tsne = TSNE(n_components=2, perplexity=perplexity)
    embedded_data = tsne.fit_transform(data_array)

    # 获取所有类别标签
    unique_labels = np.unique(labels_array)

    # 绘制降维后的数据，按照类别标签使用不同颜色绘制散点图
    plt.figure(figsize=(8, 6))
    for label in unique_labels:
        indices = np.where(labels_array == label)[0]
        plt.scatter(embedded_data[indices, 0], embedded_data[indices, 1],
                    label=f'{"health" if label == 0 else "lungcancer"}')

    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    # lim_value = 80
    # plt.xlim((-lim_value, lim_value))
    # plt.ylim((-lim_value, lim_value))
    plt.title(title)
    plt.legend()
    if save_root is not None:
        save_dir = os.path.join(save_root, "data_plot")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{title.replace(' ', '_').replace(',', '')}.svg")
        plt.savefig(save_path, dpi=300, format="png")  # 保存图像为png文件
    plt.show()

    # 保存数据为xlsx文件
    df = pd.DataFrame(embedded_data, columns=['Component 1', 'Component 2'])
    df['Label'] = labels_array
    if save_root is not None:
        save_dir = os.path.join(save_root, "data_excel")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{title.replace(' ', '_')}.xlsx")
        df.to_excel(save_path, index=False)


def plot_tsne_combined(train_loader, test_loader, title, model=None, func_name=None, perplexity=5, save_root=None):
    # 将数据集和标签分开
    data_list = []
    labels_list = []
    loader_types_list = []  # 用于标记数据来自训练集还是测试集

    for loader, loader_type in zip([train_loader, test_loader], ["Train", "Test"]):
        for batch_idx, data in enumerate(loader):
            data_lq, data_x, data_cl = data
            if model is None:
                flattened_data = data_x.view(data_x.shape[0], -1).numpy()
            else:
                feature_data = model_func(model, data_x, func_name)
                flattened_data = feature_data.view(feature_data.shape[0], -1).detach().numpy()

            data_list.append(flattened_data)
            labels_list.append(np.argmax(data_cl.numpy(), axis=1))  # 将独热编码标签转换为类别标签
            loader_types_list.append([loader_type] * data_x.size(0))
    # 将数据转换为 numpy 数组
    data_array = np.concatenate(data_list, axis=0)
    labels_array = np.concatenate(labels_list, axis=0)
    loader_types_array = np.concatenate(loader_types_list, axis=0)

    # 使用 t-SNE 进行降维
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0)
    embedded_data = tsne.fit_transform(data_array)

    # 获取所有类别标签
    unique_labels = np.unique(labels_array)
    unique_loader_types = np.unique(loader_types_array)

    # 绘制降维后的数据，按照类别标签使用不同颜色绘制散点图，并标记数据来自训练集还是测试集
    plt.figure(figsize=(8, 6))
    for label in unique_labels:
        for data_type in unique_loader_types:
            marker = "o" if data_type == "Train" else "^"
            # color = "b" if label == 0 else "r"
            if label == 0:
                color = [40, 120, 181] if data_type == "Test" else [154, 201, 219]
            else:
                color = [200, 36, 35] if data_type == "Test" else [255, 136, 132]
            legend_name = f"{data_type} Health" if label == 0 else f"{data_type} Lungcancer"
            indices = np.where((labels_array == label) & (loader_types_array == data_type))[0]
            plt.scatter(embedded_data[indices, 0], embedded_data[indices, 1],
                        label=legend_name, marker=marker, c=[c / 255 for c in color])

    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    x_max_value, y_max_value = [np.ceil(x + 1) for x in np.max(embedded_data, axis=0)]
    x_min_value, y_min_value = [np.floor(x - 0.5) for x in np.min(embedded_data, axis=0)]
    # plt.title(title)
    plt.xlim((x_min_value, x_max_value))
    plt.ylim((y_min_value, y_max_value))
    plt.legend(loc='upper right')
    if save_root is not None:
        save_dir = os.path.join(save_root, "data_plot")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{title.replace(' ', '_').replace(',', '')}.png")
        plt.savefig(save_path, dpi=300, format="png")  # 保存图像为svg文件
    plt.show()

    # 保存数据为xlsx文件
    df = pd.DataFrame(embedded_data, columns=['Component 1', 'Component 2'])
    df['Label'] = labels_array
    df['Loader Type'] = loader_types_array
    if save_root is not None:
        save_dir = os.path.join(save_root, "data_excel")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{title.replace(' ', '_')}.xlsx")
        df.to_excel(save_path, index=False)


def model_func(model, input_x, func_name=None):
    # model.encoder.fcnplus.backbone
    if model.encoder.fcnplus.backbone.residual:
        res = input_x
    convblock1_x = model.encoder.fcnplus.backbone.convblock1(input_x)
    convblock2_x = model.encoder.fcnplus.backbone.convblock2(convblock1_x)
    convblock3_x = model.encoder.fcnplus.backbone.convblock3(convblock2_x)
    backbone_x = convblock3_x
    if model.encoder.fcnplus.backbone.residual:
        backbone_x = model.encoder.fcnplus.backbone.add(backbone_x, model.encoder.fcnplus.backbone.shortcut(res))

    ema1d_x = model.encoder.ema1d(backbone_x)
    head_x = model.encoder.fcnplus.head(ema1d_x)
    if func_name == "convblock1":
        return convblock1_x
    elif func_name == "convblock2":
        return convblock2_x
    elif func_name == "convblock3":
        return convblock3_x
    elif func_name == "backbone":
        return backbone_x
    elif func_name == "ema1d":
        return ema1d_x
    else:
        return head_x


if __name__ == "__main__":
    save_root = "./tsne-result/ema1dv3_combined_max_pp/"
    plot_combined = True
    train_labels = train_loader.dataset.tensors[2]
    train_class_labels = torch.argmax(train_labels, dim=1)
    train_class_counts = torch.bincount(train_class_labels)

    test_labels = test_loader.dataset.tensors[2]
    test_class_labels = torch.argmax(test_labels, dim=1)
    test_class_counts = torch.bincount(test_class_labels)

    train_perplexity = torch.max(train_class_counts).item()
    test_perplexity = torch.max(test_class_counts).item()
    # train_perplexity = test_perplexity = 5

    combined_class_counts = train_class_counts + test_class_counts
    print(train_class_counts, test_class_counts, combined_class_counts)
    combined_perplexity = torch.max(combined_class_counts).item()

    if plot_combined:
        plot_tsne_combined(train_loader, test_loader, f't-SNE on Combined Data',
                           perplexity=combined_perplexity, save_root=save_root)
        print("Plot T-SNE on Origin Combined Data")
    else:
        # 在训练数据上绘制 t-SNE 散点图
        plot_tsne(train_loader, 't-SNE on Train Data', perplexity=train_perplexity, save_root=save_root)
        print("Plot T-SNE on Origin Train Data")
        # 在测试数据上绘制 t-SNE 散点图
        plot_tsne(test_loader, 't-SNE on Test Data', perplexity=test_perplexity, save_root=save_root)
        print("Plot T-SNE on Origin Test Data")

    model_path = "./output/TS_CATMA/ts_catma_loop_1/ckpt_epoch_50.pth"
    args = argparse.Namespace(cfg='config_file.txt')
    args, config = parse_option(args)
    model = build_model(config)
    model.eval()
    load_model(model, model_path)
    func_name_list = ["convblock1", "convblock2", "convblock3", "backbone", "ema1d", "encoder"]
    for func_name in func_name_list:
        if plot_combined:
            plot_tsne_combined(train_loader, test_loader, f't-SNE on Combined Data, Layer {func_name}', model=model,
                               func_name=func_name,
                               perplexity=combined_perplexity, save_root=save_root)
        else:
            plot_tsne(train_loader, f't-SNE on Train Data, Layer {func_name}', model=model, func_name=func_name,
                      perplexity=train_perplexity, save_root=save_root)
            print(f"Plot T-SNE on Train Data, Layer {func_name}")
            plot_tsne(test_loader, f't-SNE on Test Data, Layer {func_name}', model=model, func_name=func_name,
                      perplexity=test_perplexity, save_root=save_root)
            print(f"Plot T-SNE on Test Data, Layer {func_name}")
