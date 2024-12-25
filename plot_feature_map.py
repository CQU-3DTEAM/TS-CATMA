import argparse
import os

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from data_loader import train_loader, test_loader
from main import parse_option
from model import build_model
from tool import load_model


# Save Feature Maps From Hooks
def feature_map_hook(module, input, output, layer_name, save_root, threshold_ratio):
    global sample_start_idx, labels
    save_dir = os.path.join(save_root, layer_name)
    os.makedirs(save_dir, exist_ok=True)
    # 将特征图输出保存为图像文件
    for idx in range(output.size(0)):
        sample_idx = sample_start_idx + idx + 1
        label = "health" if np.argmax(labels[idx]) == 0 else "lungcancer"
        feature_map = output[idx].squeeze().detach().cpu().numpy()
        # Normalize feature map
        normalized_map = cv2.normalize(feature_map, None, 0, 255, cv2.NORM_MINMAX)
        # Flatten the normalized map
        flattened_map = normalized_map.flatten()
        # Sort the flattened map in descending order
        sorted_map = np.sort(flattened_map)[::-1]
        # Calculate the threshold value based on the top threshold_ratio of pixels
        threshold_value = sorted_map[int(len(sorted_map) * threshold_ratio)]
        # Apply thresholding
        _, thresholded_map = cv2.threshold(normalized_map, threshold_value, 255, cv2.THRESH_TOZERO)
        save_path = os.path.join(save_dir, f'sample{sample_idx}_{label}.png')
        # 将图像保存为黑白图像
        cv2.imwrite(save_path, thresholded_map)
        print(f"Saving Feature Maps in {save_path}")


# Register Hooks For Getting Feature Maps
def register_encoder_hooks(model, layer_names, save_path, threshold_ratio=0.01):
    # 注册hook函数到指定层
    hooks = []
    for name, module in model.encoder.named_modules():
        if name in layer_names:
            hook = module.register_forward_hook(
                lambda module, input, output, layer_name=name: feature_map_hook(module, input, output, layer_name,
                                                                                save_path, threshold_ratio))
            hooks.append(hook)
    return hooks


# Remove All Hooks
def remove_hooks(hooks):
    # 移除已注册的hook函数
    for hook in hooks:
        hook.remove()


# Get And Save Weight Maps
def execute_model_origin(model, data_x, data_cl, sample_start_idx, save_weight_dir=None, threshold_ratio=0.01):
    x = model.encoder.fcnplus.backbone(data_x)
    b, c, l = x.size()  # 注意这里的尺寸变化，l代表长度
    group_x = x.reshape(b * model.encoder.ema1d.groups, -1, l)  # b*g,c//g,l
    x_pooled = model.encoder.ema1d.pool(group_x)
    hw = model.encoder.ema1d.conv1x1(x_pooled)
    x1 = model.encoder.ema1d.gn(group_x * hw.sigmoid())
    x2 = model.encoder.ema1d.conv1x3(group_x)
    x11 = model.encoder.ema1d.softmax(
        model.encoder.ema1d.agp(x1).reshape(b * model.encoder.ema1d.groups, -1, 1).permute(0, 2, 1))
    x12 = x2.reshape(b * model.encoder.ema1d.groups, c // model.encoder.ema1d.groups, -1)  # b*g, c//g, l
    x21 = model.encoder.ema1d.softmax(
        model.encoder.ema1d.agp(x2).reshape(b * model.encoder.ema1d.groups, -1, 1).permute(0, 2, 1))
    x22 = x1.reshape(b * model.encoder.ema1d.groups, c // model.encoder.ema1d.groups, -1)  # b*g, c//g, l
    weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * model.encoder.ema1d.groups, 1, l)
    group_x_ones = torch.ones(group_x.shape)
    ema1d_out = (group_x * weights.sigmoid()).reshape(b, c, l)
    weights = (group_x_ones * weights).reshape(b, c, l)
    head_out = model.encoder.fcnplus.head(ema1d_out)
    if save_weight_dir is not None:
        os.makedirs(save_weight_dir, exist_ok=True)
        # 将特征图输出保存为图像文件
        for idx in range(weights.size(0)):
            sample_idx = sample_start_idx + idx + 1
            label = "health" if np.argmax(data_cl[idx]) == 0 else "lungcancer"
            feature_map = weights[idx].squeeze().detach().cpu().numpy()
            save_img(feature_map, save_weight_dir, sample_idx, label, threshold_ratio)


# Get And Save Weight Maps
def execute_model(model, data_x, data_cl, sample_start_idx, save_weight_dir=None, threshold_ratio=0.01):
    x = model.encoder.fcnplus.backbone(data_x)

    b, c, l = x.size()  # 注意这里的尺寸变化，l代表长度
    group_x = x.reshape(b * model.encoder.ema1d.groups, -1, l)  # b*g,c//g,l
    x1 = model.encoder.ema1d.conv1x1(group_x)
    x2 = model.encoder.ema1d.conv1x3(group_x)
    x3 = model.encoder.ema1d.conv1x5(group_x)

    x11 = model.encoder.ema1d.softmax(
        model.encoder.ema1d.agp(x1).reshape(b * model.encoder.ema1d.groups, -1, 1).permute(0, 2, 1))
    x12 = x1.reshape(b * model.encoder.ema1d.groups, c // model.encoder.ema1d.groups, -1)  # b*g, c//g, l

    x21 = model.encoder.ema1d.softmax(
        model.encoder.ema1d.agp(x2).reshape(b * model.encoder.ema1d.groups, -1, 1).permute(0, 2, 1))
    x22 = x2.reshape(b * model.encoder.ema1d.groups, c // model.encoder.ema1d.groups, -1)  # b*g, c//g, l

    x31 = model.encoder.ema1d.softmax(
        model.encoder.ema1d.agp(x3).reshape(b * model.encoder.ema1d.groups, -1, 1).permute(0, 2, 1))
    x32 = x3.reshape(b * model.encoder.ema1d.groups, c // model.encoder.ema1d.groups, -1)  # b*g, c//g, l

    weight1 = x12
    weight3 = x22
    weight5 = x32
    weight13 = (torch.matmul(x11, x22) + torch.matmul(x21, x12)).reshape(b * model.encoder.ema1d.groups, 1, l).reshape(
        b * model.encoder.ema1d.groups, 1, l)
    weight15 = (torch.matmul(x11, x32) + torch.matmul(x31, x12)).reshape(b * model.encoder.ema1d.groups, 1, l).reshape(
        b * model.encoder.ema1d.groups, 1, l)
    weights = (
            torch.matmul(x11, x22) + torch.matmul(x21, x12)
            + torch.matmul(x11, x32) + torch.matmul(x31, x12)
        # + torch.matmul(x21, x32) + torch.matmul(x31, x22)
    ).reshape(b * model.encoder.ema1d.groups, 1, l)
    group_x_ones = torch.ones(group_x.shape)
    ema1d_out = (group_x * weights.sigmoid()).reshape(b, c, l)
    weight1 = (group_x * weight1.sigmoid()).reshape(b, c, l)
    weight3 = (group_x * weight3.sigmoid()).reshape(b, c, l)
    weight5 = (group_x * weight5.sigmoid()).reshape(b, c, l)
    weight13 = (group_x * weight13.sigmoid()).reshape(b, c, l)
    weight15 = (group_x * weight15.sigmoid()).reshape(b, c, l)
    weights = (group_x_ones * weights).reshape(b, c, l)

    head_out = model.encoder.fcnplus.head(ema1d_out)
    if save_weight_dir is not None:
        os.makedirs(save_weight_dir, exist_ok=True)
        # 将特征图输出保存为图像文件
        for idx in range(weights.size(0)):
            sample_idx = sample_start_idx + idx + 1
            label = "health" if np.argmax(data_cl[idx]) == 0 else "lungcancer"
            feature_map1 = weight1[idx].squeeze().detach().cpu().numpy()
            feature_map3 = weight3[idx].squeeze().detach().cpu().numpy()
            feature_map5 = weight5[idx].squeeze().detach().cpu().numpy()
            feature_map13 = weight13[idx].squeeze().detach().cpu().numpy()
            feature_map15 = weight15[idx].squeeze().detach().cpu().numpy()
            feature_map = weights[idx].squeeze().detach().cpu().numpy()
            save_img(feature_map1, save_weight_dir + "_w1", sample_idx, label, threshold_ratio)
            save_img(feature_map3, save_weight_dir + "_w3", sample_idx, label, threshold_ratio)
            save_img(feature_map5, save_weight_dir + "w5", sample_idx, label, threshold_ratio)
            save_img(feature_map13, save_weight_dir + "w13", sample_idx, label, threshold_ratio)
            save_img(feature_map15, save_weight_dir + "w15", sample_idx, label, threshold_ratio)
            save_img(feature_map, save_weight_dir + "wall", sample_idx, label, threshold_ratio)


def save_img(feature_map, save_weight_dir, sample_idx, label, threshold_ratio):
    os.makedirs(save_weight_dir, exist_ok=True)
    # Normalize feature map
    normalized_map = cv2.normalize(feature_map, None, 0, 255, cv2.NORM_MINMAX)
    # Flatten the normalized map
    flattened_map = normalized_map.flatten()
    # Sort the flattened map in descending order
    sorted_map = np.sort(flattened_map)[::-1]
    # Calculate the threshold value based on the top threshold_ratio of pixels
    threshold_value = sorted_map[int(len(sorted_map) * threshold_ratio)]
    # Apply thresholding
    _, thresholded_map = cv2.threshold(normalized_map, threshold_value, 255, cv2.THRESH_TOZERO)
    save_path = os.path.join(save_weight_dir, f'sample{sample_idx}_{label}.png')
    # 将图像保存为黑白图像
    cv2.imwrite(save_path, thresholded_map)
    print(f"Saving Feature Maps in {save_path}")


if __name__ == "__main__":
    model_path = "./output/TS_CATMA/ts_catma_loop_1/ckpt_epoch_50.pth"
    args = argparse.Namespace(cfg='config_file.txt')
    args, config = parse_option(args)
    model = build_model(config)
    print(model)
    model.eval()
    load_model(model, model_path)

    encoder_layer_names = [
        "fcnplus.backbone.convblock1",
        "fcnplus.backbone.convblock2",
        "fcnplus.backbone.convblock3",
        "ema1d",
    ]
    mode = "test"
    target_dataloader = train_loader if mode == "train" else test_loader
    save_root = f'./output/TS_CATMA/feature_maps/{mode}_threshold'
    save_weight_dir = f'./output/TS_CATMA/feature_maps/{mode}_weights_threshold'
    sample_start_idx = 0  # 设置初始的 sample_start_idx
    hooks = register_encoder_hooks(model, encoder_layer_names, save_root, threshold_ratio=0.01)

    for batch_idx, data in enumerate(target_dataloader):
        sample_start_idx = sample_start_idx + batch_idx * target_dataloader.batch_size
        data_lq, data_x, data_cl = data
        labels = data_cl
        if model.encoder.name == "FCNPlusEMAEncoder":
            execute_model(model, data_x, data_cl, sample_start_idx, save_weight_dir, threshold_ratio=0.01)

    remove_hooks(hooks)
