import argparse
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from data_loader import train_loader, test_loader
from main import parse_option
from model import build_model
from tool import load_model


def origin_ts2img(dataloader, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    sample_start_idx = 0  # 设置初始的 sample_start_idx

    for batch_idx, data in enumerate(dataloader):
        sample_start_idx = sample_start_idx + batch_idx * dataloader.batch_size
        data_lq, data_x, data_cl = data
        for idx in range(data_x.size(0)):
            sample_idx = sample_start_idx + idx + 1
            label = "health" if np.argmax(data_cl[idx]) == 0 else "lungcancer"
            feature_map = data_x[idx].squeeze().detach().cpu().numpy()
            # Normalize feature map
            normalized_map = cv2.normalize(feature_map, None, 0, 255, cv2.NORM_MINMAX)
            save_path = os.path.join(save_dir, f'sample{sample_idx}_{label}.png')
            # 将图像保存为黑白图像
            cv2.imwrite(save_path, normalized_map)
            print(f"Saving Feature Maps in {save_path}")


if __name__ == "__main__":
    model_path = "./output/TS_CATMA/ts_catma_loop_1/ckpt_epoch_50.pth"
    args = argparse.Namespace(cfg='config_file.txt')
    args, config = parse_option(args)
    model = build_model(config)
    model.eval()
    load_model(model, model_path)
    mode = "test"
    target_dataloader = train_loader if mode == "train" else test_loader
    save_root = f'./feature_maps/{mode}_origin'
    origin_ts2img(dataloader=target_dataloader, save_dir=save_root)
