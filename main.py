import argparse
import os
import random
import time

import numpy as np
from torch.utils.data import DataLoader
from torchsummary import summary
from thop import profile, clever_format
from tsai.models.FCN import FCN
from tsai.models.InceptionTime import InceptionTime
from tsai.models.RNN_FCN import LSTM_FCN
from tsai.models.ResNet import ResNet

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import logging
import sys
import torch
from tensorboardX import SummaryWriter
from termcolor import colored
from torch import optim
from sklearn.metrics import roc_curve, auc, recall_score

from config import get_config
from data_loader import train_loader, val_loader, test_loader
from model import build_model, loss_function, build_toy_model
from tool import save_checkpoint, load_checkpoint, auto_resume_helper
from lr_scheduler import build_scheduler


def parse_option(args):
    parser = argparse.ArgumentParser('RDCT pre-training script', add_help=False)
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file',
                        default="./configs/TS_CATMA.yaml")
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')

    args = parser.parse_args()

    config = get_config(args)

    return args, config


def model_summary(config, log_writer, model_name="ts-cama"):
    if model_name == "fcn":
        model = FCN(c_in=22, c_out=2)
    elif model_name == "resnet":
        model = ResNet(c_in=22, c_out=2)
    elif model_name == "inception_time":
        model = InceptionTime(c_in=22, c_out=2)
    elif model_name == "lstm_fcn":
        model = LSTM_FCN(c_in=22, c_out=2, shuffle=False)
    elif model_name == "ts-cama":
        model = build_toy_model(config)
    else:
        print("No such a model!")
        return None
    print(f"Current Model {model_name.upper()}")
    model.cuda()
    # summary(model.encoder, input_size=(22, 360))
    log_writer.add_graph(model.encoder, torch.ones(64, 22, 360).cuda())
    summary(model, (22, 360))
    flops, params = profile(model, inputs=(torch.ones(1, 22, 360).cuda(), ))
    flops, params = clever_format([flops, params], '%.3f')
    print(f"FLOPs: {flops}")
    print(f"Params: {params}")
    return flops, params


def main(config, log_writer):
    logger.info(f"Creating model:{config.MODEL.NAME}")
    model = build_model(config)
    optimizers = {
        "encoder": optim.AdamW([{'params': model.encoder.parameters()}]),
        "label_classifier": optim.AdamW([{'params': model.label_classifier.parameters()}]),
        "domain_classifier": optim.AdamW([{'params': model.domain_classifier.parameters()}]),
        # "decoder": optim.AdamW([{'params': model.decoder.parameters()}])
    }
    model.cuda()
    logger.info(str(model))
    # lr_schedulers = None
    lr_schedulers = {key: build_scheduler(config, optimizers[key], len(train_loader)) for key in optimizers.keys()}
    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT, logger)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')
    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model, optimizers, lr_schedulers, logger)
    else:
        max_accuracy = 0
    src_cl_acc = val_cl_acc = tgt_cl_acc = total_dl_acc = auc_value = sensitivity = specificity = 0
    logger.info(f"Start Training")
    train_time, test_time = 0, 0
    thresholds = config.TRAIN.PSEUDO_THRESHOLDS
    no_pseudo_epoch = config.TRAIN.NO_PSEUDO_EPOCH
    combined_dataloader = None
    for epoch in range(config.TRAIN.START_EPOCH + 1, config.TRAIN.EPOCHS + 1):
        train_start_time = torch.cuda.Event(enable_timing=True)
        train_end_time = torch.cuda.Event(enable_timing=True)
        train_start_time.record()
        if epoch > no_pseudo_epoch and (epoch % config.TRAIN.PSEUDO_FREQ == 1 or combined_dataloader is None):
            combined_dataloader = update_pseudo_label(config, model, train_loader, val_loader, test_loader, thresholds,
                                                      log_writer)
        if epoch <= no_pseudo_epoch:
            loss = train_one_epoch(config, model, train_loader, test_loader, optimizers, epoch, lr_schedulers,
                                   log_writer,
                                   mode="adversarial")
        else:
            loss = train_pseudo_one_epoch(config, model, train_loader, test_loader, combined_dataloader, optimizers,
                                          epoch, lr_schedulers, log_writer,
                                          mode="adversarial")
        train_end_time.record()
        torch.cuda.synchronize()
        train_time += train_start_time.elapsed_time(train_end_time)

        if epoch % config.SAVE_FREQ == 0 or epoch == config.TRAIN.EPOCHS:
            save_checkpoint(config, epoch, model, max_accuracy, optimizers, lr_schedulers, logger)
        log_writer.flush()
        log_writer.add_scalar("loss/loss_epoch", loss, epoch)
        if epoch % config.TEST_FREQ == 0:
            src_cl_acc, val_cl_acc, tgt_cl_acc, total_dl_acc, auc_value, sensitivity, specificity = test_model(config,
                                                                                                               model,
                                                                                                               train_loader,
                                                                                                               val_loader,
                                                                                                               test_loader,
                                                                                                               epoch,
                                                                                                               log_writer)

            max_accuracy = max(tgt_cl_acc, max_accuracy)
            logger.info(
                "Epoch [{}] max_accuracy: {}%, src_cl_acc: {}%, val_cl_acc {}%, tgt_cl_acc:{}%, total_dl_acc:{}%, auc:{}, "
                "sensitivity:{}, specificity:{}".
                    format(epoch, max_accuracy, src_cl_acc, val_cl_acc, tgt_cl_acc, total_dl_acc, auc_value,
                           sensitivity, specificity))
        if epoch == config.TRAIN.EPOCHS:
            test_start_time = torch.cuda.Event(enable_timing=True)
            test_end_time = torch.cuda.Event(enable_timing=True)
            test_start_time.record()
            test_model(config, model, train_loader, val_loader, test_loader, epoch, log_writer)
            test_end_time.record()
            torch.cuda.synchronize()
            test_time = test_start_time.elapsed_time(test_end_time)
        if epoch % config.PRINT_FREQ == 0:
            pass
    logger.info(f"Train Time:{train_time}, Test Time:{test_time}")
    return max_accuracy, src_cl_acc, val_cl_acc, tgt_cl_acc, total_dl_acc, auc_value, sensitivity, specificity, train_time, test_time


def train_one_epoch(config, model, source_dataloader, target_dataloader, optimizers, epoch, lr_schedulers, log_writer,
                    mode="all"):
    model.train()
    # Set the experimental step
    start_steps = epoch * len(source_dataloader)  # start step
    total_steps = config.TRAIN.EPOCHS * len(source_dataloader)  # total steps
    loss = None
    if mode == "adversarial":
        for batch_idx, (sdata, tdata) in enumerate(zip(source_dataloader, target_dataloader)):
            iteration_steps = start_steps + batch_idx
            p = float(batch_idx + start_steps) / total_steps  # a variable for adjusting learning rate
            constant = 2. / (1 + np.exp(-config.MODEL.DOMAIN_CLS.gamma * p)) - 1  # a constant of RevGrad

            # Get data for the source and target domains
            src_lq, src_x, src_cl = sdata
            tgt_lq, tgt_x, tgt_cl = tdata
            src_lq, src_x, src_cl = src_lq.cuda(), src_x.cuda(), src_cl.cuda()
            tgt_lq, tgt_x, tgt_cl = tgt_lq.cuda(), tgt_x.cuda(), tgt_cl.cuda()

            # stage1
            optimizers["domain_classifier"].zero_grad()
            # optimizers["decoder"].zero_grad()
            _, domain_loss_stage1, _ = model(src_lq, src_x, src_cl, tgt_lq, tgt_x, constant, mask=None,
                                             train_mode="domain+recon")
            loss_stage1 = loss_function(epoch, None, domain_loss_stage1, None, mode="domain+recon")
            loss_stage1.backward()
            torch.nn.utils.clip_grad_norm_(model.domain_classifier.parameters(), max_norm=config.TRAIN.CLIP_GRAD)
            # torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm=config.TRAIN.CLIP_GRAD)
            optimizers["domain_classifier"].step()
            # optimizers["decoder"].step()

            # stage2
            optimizers["encoder"].zero_grad()
            optimizers["label_classifier"].zero_grad()
            label_loss, domain_loss_stage2, _ = model(src_lq, src_x, src_cl, tgt_lq, tgt_x, constant, mask=None,
                                                      train_mode="encoder+label")
            loss_stage2 = loss_function(epoch, label_loss, domain_loss_stage2, None, mode="encoder+label")
            loss_stage2.backward()
            loss = loss_stage1 + loss_stage2
            torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), max_norm=config.TRAIN.CLIP_GRAD)
            torch.nn.utils.clip_grad_norm_(model.label_classifier.parameters(), max_norm=config.TRAIN.CLIP_GRAD)
            optimizers["encoder"].step()
            optimizers["label_classifier"].step()

            # statistics
            log_writer.add_scalar("loss/label_loss", label_loss, iteration_steps)
            log_writer.add_scalar("loss/domain_loss_stage1", domain_loss_stage1, iteration_steps)
            log_writer.add_scalar("loss/domain_loss_stage2", domain_loss_stage2, iteration_steps)
            # log_writer.add_scalar("loss/recon_loss", recon_loss, iteration_steps)
            log_writer.add_scalar("loss/total_loss_stage1", loss_stage1, iteration_steps)
            log_writer.add_scalar("loss/total_loss_stage2", loss_stage2, iteration_steps)
            lr_info = f"Iteration [{iteration_steps}]"
            for key in optimizers.keys():
                lr = optimizers[key].param_groups[0]["lr"]
                log_writer.add_scalar(f"lr/{key}", lr, iteration_steps)
                lr_info = lr_info + f" lr/{key}: {lr}"
            if epoch % config.PRINT_FREQ == 0:
                logger.info("Iteration [{}] domain_loss_stage1: {}, loss_stage1:{}, "
                            "domain_loss_stage2:{}, label_loss: {}, loss_stage2:{}".
                            format(iteration_steps, domain_loss_stage1, loss_stage1,
                                   domain_loss_stage2, label_loss, loss_stage2))
                logger.info(lr_info)
            # for key in optimizers.keys():
            #     optimizers[key].step()
            for key in lr_schedulers.keys():
                lr_schedulers[key].step_update(iteration_steps)
    elif mode == "label_only":
        for batch_idx, (sdata, tdata) in enumerate(zip(source_dataloader, target_dataloader)):
            iteration_steps = start_steps + batch_idx
            optimizers["encoder"].zero_grad()
            optimizers["label_classifier"].zero_grad()
            p = float(batch_idx + start_steps) / total_steps  # a variable for adjusting learning rate
            constant = 2. / (1 + np.exp(-config.MODEL.DOMAIN_CLS.gamma * p)) - 1  # a constant of RevGrad

            # Get data for the source and target domains
            src_lq, src_x, src_cl = sdata
            src_lq, src_x, src_cl = src_lq.cuda(), src_x.cuda(), src_cl.cuda()

            label_loss, domain_loss, recon_loss = model(src_lq, src_x, src_cl, None, None, constant, mask=None,
                                                        train_mode="label_only")

            log_writer.add_scalar("loss/label_loss", label_loss, iteration_steps)
            loss = loss_function(epoch, label_loss, domain_loss, recon_loss, mode="label_only")
            log_writer.add_scalar("loss/total_loss", loss, iteration_steps)
            for key in optimizers.keys():
                lr = optimizers[key].param_groups[0]["lr"]
                log_writer.add_scalar(f"lr/{key}", lr, iteration_steps)
            if epoch % config.PRINT_FREQ == 0:
                logger.info("Iteration [{}] Total Loss: {}, Label Loss:{}, lr: {}".
                            format(iteration_steps, loss, label_loss, lr))
            loss.backward()  # the standard PyTorch training mode
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.TRAIN.CLIP_GRAD)
            optimizers["encoder"].step()
            optimizers["label_classifier"].step()
            for key in lr_schedulers.keys():
                lr_schedulers[key].step_update(iteration_steps)
    else:
        for batch_idx, (sdata, tdata) in enumerate(zip(source_dataloader, target_dataloader)):
            iteration_steps = start_steps + batch_idx
            optimizers.zero_grad()
            p = float(batch_idx + start_steps) / total_steps  # a variable for adjusting learning rate
            constant = 2. / (1 + np.exp(-config.MODEL.DOMAIN_CLS.gamma * p)) - 1  # a constant of RevGrad

            # Get data for the source and target domains
            src_lq, src_x, src_cl = sdata
            tgt_lq, tgt_x, tgt_cl = tdata
            src_lq, src_x, src_cl = src_lq.cuda(), src_x.cuda(), src_cl.cuda()
            tgt_lq, tgt_x, tgt_cl = tgt_lq.cuda(), tgt_x.cuda(), tgt_cl.cuda()

            label_loss, domain_loss, recon_loss = model(src_lq, src_x, src_cl, tgt_lq, tgt_x, constant, mask=None,
                                                        train_mode="all")

            log_writer.add_scalar("loss/label_loss", label_loss, iteration_steps)
            log_writer.add_scalar("loss/domain_loss", domain_loss, iteration_steps)
            log_writer.add_scalar("loss/recon_loss", recon_loss, iteration_steps)
            loss = loss_function(epoch, label_loss, domain_loss, recon_loss)
            log_writer.add_scalar("loss/total_loss", loss, iteration_steps)
            for key in optimizers.keys():
                lr = optimizers[key].param_groups[0]["lr"]
                log_writer.add_scalar(f"lr/{key}", lr, iteration_steps)
            if epoch % config.PRINT_FREQ == 0:
                logger.info("Iteration [{}] Total Loss: {}, Label Loss:{}, Domain Loss:{}, Recon Loss:{}, lr: {}".
                            format(iteration_steps, loss, label_loss, domain_loss, recon_loss, lr))
            loss.backward()  # the standard PyTorch training mode
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.TRAIN.CLIP_GRAD)
            for key in optimizers.keys():
                optimizers[key].step()
            for key in lr_schedulers.keys():
                lr_schedulers[key].step_update(iteration_steps)
    return loss


def train_pseudo_one_epoch(config, model, source_dataloader, target_dataloader, combined_dataloader, optimizers, epoch,
                           lr_schedulers,
                           log_writer, mode="all"):
    model.train()
    # Set the experimental step
    start_steps = epoch * len(source_dataloader)  # start step
    total_steps = config.TRAIN.EPOCHS * len(source_dataloader)  # total steps
    loss = None
    if mode == "adversarial":
        for batch_idx, (sdata, tdata, cdata) in enumerate(
                zip(source_dataloader, target_dataloader, combined_dataloader)):
            iteration_steps = start_steps + batch_idx
            p = float(batch_idx + start_steps) / total_steps  # a variable for adjusting learning rate
            constant = 2. / (1 + np.exp(-config.MODEL.DOMAIN_CLS.gamma * p)) - 1  # a constant of RevGrad

            # Get data for the source and target domains
            src_lq, src_x, src_cl = sdata
            tgt_lq, tgt_x, tgt_cl = tdata
            com_lq, com_x, com_cl = cdata
            src_lq, src_x, src_cl = src_lq.cuda(), src_x.cuda(), src_cl.cuda()
            tgt_lq, tgt_x, tgt_cl = tgt_lq.cuda(), tgt_x.cuda(), tgt_cl.cuda()
            com_lq, com_x, com_cl = com_lq.cuda(), com_x.cuda(), com_cl.cuda()

            # stage1
            optimizers["domain_classifier"].zero_grad()
            # optimizers["decoder"].zero_grad()
            _, domain_loss_stage1, _ = model(src_lq, src_x, src_cl, tgt_lq, tgt_x, constant, mask=None,
                                             train_mode="domain+recon")
            loss_stage1 = loss_function(epoch, None, domain_loss_stage1, None, mode="domain+recon")
            loss_stage1.backward()
            torch.nn.utils.clip_grad_norm_(model.domain_classifier.parameters(), max_norm=config.TRAIN.CLIP_GRAD)
            # torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm=config.TRAIN.CLIP_GRAD)
            optimizers["domain_classifier"].step()
            # optimizers["decoder"].step()

            # stage2
            optimizers["encoder"].zero_grad()
            optimizers["label_classifier"].zero_grad()
            label_loss, domain_loss_stage2, _ = model(com_x, src_x, com_cl, tgt_lq, tgt_x, constant, mask=None,
                                                      train_mode="encoder+label_pseudo")
            loss_stage2 = loss_function(epoch, label_loss, domain_loss_stage2, None, mode="encoder+pseudolabel")
            loss_stage2.backward()
            loss = loss_stage1 + loss_stage2
            torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), max_norm=config.TRAIN.CLIP_GRAD)
            torch.nn.utils.clip_grad_norm_(model.label_classifier.parameters(), max_norm=config.TRAIN.CLIP_GRAD)
            optimizers["encoder"].step()
            optimizers["label_classifier"].step()

            # statistics
            log_writer.add_scalar("loss/label_loss", label_loss, iteration_steps)
            log_writer.add_scalar("loss/domain_loss_stage1", domain_loss_stage1, iteration_steps)
            log_writer.add_scalar("loss/domain_loss_stage2", domain_loss_stage2, iteration_steps)
            # log_writer.add_scalar("loss/recon_loss", recon_loss, iteration_steps)
            log_writer.add_scalar("loss/total_loss_stage1", loss_stage1, iteration_steps)
            log_writer.add_scalar("loss/total_loss_stage2", loss_stage2, iteration_steps)
            lr_info = f"Iteration [{iteration_steps}]"
            for key in optimizers.keys():
                lr = optimizers[key].param_groups[0]["lr"]
                log_writer.add_scalar(f"lr/{key}", lr, iteration_steps)
                lr_info = lr_info + f" lr/{key}: {lr}"
            if epoch % config.PRINT_FREQ == 0:
                logger.info("Iteration [{}] domain_loss_stage1: {}, loss_stage1:{}, "
                            "domain_loss_stage2:{}, label_loss: {}, loss_stage2:{}".
                            format(iteration_steps, domain_loss_stage1, loss_stage1,
                                   domain_loss_stage2, label_loss, loss_stage2))
                logger.info(lr_info)
            # for key in optimizers.keys():
            #     optimizers[key].step()
            for key in lr_schedulers.keys():
                lr_schedulers[key].step_update(iteration_steps)
    elif mode == "label_only":
        for batch_idx, (sdata, tdata, cdata) in enumerate(
                zip(source_dataloader, target_dataloader, combined_dataloader)):
            iteration_steps = start_steps + batch_idx
            p = float(batch_idx + start_steps) / total_steps  # a variable for adjusting learning rate
            constant = 2. / (1 + np.exp(-config.MODEL.DOMAIN_CLS.gamma * p)) - 1  # a constant of RevGrad

            # Get data for the source and target domains
            com_lq, com_x, com_cl = cdata
            com_lq, com_x, com_cl = com_lq.cuda(), com_x.cuda(), com_cl.cuda()

            label_loss, domain_loss, recon_loss = model(com_lq, com_x, com_cl, None, None, constant, mask=None,
                                                        train_mode="label_only")
            log_writer.add_scalar("loss/label_loss", label_loss, iteration_steps)
            loss = loss_function(epoch, label_loss, domain_loss, recon_loss, mode="label_only")
            log_writer.add_scalar("loss/total_loss", loss, iteration_steps)
            for key in optimizers.keys():
                lr = optimizers[key].param_groups[0]["lr"]
                log_writer.add_scalar(f"lr/{key}", lr, iteration_steps)
            if epoch % config.PRINT_FREQ == 0:
                logger.info("Iteration [{}] Total Loss: {}, Label Loss:{}, lr: {}".
                            format(iteration_steps, loss, label_loss, lr))
            loss.backward()  # the standard PyTorch training mode
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.TRAIN.CLIP_GRAD)
            optimizers["encoder"].step()
            optimizers["label_classifier"].step()
            for key in lr_schedulers.keys():
                lr_schedulers[key].step_update(iteration_steps)
    else:
        for batch_idx, (sdata, tdata) in enumerate(zip(source_dataloader, target_dataloader)):
            iteration_steps = start_steps + batch_idx
            optimizers.zero_grad()
            p = float(batch_idx + start_steps) / total_steps  # a variable for adjusting learning rate
            constant = 2. / (1 + np.exp(-config.MODEL.DOMAIN_CLS.gamma * p)) - 1  # a constant of RevGrad

            # Get data for the source and target domains
            src_lq, src_x, src_cl = sdata
            tgt_lq, tgt_x, tgt_cl = tdata
            src_lq, src_x, src_cl = src_lq.cuda(), src_x.cuda(), src_cl.cuda()
            tgt_lq, tgt_x, tgt_cl = tgt_lq.cuda(), tgt_x.cuda(), tgt_cl.cuda()

            label_loss, domain_loss, recon_loss = model(src_lq, src_x, src_cl, tgt_lq, tgt_x, constant, mask=None,
                                                        train_mode="all")

            log_writer.add_scalar("loss/label_loss", label_loss, iteration_steps)
            log_writer.add_scalar("loss/domain_loss", domain_loss, iteration_steps)
            log_writer.add_scalar("loss/recon_loss", recon_loss, iteration_steps)
            loss = loss_function(epoch, label_loss, domain_loss, recon_loss)
            log_writer.add_scalar("loss/total_loss", loss, iteration_steps)
            for key in optimizers.keys():
                lr = optimizers[key].param_groups[0]["lr"]
                log_writer.add_scalar(f"lr/{key}", lr, iteration_steps)
            if epoch % config.PRINT_FREQ == 0:
                logger.info("Iteration [{}] Total Loss: {}, Label Loss:{}, Domain Loss:{}, Recon Loss:{}, lr: {}".
                            format(iteration_steps, loss, label_loss, domain_loss, recon_loss, lr))
            loss.backward()  # the standard PyTorch training mode
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.TRAIN.CLIP_GRAD)
            for key in optimizers.keys():
                optimizers[key].step()
            for key in lr_schedulers.keys():
                lr_schedulers[key].step_update(iteration_steps)
    return loss


def update_pseudo_label(config, model, source_dataloader, val_dataloader, target_dataloader, thresholds, log_writer):
    # 生成目标域伪标签和置信度
    model.eval()
    # val_predicted_labels = []
    # val_labels = []
    # val_confidences = []
    # with torch.no_grad():
    #     for batch_idx, tdata in enumerate(val_dataloader):
    #         tgt_lq, tgt_x, tgt_cl = tdata
    #         tgt_lq, tgt_x, tgt_cl = tgt_lq.cuda(), tgt_x.cuda(), tgt_cl.cuda()
    #
    #         tgt_dl = torch.zeros(tgt_x.size(0), 2)  # 维度为（input2.size()[0], 2）的一张量
    #         tgt_dl[:, 0], tgt_dl[:, 1] = 0, 1
    #
    #         if model.encoder.name == "TSTransformerEncoder":
    #             tgt_x = tgt_x.permute(0, 2, 1)  # (batch_size, seq_length, feat_dim)
    #         tgt_feature = model.encoder_stage(tgt_x)
    #
    #         tgt_pred_cl, tgt_cls_logits = model.label_classifier(tgt_feature)
    #
    #         val_confidence, val_predicted = torch.max(tgt_pred_cl, 1)
    #         val_predicted_labels.append(val_predicted)
    #         val_confidences.append(val_confidence)
    #         true_confidence, true_predicted = torch.max(tgt_cl, 1)
    #         val_labels.append(true_predicted)
    # val_predicted_labels = torch.cat(val_predicted_labels)
    # val_confidences = torch.cat(val_confidences)
    # val_labels = torch.cat(val_labels)
    # print("val:", val_predicted_labels, val_confidences, val_labels)
    # class_1_min_conf, class_2_min_conf = 0.5, 0.5
    # for true_label, pred_label, conf in zip(val_labels, val_predicted, val_confidences):
    #     if true_label != pred_label:
    #         if true_label == 0:
    #             class_1_min_conf = max(class_1_min_conf, conf)
    #         else:
    #             class_2_min_conf = max(class_2_min_conf, conf)
    # thresholds = [class_1_min_conf + 0.0001, class_2_min_conf + 0.0001]
    # print("thresholds:", thresholds)

    predicted_labels = []
    pseudo_labels = []
    confidences = []
    with torch.no_grad():
        for batch_idx, tdata in enumerate(target_dataloader):
            tgt_lq, tgt_x, tgt_cl = tdata
            tgt_lq, tgt_x, tgt_cl = tgt_lq.cuda(), tgt_x.cuda(), tgt_cl.cuda()

            tgt_dl = torch.zeros(tgt_x.size(0), 2)  # 维度为（input2.size()[0], 2）的一张量
            tgt_dl[:, 0], tgt_dl[:, 1] = 0, 1

            if model.encoder.name == "TSTransformerEncoder":
                tgt_x = tgt_x.permute(0, 2, 1)  # (batch_size, seq_length, feat_dim)
            tgt_feature = model.encoder_stage(tgt_x)

            tgt_pred_cl, tgt_cls_logits = model.label_classifier(tgt_feature)

            confidence, predicted = torch.max(tgt_pred_cl, 1)
            predicted_labels.append(predicted)
            eye_tensor = torch.eye(2)
            pseudo_labels.append(eye_tensor[predicted])
            confidences.append(confidence)
    pseudo_labels = torch.cat(pseudo_labels)
    # pseudo_labels = torch.abs(pseudo_labels - 0.2)
    predicted_labels = torch.cat(predicted_labels).cuda()
    confidences = torch.cat(confidences).cuda()

    # 将 thresholds 转换为 torch.Tensor
    thresholds = torch.tensor(thresholds).cuda()
    # 根据类别阈值选择样本
    selected_mask = confidences >= thresholds[predicted_labels]
    selected_indices = torch.nonzero(selected_mask, as_tuple=True)[0]
    print("selected_indices:", selected_indices)

    # n_percent = 1  # 选择10%的样本
    # num_samples = len(selected_indices)
    # num_selected = int(num_samples * n_percent)
    # selected_indices = selected_indices[torch.randperm(num_samples)[:num_selected]]

    print("tgt_label:", target_dataloader.dataset.tensors[2][selected_indices])
    print("pseudo_label:", pseudo_labels[selected_indices])
    print("confidence:", confidences[selected_indices])
    print("selected_indices_random:", selected_indices)
    # 创建新的源域数据集和目标域伪标签数据集
    new_target_dataset = torch.utils.data.TensorDataset(target_dataloader.dataset.tensors[0],
                                                        target_dataloader.dataset.tensors[1],
                                                        pseudo_labels)
    selected_target_dataset = torch.utils.data.Subset(new_target_dataset, selected_indices)

    source_subset = torch.utils.data.Subset(source_dataloader.dataset, range(len(source_dataloader.dataset)))

    # 创建新的训练集,包含打乱后的源域样本和选择的目标域伪标签样本
    combined_dataset = torch.utils.data.ConcatDataset([selected_target_dataset, source_subset])
    batch_size = 85
    combined_dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
    return combined_dataloader


def test_model(config, model, source_dataloader, val_dataloader, target_dataloader, epoch, log_writer):
    model.eval()
    # Save the intermediate variable
    src_cl_correct = 0.0  # source
    val_cl_correct = 0.0  # val
    tgt_cl_correct = 0.0  # target
    total_dl_correct = 0.0  # domain
    tgt_dl_correct = 0.0  # target
    src_dl_correct = 0.0  # source
    tgt_cl_tensor_list = []
    tgt_pred_cl_tensor_list = []
    with torch.no_grad():
        for batch_idx, sdata in enumerate(source_dataloader):
            p = float(batch_idx) / len(source_dataloader)
            constant = 2. / (1. + np.exp(-10 * p)) - 1

            src_lq, src_x, src_cl = sdata  # obtain the source data and labels
            src_lq, src_x, src_cl = src_lq.cuda(), src_x.cuda(), src_cl.cuda()

            src_dl = torch.zeros(src_x.size(0), 2)  # 维度为（input1.size()[0], 2）的零张量
            src_dl[:, 0], src_dl[:, 1] = 1, 0

            if model.encoder.name == "TSTransformerEncoder":
                src_x = src_x.permute(0, 2, 1)  # (batch_size, seq_length, feat_dim)
            src_feature = model.encoder_stage(src_x)

            src_pred_cl, _ = model.label_classifier(src_feature)

            src_pred_cl = torch.argmax(src_pred_cl, dim=1)
            src_cl = torch.argmax(src_cl, dim=1)
            src_cl_correct += torch.sum(src_pred_cl == src_cl).item()

            src_pred_dl, _ = model.domain_classifier(src_feature, constant)  ##
            src_pred_dl = torch.argmax(src_pred_dl, dim=1).cuda()
            src_dl = torch.argmax(src_dl, dim=1).cuda()
            src_dl_correct += torch.sum(src_pred_dl == src_dl).item()
        for batch_idx, vdata in enumerate(val_dataloader):
            p = float(batch_idx) / len(source_dataloader)
            constant = 2. / (1. + np.exp(-10 * p)) - 1

            src_lq, src_x, src_cl = vdata  # obtain the source data and labels
            src_lq, src_x, src_cl = src_lq.cuda(), src_x.cuda(), src_cl.cuda()

            src_dl = torch.zeros(src_x.size(0), 2)  # 维度为（input1.size()[0], 2）的零张量
            src_dl[:, 0], src_dl[:, 1] = 1, 0

            if model.encoder.name == "TSTransformerEncoder":
                src_x = src_x.permute(0, 2, 1)  # (batch_size, seq_length, feat_dim)
            src_feature = model.encoder_stage(src_x)

            src_pred_cl, _ = model.label_classifier(src_feature)

            src_pred_cl = torch.argmax(src_pred_cl, dim=1)
            src_cl = torch.argmax(src_cl, dim=1)
            val_cl_correct += torch.sum(src_pred_cl == src_cl).item()
        for batch_idx, tdata in enumerate(target_dataloader):
            p = float(batch_idx) / len(target_dataloader)
            constant = 2. / (1. + np.exp(-10 * p)) - 1

            tgt_lq, tgt_x, tgt_cl = tdata
            tgt_lq, tgt_x, tgt_cl = tgt_lq.cuda(), tgt_x.cuda(), tgt_cl.cuda()

            tgt_dl = torch.zeros(tgt_x.size(0), 2)  # 维度为（input2.size()[0], 2）的一张量
            tgt_dl[:, 0], tgt_dl[:, 1] = 0, 1

            if model.encoder.name == "TSTransformerEncoder":
                tgt_x = tgt_x.permute(0, 2, 1)  # (batch_size, seq_length, feat_dim)
            tgt_feature = model.encoder_stage(tgt_x)

            tgt_pred_cl, _ = model.label_classifier(tgt_feature)
            # print("tgt:", output2, label2)
            tgt_pred_cl = torch.argmax(tgt_pred_cl, dim=1)
            tgt_cl = torch.argmax(tgt_cl, dim=1)
            tgt_cl_correct += torch.sum(tgt_pred_cl == tgt_cl).item()

            tgt_pred_dl, _ = model.domain_classifier(tgt_feature, constant)
            tgt_pred_dl = torch.argmax(tgt_pred_dl, dim=1).cuda()
            tgt_dl = torch.argmax(tgt_dl, dim=1).cuda()
            tgt_dl_correct += torch.sum(tgt_pred_dl == tgt_dl).item()

            tgt_cl_tensor_list.append(tgt_cl)
            tgt_pred_cl_tensor_list.append(tgt_pred_cl)
    total_dl_correct = tgt_dl_correct + src_dl_correct
    src_num = len(source_dataloader.dataset)
    val_num = len(val_dataloader.dataset)
    tgt_num = len(target_dataloader.dataset)

    # label acc
    src_cl_acc = 100. * float(src_cl_correct) / src_num
    val_cl_acc = 100. * float(val_cl_correct) / val_num
    tgt_cl_acc = 100. * float(tgt_cl_correct) / tgt_num

    # domain acc
    total_dl_acc = 100. * float(total_dl_correct) / (src_num + tgt_num)

    tgt_cl = torch.cat(tgt_cl_tensor_list, dim=0).detach().cpu().numpy()
    tgt_pred_cl = torch.cat(tgt_pred_cl_tensor_list, dim=0).detach().cpu().numpy()

    # auc
    fpr, tpr, thresholds = roc_curve(tgt_cl, tgt_pred_cl)
    auc_value = auc(fpr, tpr)

    # sensitivity
    sensitivity = recall_score(tgt_cl, tgt_pred_cl)

    # specificity
    specificity = recall_score(np.logical_not(tgt_cl), np.logical_not(tgt_pred_cl))

    if log_writer is not None:
        log_writer.add_scalar("test/acc/src_cl_acc", src_cl_acc, epoch)
        log_writer.add_scalar("test/acc/val_cl_acc", val_cl_acc, epoch)
        log_writer.add_scalar("test/acc/tgt_cl_acc", tgt_cl_acc, epoch)
        log_writer.add_scalar("test/acc/total_dl_acc", total_dl_acc, epoch)
        log_writer.add_scalar("test/auc", auc_value, epoch)
        log_writer.add_scalar("test/sensitivity", sensitivity, epoch)
        log_writer.add_scalar("test/specificity", specificity, epoch)
    return src_cl_acc, val_cl_acc, tgt_cl_acc, total_dl_acc, auc_value, sensitivity, specificity


def create_logger(output_dir, name='', use_filehandler=True):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    if use_filehandler:
        # create file handlers
        file_handler = logging.FileHandler(os.path.join(output_dir, f'log.txt'), mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(file_handler)

    # create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console_handler)

    return logger


def metric_model(config):
    logger.info(f"Creating model:{config.MODEL.NAME}")
    model = build_model(config)
    optimizers = {
        "encoder": optim.RAdam([{'params': model.encoder.parameters()}]),
        "label_classifier": optim.RAdam([{'params': model.label_classifier.parameters()}]),
        "domain_classifier": optim.RAdam([{'params': model.domain_classifier.parameters()}]),
        # "decoder": optim.RAdam([{'params': model.decoder.parameters()}])
    }
    model.cuda()
    logger.info(str(model))
    # lr_schedulers = None
    lr_schedulers = {key: build_scheduler(config, optimizers[key], len(train_loader)) for key in optimizers.keys()}
    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT, logger)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')
    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model, optimizers, lr_schedulers, logger)
    else:
        max_accuracy = 0
    logger.info(f"Start Metric")
    test_start_time = torch.cuda.Event(enable_timing=True)
    test_end_time = torch.cuda.Event(enable_timing=True)
    test_start_time.record()
    src_cl_acc, val_cl_acc, tgt_cl_acc, total_dl_acc, auc_value, sensitivity, specificity = test_model(config, model,
                                                                                                       train_loader,
                                                                                                       val_loader,
                                                                                                       test_loader,
                                                                                                       None,
                                                                                                       None)

    test_end_time.record()
    torch.cuda.synchronize()
    test_time = test_start_time.elapsed_time(test_end_time)
    return tgt_cl_acc, total_dl_acc, auc_value, sensitivity, specificity, test_time


if __name__ == "__main__":
    mode = "train"
    if mode == "train":
        args = argparse.Namespace(cfg='config_file.txt')
        args, config = parse_option(args)
        origin_output = config.OUTPUT
        model_name_path = origin_output.rsplit(f"/{config.TAG}")[0]
        txt_output_path = model_name_path
        origin_output = f"{model_name_path}/{config.TAG}"
        statistics = []
        for i in range(1, config.TRAIN.LOOP + 1):
            # for i in range(1, 2):
            config.defrost()
            config.OUTPUT = "{}_{}".format(origin_output, str(i))
            config.freeze()
            os.makedirs(config.OUTPUT, exist_ok=True)
            logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}_loop_{i}")
            path = os.path.join(config.OUTPUT, "config.json")
            with open(path, "w") as f:
                f.write(config.dump())
            logger.info(f"Full config saved to {path}")
            log_writer = SummaryWriter(log_dir=config.OUTPUT)
            logger.info(config.dump())
            statistics.append(main(config, log_writer))

        (max_accuracy, src_cl_acc, val_cl_acc, tgt_cl_acc, total_dl_acc, auc_value,
         sensitivity, specificity, train_time, test_time) = np.array(statistics).mean(axis=0)
        print("statistics:\n", statistics)
        print("max_accuracy: {}, src_cl_acc: {}, val_cl_acc: {}, tgt_cl_acc: {}, total_dl_acc: {}, auc_value: {}, "
              "sensitivity: {}, specificity: {}, train_time: {}, test_time: {}".format(
            max_accuracy, src_cl_acc, val_cl_acc, tgt_cl_acc, total_dl_acc, auc_value,
            sensitivity, specificity, train_time, test_time))
        with open(f"{txt_output_path}/{config.TAG}_statistics_record.txt", "a") as f:
            f.write(str(statistics))
            f.write(
                "\nmax_accuracy: {}, src_cl_acc: {}, val_cl_acc: {}, tgt_cl_acc: {}, total_dl_acc: {}, auc_value: {}, "
                "sensitivity: {}, specificity: {}, train_time: {}, test_time: {}\n".format(
                    max_accuracy, src_cl_acc, val_cl_acc, tgt_cl_acc, total_dl_acc, auc_value,
                    sensitivity, specificity, train_time, test_time))
    elif mode == "test":
        args = argparse.Namespace(cfg='config_file.txt')
        args, config = parse_option(args)
        origin_output = config.OUTPUT
        model_name_path = origin_output.rsplit(f"/{config.TAG}")[0]
        txt_output_path = model_name_path
        log_output_path = model_name_path
        origin_output = f"{model_name_path}/{config.TAG}"
        logger = create_logger(output_dir=log_output_path, name=f"{config.MODEL.NAME}", use_filehandler=False)
        statistics = []
        for i in range(1, config.TRAIN.LOOP + 1):
            config.defrost()
            config.OUTPUT = "{}_{}".format(origin_output, str(i))
            config.freeze()
            os.makedirs(config.OUTPUT, exist_ok=True)
            logger.info(config.dump())
            statistics.append(metric_model(config))

        (tgt_cl_acc, total_dl_acc, auc_value, sensitivity, specificity, test_time) = np.array(statistics).mean(axis=0)
        print("statistics:\n", statistics)
        print("tgt_cl_acc: {}, total_dl_acc: {}, auc_value: {}, sensitivity: {}, specificity: {}, test_time: {}".format(
            tgt_cl_acc, total_dl_acc, auc_value, sensitivity, specificity, test_time))
        with open(f"{txt_output_path}/{config.TAG}_metric_statistics_record.txt", "a") as f:
            f.write(str(statistics))
            f.write(
                "\ntgt_cl_acc: {}, total_dl_acc: {}, auc_value: {}, sensitivity: {}, specificity: {}, test_time: {}\n".format(
                    tgt_cl_acc, total_dl_acc, auc_value, sensitivity, specificity, test_time))
    elif mode == "summary":
        args = argparse.Namespace(cfg='config_file.txt')
        args, config = parse_option(args)
        log_writer = SummaryWriter(log_dir="/media/zrpgs/86286B2C286B1A87/wsd/code/DA_FEDA/TS-CATMA/summary/")
        # model_name_list = ["fcn", "resnet", "inception_time", "lstm_fcn", "ts-cama"]
        model_name_list = ["ts-cama"]
        flops_list, params_list = [], []
        for model_name in model_name_list:
            flops, params = model_summary(config, log_writer, model_name)
            flops_list.append(flops)
            params_list.append(params)
        for i in range(len(model_name_list)):
            print(f"{model_name_list[i]}: FLOPs is {flops_list[i]}, Params is {params_list[i]}")

