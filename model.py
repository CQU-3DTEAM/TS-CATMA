import torch
import torch as t
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pandas as pd
import torch.utils.data as Data
from torch.autograd import Variable
from transformer_endecoder import build_en_decoder
from tsai.models.FCNPlus import FCNPlus


class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer  
    """

    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant  ## .neg(), Returns the input tensor as negative by element, out=−1∗input
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.feature = nn.Linear(360, 200)  ## input layer, according to the data format
        self.batch = nn.BatchNorm1d(200)  ## normalization

    def forward(self, x):
        x = F.relu(self.feature(x))  ## The activation function relu
        x = self.batch(x)
        return x


class class_classifier(nn.Module):
    def __init__(self, input_dim=22, seq_length=360, num_classes=2, hidden_dim=128):
        super(class_classifier, self).__init__()

        # 输入维度
        self.input_dim = input_dim
        self.seq_length = seq_length

        # 计算全连接层的输入维度
        fc_input_dim = input_dim * seq_length
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, num_classes)
        )

        # 初始化权重
        self.apply(self._init_weights)

    def forward(self, x):
        # 将输入数据展平
        x = x.view(x.size(0), -1)
        # 经过全连接层
        x = self.fc(x)
        # 返回softmax以便于计算交叉熵损失
        return F.softmax(x, dim=1), x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class Domain_classifier(nn.Module):
    def __init__(self, input_dim=22, seq_length=360, num_classes=2, hidden_dim=128):
        super(Domain_classifier, self).__init__()

        # 输入维度
        self.input_dim = input_dim
        self.seq_length = seq_length

        # 计算全连接层的输入维度
        fc_input_dim = input_dim * seq_length

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, num_classes)
        )

        # 初始化权重
        self.apply(self._init_weights)

    def forward(self, x, constant, reverse_grad=True):
        if reverse_grad:
            input_x = GradReverse.grad_reverse(x, constant)  ##  reverse the gradient.
        else:
            input_x = x
        # 将输入数据展平
        x = input_x.view(input_x.size(0), -1)
        # 经过全连接层
        x = self.fc(x)
        # 返回softmax以便于计算交叉熵损失
        return F.softmax(x, dim=1), x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class class_classifier_fcnhead(nn.Module):
    def __init__(self, input_dim=22, seq_length=360, num_classes=2, hidden_dim=128):
        super(class_classifier_fcnhead, self).__init__()

        # 输入维度
        self.input_dim = input_dim
        self.seq_length = seq_length

        self.fcn_head = FCNPlus(self.input_dim, c_out=num_classes, layers=[128, 256, 128], kss=[7, 5, 3], use_bn=True, residual=True).head

        # 计算全连接层的输入维度
        # fc_input_dim = input_dim * seq_length
        # # 全连接层
        # self.fc = nn.Sequential(
        #     nn.Linear(fc_input_dim, hidden_dim),
        #     nn.Sigmoid(),
        #     nn.Linear(hidden_dim, num_classes)
        # )
        #
        # # 初始化权重
        # self.apply(self._init_weights)

    def forward(self, x):
        # # 将输入数据展平
        # x = x.view(x.size(0), -1)
        # # 经过全连接层
        # x = self.fc(x)
        # 返回softmax以便于计算交叉熵损失
        x = self.fcn_head(x)
        return F.softmax(x, dim=1), x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class Domain_classifier_fcnhead(nn.Module):
    def __init__(self, input_dim=22, seq_length=360, num_classes=2, hidden_dim=128):
        super(Domain_classifier_fcnhead, self).__init__()

        # 输入维度
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.fcn_head = FCNPlus(self.input_dim, c_out=num_classes, layers=[128, 256, 128], kss=[7, 5, 3], use_bn=True, residual=True).head

        # # 计算全连接层的输入维度
        # fc_input_dim = input_dim * seq_length
        #
        # # 全连接层
        # self.fc = nn.Sequential(
        #     nn.Linear(fc_input_dim, hidden_dim),
        #     nn.Sigmoid(),
        #     nn.Linear(hidden_dim, num_classes)
        # )
        #
        # # 初始化权重
        # self.apply(self._init_weights)

    def forward(self, x, constant, reverse_grad=True):
        if reverse_grad:
            input_x = GradReverse.grad_reverse(x, constant)  ##  reverse the gradient.
        else:
            input_x = x
        x = self.fcn_head(input_x)
        # # 将输入数据展平
        # x = input_x.view(input_x.size(0), -1)
        # # 经过全连接层
        # x = self.fc(x)
        # # 返回softmax以便于计算交叉熵损失

        return F.softmax(x, dim=1), x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class AttentionLoss(nn.Module):
    def __init__(self, att_dim, num_channels):
        super(AttentionLoss, self).__init__()
        self.num_channels = num_channels
        self.attention = nn.Sequential(
            nn.Linear(num_channels * att_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.mse_loss = nn.MSELoss(reduction='none')

        # 初始化权重
        self.apply(self._init_weights)

    def forward(self, x, y):
        # x: 重构数据, shape=(batch_size, num_channels, seq_len)
        # y: 原始数据, shape=(batch_size, num_channels, seq_len)
        batch_size, num_channels, seq_len = x.size()

        # 将通道维度和时间维度合并
        x_flat = x.view(batch_size, -1)  # (batch_size, num_channels * seq_len)

        # 计算注意力权重
        att_weight = self.attention(x_flat)  # (batch_size, 1)
        att_weight = att_weight.view(batch_size, 1, 1)  # (batch_size, 1, 1)
        att_weight = torch.softmax(att_weight, dim=1)  # 对通道维度做softmax

        # 计算加权损失
        loss = self.mse_loss(x, y)  # (batch_size, num_channels, seq_len)
        weighted_loss = loss * att_weight
        weighted_loss = weighted_loss.mean()

        return weighted_loss

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class TSCATMA(nn.Module):
    def __init__(self, encoder, decoder, config):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.label_classifier = class_classifier(input_dim=config.MODEL.LABEL_CLS.input_dim,
                                                 seq_length=config.MODEL.LABEL_CLS.seq_length,
                                                 num_classes=config.MODEL.LABEL_CLS.num_classes,
                                                 hidden_dim=config.MODEL.LABEL_CLS.hidden_dim)
        self.domain_classifier = Domain_classifier(input_dim=config.MODEL.DOMAIN_CLS.input_dim,
                                                   seq_length=config.MODEL.DOMAIN_CLS.seq_length,
                                                   num_classes=config.MODEL.DOMAIN_CLS.num_classes,
                                                   hidden_dim=config.MODEL.DOMAIN_CLS.hidden_dim)
        self.recon_criterion = nn.MSELoss()
        # self.recon_criterion = AttentionLoss(config.MODEL.LABEL_CLS.seq_length, config.MODEL.LABEL_CLS.input_dim)
        self.label_criterion1 = nn.BCEWithLogitsLoss()
        self.label_criterion2 = LabelSmoothingLoss(classes=2, smoothing=0.08)
        self.domain_criterion = nn.BCEWithLogitsLoss()

    def encoder_stage(self, x):
        encoder_names = [
            "FCNEncoder",
            "FCNPlusEncoder",
            "FCNPlusEMAEncoder",
            "FCNPlusEncoderBackbone",
            "ResnetEncoder",
            "ResnetEMAEncoder",
            "LSTMEncoder",
            "LSTMEMAEncoder",
            "TransformerEncoder",
            "TransformerEMAEncoder",
        ]
        if self.encoder.name == "TSTransformerEncoder":
            padding_masks = torch.ones(x.shape[0], x.shape[1], dtype=torch.bool).cuda()
            feature = self.encoder(x, padding_masks)  # (batch_size, seq_length, feat_dim)
        elif self.encoder.name in encoder_names:
            feature = self.encoder(x)
        else:
            print("No encoder name Matched")
            feature = self.encoder(x)
        return feature

    def decoder_stage(self, x):
        if self.encoder.name == "TSTransformerDecoder":
            padding_masks = torch.ones(x.shape[0], x.shape[1], dtype=torch.bool).cuda()
            recon = self.decoder(x, padding_masks)  # (batch_size, seq_length, feat_dim)
        elif self.encoder.name == "LinearDecoder":
            padding_masks = torch.ones(x.shape[0], x.shape[1], dtype=torch.bool).cuda()
            recon = self.decoder(x, padding_masks)  # (batch_size, seq_length, feat_dim)
        else:
            recon = x
        return recon

    def to_op(self, src_lq, src_x, src_cl, src_dl, tgt_lq, tgt_x, tgt_dl, constant, mask=None):
        if self.encoder.name == "TSTransformerEncoder":
            src_x = src_x.permute(0, 2, 1)  # (batch_size, seq_length, feat_dim)
            tgt_x = tgt_x.permute(0, 2, 1)
            src_lq = src_lq.permute(0, 2, 1)
            tgt_lq = tgt_lq.permute(0, 2, 1)

        # encoder stage
        src_feature = self.encoder_stage(src_x)
        tgt_feature = self.encoder_stage(tgt_x)
        src_lq_feature = self.encoder_stage(src_lq)
        tgt_lq_feature = self.encoder_stage(tgt_lq)

        # label classifier
        src_cls_preds, src_cls_logits = self.label_classifier(src_feature)
        # tgt_cls_preds, tgt_logits = self.label_classifier(src_feature)
        # calculate label loss
        label_loss = self.label_criterion(src_cls_logits, src_cl)  # Calculate the cross-loss of the source domain

        # domain classifier
        src_dom_preds, src_dom_logits = self.domain_classifier(src_feature, constant, )
        tgt_dom_preds, tgt_dom_logits = self.domain_classifier(tgt_feature, constant)
        # calculate domain loss
        tgt_loss = self.domain_criterion(src_dom_logits, src_dl)  # source domain adversarial loss
        src_loss = self.domain_criterion(tgt_dom_logits, tgt_dl)  # target domain adversarial loss
        domain_loss = tgt_loss + src_loss  # domain loss = target adversarial loss + source adversarial loss

        # decoder stage
        src_recon = self.decoder_stage(src_lq_feature)
        tgt_recon = self.decoder_stage(tgt_lq_feature)
        # calculate reconstruction loss
        src_recon_loss = self.recon_criterion(src_x.permute(0, 2, 1), src_recon.permute(0, 2, 1))
        tgt_recon_loss = self.recon_criterion(tgt_x.permute(0, 2, 1), tgt_recon.permute(0, 2, 1))
        recon_loss = src_recon_loss + tgt_recon_loss

        return label_loss, domain_loss, recon_loss

    def forward(self, src_lq, src_x, src_cl, tgt_lq, tgt_x, constant=0, mask=None, train_mode="summary"):
        if train_mode == "domain+recon":
            # 创建源标签和目标标签
            src_dl = torch.zeros(src_x.size(0), 2).cuda()
            tgt_dl = torch.zeros(tgt_x.size(0), 2).cuda()
            # 将源标签的第一列置为1，第二列置为0
            src_dl[:, 0], src_dl[:, 1] = 1, 0
            # 将目标标签的第一列置为0，第二列置为1
            tgt_dl[:, 0], tgt_dl[:, 1] = 0, 1

            if self.encoder.name == "TSTransformerEncoder":
                src_x = src_x.permute(0, 2, 1)  # (batch_size, seq_length, feat_dim)
                tgt_x = tgt_x.permute(0, 2, 1)
                # src_lq = src_lq.permute(0, 2, 1)
                # tgt_lq = tgt_lq.permute(0, 2, 1)

            # encoder stage
            src_feature = self.encoder_stage(src_x)
            tgt_feature = self.encoder_stage(tgt_x)
            # src_lq_feature = self.encoder_stage(src_lq)
            # tgt_lq_feature = self.encoder_stage(tgt_lq)

            # domain classifier
            src_dom_preds, src_dom_logits = self.domain_classifier(src_feature, constant, reverse_grad=False)
            tgt_dom_preds, tgt_dom_logits = self.domain_classifier(tgt_feature, constant, reverse_grad=False)
            # calculate domain loss
            tgt_loss = self.domain_criterion(src_dom_logits, src_dl)  # source domain adversarial loss
            src_loss = self.domain_criterion(tgt_dom_logits, tgt_dl)  # target domain adversarial loss
            domain_loss = tgt_loss + src_loss  # domain loss = target adversarial loss + source adversarial loss

            # # decoder stage
            # src_recon = self.decoder_stage(src_lq_feature)
            # tgt_recon = self.decoder_stage(tgt_lq_feature)
            # # calculate reconstruction loss
            # src_recon_loss = self.recon_criterion(src_x.permute(0, 2, 1), src_recon.permute(0, 2, 1))
            # tgt_recon_loss = self.recon_criterion(tgt_x.permute(0, 2, 1), tgt_recon.permute(0, 2, 1))
            # recon_loss = src_recon_loss + tgt_recon_loss

            # domain grad penalty
            grad_penalty_src = gradient_penalty(src_feature, src_dom_preds, factor=0.1)
            grad_penalty_tgt = gradient_penalty(tgt_feature, tgt_dom_preds, factor=0.1)
            domain_loss = domain_loss + (grad_penalty_src + grad_penalty_tgt) / 2

            return None, domain_loss, None

        elif train_mode == "label_only":
            if self.encoder.name == "TSTransformerEncoder":
                src_x = src_x.permute(0, 2, 1)  # (batch_size, seq_length, feat_dim)
            # encoder stage
            src_feature = self.encoder_stage(src_x)
            # label classifier
            src_cls_preds, src_cls_logits = self.label_classifier(src_feature)
            # calculate label loss
            label_loss = self.label_criterion1(src_cls_logits, src_cl)  # Calculate the cross-loss of the source domain
            return label_loss, None, None

        elif train_mode == "encoder+label":
            # 创建源标签和目标标签
            src_dl = torch.zeros(src_x.size(0), 2).cuda()
            tgt_dl = torch.zeros(tgt_x.size(0), 2).cuda()
            # 将源标签的第一列置为0，第二列置为1
            src_dl[:, 0], src_dl[:, 1] = 0, 1
            # 将目标标签的第一列置为1，第二列置为0
            tgt_dl[:, 0], tgt_dl[:, 1] = 1, 0

            if self.encoder.name == "TSTransformerEncoder":
                src_x = src_x.permute(0, 2, 1)  # (batch_size, seq_length, feat_dim)
                tgt_x = tgt_x.permute(0, 2, 1)

            # encoder stage
            src_feature = self.encoder_stage(src_x)
            tgt_feature = self.encoder_stage(tgt_x)

            # label classifier
            src_cls_preds, src_cls_logits = self.label_classifier(src_feature)
            # calculate label loss
            label_loss = self.label_criterion1(src_cls_logits, src_cl)  # Calculate the cross-loss of the source domain

            # domain classifier
            src_dom_preds, src_dom_logits = self.domain_classifier(src_feature, constant, reverse_grad=False)
            tgt_dom_preds, tgt_dom_logits = self.domain_classifier(tgt_feature, constant, reverse_grad=False)
            # calculate domain loss
            tgt_loss = self.domain_criterion(src_dom_logits, src_dl)  # source domain adversarial loss
            src_loss = self.domain_criterion(tgt_dom_logits, tgt_dl)  # target domain adversarial loss
            domain_loss = tgt_loss + src_loss  # domain loss = target adversarial loss + source adversarial loss

            # encoder l2 penalty
            # l2_loss = model_l2_penalty(self.encoder, 1)
            # label_loss = label_loss + l2_loss

            # domain grad penalty
            # grad_penalty_src = gradient_penalty(src_feature, src_dom_preds, factor=0.01)
            # grad_penalty_tgt = gradient_penalty(tgt_feature, tgt_dom_preds, factor=0.01)
            # domain_loss = domain_loss + (grad_penalty_src + grad_penalty_tgt) / 2

            return label_loss, domain_loss, None

        elif train_mode == "encoder+label_pseudo":
            # 创建源标签和目标标签
            src_dl = torch.zeros(src_x.size(0), 2).cuda()
            tgt_dl = torch.zeros(tgt_x.size(0), 2).cuda()
            # 将源标签的第一列置为0，第二列置为1
            src_dl[:, 0], src_dl[:, 1] = 0, 1
            # 将目标标签的第一列置为1，第二列置为0
            tgt_dl[:, 0], tgt_dl[:, 1] = 1, 0

            if self.encoder.name == "TSTransformerEncoder":
                src_x = src_x.permute(0, 2, 1)  # (batch_size, seq_length, feat_dim)
                tgt_x = tgt_x.permute(0, 2, 1)
                com_x = src_lq.permute(0, 2, 1)
            else:
                com_x = src_lq

            com_cl = src_cl

            # encoder stage
            src_feature = self.encoder_stage(src_x)
            tgt_feature = self.encoder_stage(tgt_x)
            com_feature = self.encoder_stage(com_x)

            # label classifier
            com_cls_preds, com_cls_logits = self.label_classifier(com_feature)
            # calculate label loss
            # label_loss = self.label_criterion(com_cls_preds, com_cl)  # Calculate the cross-loss of the source domain
            label_loss = self.label_criterion2(com_cls_logits, com_cl)

            # domain classifier
            src_dom_preds, src_dom_logits = self.domain_classifier(src_feature, constant, reverse_grad=False)
            tgt_dom_preds, tgt_dom_logits = self.domain_classifier(tgt_feature, constant, reverse_grad=False)
            # calculate domain loss
            tgt_loss = self.domain_criterion(src_dom_logits, src_dl)  # source domain adversarial loss
            src_loss = self.domain_criterion(tgt_dom_logits, tgt_dl)  # target domain adversarial loss
            domain_loss = tgt_loss + src_loss  # domain loss = target adversarial loss + source adversarial loss

            # encoder l2 penalty
            # l2_loss = model_l2_penalty(self.encoder, 1)
            # label_loss = label_loss + l2_loss

            # domain grad penalty
            # grad_penalty_src = gradient_penalty(src_feature, src_dom_preds, factor=0.01)
            # grad_penalty_tgt = gradient_penalty(tgt_feature, tgt_dom_preds, factor=0.01)
            # domain_loss = domain_loss + (grad_penalty_src + grad_penalty_tgt) / 2

            return label_loss, domain_loss, None

        elif train_mode == "summary":
            # 创建源标签和目标标签
            src_dl = torch.zeros(src_x.size(0), 2).cuda()
            tgt_dl = torch.zeros(tgt_x.size(0), 2).cuda()
            # 将源标签的第一列置为1，第二列置为0
            src_dl[:, 0], src_dl[:, 1] = 1, 0
            # 将目标标签的第一列置为0，第二列置为1
            tgt_dl[:, 0], tgt_dl[:, 1] = 0, 1

            if self.encoder.name == "TSTransformerEncoder":
                src_x = src_x.permute(0, 2, 1)  # (batch_size, seq_length, feat_dim)
                tgt_x = tgt_x.permute(0, 2, 1)

            # encoder stage
            src_feature = self.encoder_stage(src_x)
            tgt_feature = self.encoder_stage(tgt_x)

            # label classifier
            src_cls_preds, src_cls_logits = self.label_classifier(src_feature)

            # domain classifier
            src_dom_preds, src_dom_logits = self.domain_classifier(src_feature, constant, reverse_grad=False)
            tgt_dom_preds, tgt_dom_logits = self.domain_classifier(tgt_feature, constant, reverse_grad=False)

            return src_cls_logits + src_dom_logits + tgt_dom_logits
        else:
            # 创建源标签和目标标签
            src_dl = torch.zeros(src_x.size(0), 2).cuda()
            tgt_dl = torch.zeros(tgt_x.size(0), 2).cuda()
            # 将源标签的第一列置为1，第二列置为0
            src_dl[:, 0], src_dl[:, 1] = 1, 0
            # 将目标标签的第一列置为0，第二列置为1
            tgt_dl[:, 0], tgt_dl[:, 1] = 0, 1
            label_loss, domain_loss, recon_loss = self.to_op(src_lq, src_x, src_cl, src_dl, tgt_lq, tgt_x, tgt_dl,
                                                             constant, mask=None)
            return label_loss, domain_loss, recon_loss


class ToyModel(nn.Module):
    def __init__(self, encoder, decoder, config):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.label_classifier = class_classifier(input_dim=config.MODEL.LABEL_CLS.input_dim,
                                                 seq_length=config.MODEL.LABEL_CLS.seq_length,
                                                 num_classes=config.MODEL.LABEL_CLS.num_classes,
                                                 hidden_dim=config.MODEL.LABEL_CLS.hidden_dim)
        self.domain_classifier = Domain_classifier(input_dim=config.MODEL.DOMAIN_CLS.input_dim,
                                                   seq_length=config.MODEL.DOMAIN_CLS.seq_length,
                                                   num_classes=config.MODEL.DOMAIN_CLS.num_classes,
                                                   hidden_dim=config.MODEL.DOMAIN_CLS.hidden_dim)

    def encoder_stage(self, x):
        encoder_names = [
            "FCNEncoder",
            "FCNPlusEncoder",
            "FCNPlusEMAEncoder",
            "FCNPlusEncoderBackbone",
            "ResnetEncoder",
            "ResnetEMAEncoder",
            "LSTMEncoder",
            "LSTMEMAEncoder",
            "TransformerEncoder",
            "TransformerEMAEncoder",
        ]
        if self.encoder.name == "TSTransformerEncoder":
            padding_masks = torch.ones(x.shape[0], x.shape[1], dtype=torch.bool).cuda()
            feature = self.encoder(x, padding_masks)  # (batch_size, seq_length, feat_dim)
        elif self.encoder.name in encoder_names:
            feature = self.encoder(x)
        else:
            print("No encoder name Matched")
            feature = self.encoder(x)
        return feature

    def decoder_stage(self, x):
        if self.encoder.name == "TSTransformerDecoder":
            padding_masks = torch.ones(x.shape[0], x.shape[1], dtype=torch.bool).cuda()
            recon = self.decoder(x, padding_masks)  # (batch_size, seq_length, feat_dim)
        elif self.encoder.name == "LinearDecoder":
            padding_masks = torch.ones(x.shape[0], x.shape[1], dtype=torch.bool).cuda()
            recon = self.decoder(x, padding_masks)  # (batch_size, seq_length, feat_dim)
        else:
            recon = x
        return recon

    def forward(self, src_x, constant=0, mask=None, train_mode="summary"):
        if train_mode == "summary":
            if self.encoder.name == "TSTransformerEncoder":
                src_x = src_x.permute(0, 2, 1)  # (batch_size, seq_length, feat_dim)

            # encoder stage
            src_feature = self.encoder_stage(src_x)
            tgt_feature = self.encoder_stage(src_x)

            # label classifier
            src_cls_preds, src_cls_logits = self.label_classifier(src_feature)

            # domain classifier
            src_dom_preds, src_dom_logits = self.domain_classifier(src_feature, constant, reverse_grad=False)
            tgt_dom_preds, tgt_dom_logits = self.domain_classifier(tgt_feature, constant, reverse_grad=False)

            return src_cls_logits + src_dom_logits + tgt_dom_logits


def model_l2_penalty(model, l2_lambda=0.01):
    l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
    return l2_lambda * l2_norm


def gradient_penalty(features, preds, factor=0.01):
    gradients = torch.autograd.grad(outputs=preds, inputs=features,
                                    grad_outputs=torch.ones_like(preds),
                                    create_graph=True, retain_graph=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    return factor * gradient_penalty


def gradient_penalty_v2(critic, real_data, fake_data, factor=0.01):
    # 梯度惩罚
    alpha = torch.rand(real_data.size(0), 1).cuda()
    interpolated_features = alpha * real_data + (1 - alpha) * fake_data
    interpolated_features.requires_grad_(True)
    interpolated_predictions = critic(interpolated_features)
    gradients = torch.autograd.grad(outputs=interpolated_predictions, inputs=interpolated_features,
                                    grad_outputs=torch.ones_like(interpolated_predictions),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return factor * gradient_penalty


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # 创建平滑的标签
            true_dist = target * self.confidence + (1 - target) * self.smoothing / (self.cls - 1)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def build_model(config):
    encoder, decoder = build_en_decoder(config)
    model = TSCATMA(encoder, decoder, config)
    return model


def build_toy_model(config):
    encoder, decoder = build_en_decoder(config)
    model = ToyModel(encoder, decoder, config)
    return model


def loss_function(epoch, label_loss, domain_loss, recon_loss, mode="all"):
    # if recon_loss == 0.01:
    #     loss = 0.4 * label_loss + 0.6 * domain_loss
    # else:
    #     loss = 0.6 * label_loss + 0.4 * domain_loss + recon_loss  # total loss

    # if epoch <= 100:
    #     loss = 0 * label_loss + 0 * domain_loss + 1 * recon_loss
    # else:
    #     loss = 0.33 * label_loss + 0.33 * domain_loss + 0.33 * recon_loss
    if mode == "domain+recon":
        # recon_loss = recon_loss * 1e-6
        # loss = 1 * domain_loss + 1 * recon_loss
        loss = 1 * domain_loss
    elif mode == "label_only":
        loss = 1 * label_loss
    elif mode == "encoder+label":
        loss = 1 * label_loss + 1 * domain_loss
    elif mode == "encoder+pseudolabel":
        loss = 0.8 * label_loss + 1 * domain_loss
    else:
        recon_loss = recon_loss * 1e-6
        loss = 1 * label_loss + 1 * domain_loss + 0 * recon_loss
    return loss
