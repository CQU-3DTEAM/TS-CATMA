from typing import Optional, Any
import math

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer, \
    TransformerDecoderLayer
from timm.models.layers import trunc_normal_
from tsai.models.RNN import LSTM
from tsai.models.layers import GAP1d, ConvBlock, Permute, Squeeze
from tsai.models.FCNPlus import FCNPlus, _FCNBlockPlus
from tsai.models.RNNPlus import RNNPlus
from tsai.models.RNN_FCNPlus import RNN_FCNPlus
from tsai.models.ResNet import ResNet
from tsai.models.TransformerModel import TransformerModel


def build_en_decoder(config):
    # data windowing is used when samples don't have a predefined length or the length is too long
    max_seq_len = config.MODEL.data_window_len if config.MODEL.data_window_len is not None else config.MODEL.max_seq_len
    if config.MODEL.ENCODER.model == 'tst':
        encoder_config = config.MODEL.ENCODER
        encoder = TSTransformerEncoder(feat_dim=encoder_config.feat_dim, max_len=max_seq_len,
                                       d_model=encoder_config.d_model, n_heads=encoder_config.num_heads,
                                       num_layers=encoder_config.num_layers,
                                       dim_feedforward=encoder_config.dim_feedforward, dropout=encoder_config.dropout,
                                       pos_encoding=encoder_config.pos_encoding, activation=encoder_config.activation,
                                       norm=encoder_config.normalization_layer, freeze=encoder_config['freeze'])
    elif config.MODEL.ENCODER.model == 'fcn':
        encoder_config = config.MODEL.ENCODER
        encoder = FCNEncoder(c_in=encoder_config.feat_dim, c_out=encoder_config.out_dim)
    elif config.MODEL.ENCODER.model == 'fcnplus':
        encoder_config = config.MODEL.ENCODER
        encoder = FCNPlusEncoder(c_in=encoder_config.feat_dim, c_out=encoder_config.out_dim)
    elif config.MODEL.ENCODER.model == 'fcnplus_bb':
        encoder_config = config.MODEL.ENCODER
        encoder = FCNPlusEncoderBackbone(c_in=encoder_config.feat_dim, c_out=encoder_config.out_dim)
    elif config.MODEL.ENCODER.model == 'fcnplus_ema':
        encoder_config = config.MODEL.ENCODER
        encoder = FCNPlusEMAEncoder(c_in=encoder_config.feat_dim, c_out=encoder_config.out_dim)
    elif config.MODEL.ENCODER.model == 'resnet':
        encoder_config = config.MODEL.ENCODER
        encoder = ResnetEncoder(c_in=encoder_config.feat_dim, c_out=encoder_config.out_dim)
    elif config.MODEL.ENCODER.model == 'resnet_ema':
        encoder_config = config.MODEL.ENCODER
        encoder = ResnetEMAEncoder(c_in=encoder_config.feat_dim, c_out=encoder_config.out_dim)
    elif config.MODEL.ENCODER.model == 'lstm':
        encoder_config = config.MODEL.ENCODER
        encoder = LSTMEncoder(c_in=encoder_config.feat_dim, c_out=encoder_config.out_dim,
                              d_model=encoder_config.d_model, num_layers=encoder_config.num_layers)
    elif config.MODEL.ENCODER.model == 'lstm_ema':
        encoder_config = config.MODEL.ENCODER
        encoder = LSTMEMAEncoder(c_in=encoder_config.feat_dim, c_out=encoder_config.out_dim,
                                 d_model=encoder_config.d_model, num_layers=encoder_config.num_layers)
    elif config.MODEL.ENCODER.model == 'transformer':
        encoder_config = config.MODEL.ENCODER
        encoder = TransEncoder(c_in=encoder_config.feat_dim, c_out=encoder_config.out_dim,
                               d_model=encoder_config.d_model, num_layers=encoder_config.num_layers)
    elif config.MODEL.ENCODER.model == 'transformer_ema':
        encoder_config = config.MODEL.ENCODER
        encoder = TransEMAEncoder(c_in=encoder_config.feat_dim, c_out=encoder_config.out_dim,
                                  d_model=encoder_config.d_model, num_layers=encoder_config.num_layers)
    else:
        encoder = None
        print("No Implement Encoder!")
    if config.MODEL.DECODER.model == 'transformer':
        decoder_config = config.MODEL.DECODER
        decoder = TSTransformerDecoder(feat_dim=decoder_config.feat_dim, max_len=max_seq_len,
                                       d_model=decoder_config.d_model, n_heads=decoder_config.num_heads,
                                       num_layers=decoder_config.num_layers,
                                       dim_feedforward=decoder_config.dim_feedforward, dropout=decoder_config.dropout,
                                       pos_encoding=decoder_config.pos_encoding, activation=decoder_config.activation,
                                       norm=decoder_config.normalization_layer, freeze=decoder_config['freeze'])

    elif config.MODEL.DECODER.model == 'linear':
        decoder_config = config.MODEL.DECODER
        decoder = LinearDecoder(feat_dim=decoder_config.feat_dim, max_len=max_seq_len, d_model=decoder_config.d_model,
                                dropout=decoder_config.dropout, activation=decoder_config.activation)
    elif config.MODEL.DECODER.model == 'none':
        decoder = None
        print("Decoder is not using!")
    else:
        decoder = None
        print("No Implement Decoder!")
    return encoder, decoder


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))


# From https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class FixedPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding

    raise NotImplementedError("pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding))


class TransformerBatchNormEncoderLayer(nn.modules.Module):
    r"""This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = BatchNorm1d(d_model, eps=1e-5)  # normalizes each feature across batch samples and time steps
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerBatchNormEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        # src = src.reshape([src.shape[0], -1])  # (batch_size, seq_length * d_model)
        src = self.norm1(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        return src


class TransformerBatchNormDecoderLayer(nn.Module):
    r"""This transformer decoder layer block is made up of self-attn, multi-head attn and feedforward network.
    It differs from TransformerDecoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerBatchNormDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.BatchNorm1d(d_model, eps=1e-5)  # BatchNorm1d instead of LayerNorm
        self.norm2 = nn.BatchNorm1d(d_model, eps=1e-5)
        self.norm3 = nn.BatchNorm1d(d_model, eps=1e-5)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        '''
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)  # (seq_len, batch_size, d_model)
        tgt = tgt.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        tgt = self.norm1(tgt)
        tgt = tgt.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)  # (seq_len, batch_size, d_model)
        tgt = tgt.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        tgt = self.norm2(tgt)
        tgt = tgt.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        '''
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)  # (seq_len, batch_size, d_model)
        tgt = tgt.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        tgt = self.norm1(tgt)
        tgt = tgt.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)

        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)  # (seq_len, batch_size, d_model)
        tgt = tgt.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        tgt = self.norm2(tgt)
        tgt = tgt.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = tgt.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        tgt = self.norm3(tgt)
        tgt = tgt.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        return tgt


class TSTransformerEncoder(nn.Module):

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, dropout=0.1,
                 pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
        super(TSTransformerEncoder, self).__init__()
        self.name = "TSTransformerEncoder"

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout * (1.0 - freeze), max_len=max_len)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout * (1.0 - freeze),
                                                    activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward,
                                                             dropout * (1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_layer = nn.Linear(d_model, feat_dim)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim

        # Initialize weights
        self.apply(self._init_weights)
        self.fix_init_weight()

    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """

        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, seq_length]
        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)  # (seq_length, batch_size, d_model)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)
        # Most probably defining a Linear(d_model,feat_dim) vectorizes the operation over (seq_length, batch_size).
        output = self.output_layer(output)  # (batch_size, seq_length, feat_dim)
        return output

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.transformer_encoder.layers):
            rescale(layer.self_attn.in_proj_weight.data, layer_id + 1)
            rescale(layer.self_attn.out_proj.weight.data, layer_id + 1)
            rescale(layer.linear1.weight.data, layer_id + 1)
            rescale(layer.linear2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            self._trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            self._trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _trunc_normal_(self, tensor, mean=0., std=1.):
        # trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)
        trunc_normal_(tensor, mean=mean, std=std)


class TSTransformerDecoder(nn.Module):

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, dropout=0.1,
                 pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
        super(TSTransformerDecoder, self).__init__()
        self.name = "TSTransformerDecoder"

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout * (1.0 - freeze), max_len=max_len)

        if norm == 'LayerNorm':
            decoder_layer = TransformerDecoderLayer(d_model, self.n_heads, dim_feedforward, dropout * (1.0 - freeze),
                                                    activation=activation)
        else:
            decoder_layer = TransformerBatchNormDecoderLayer(d_model, self.n_heads, dim_feedforward,
                                                             dropout * (1.0 - freeze), activation=activation)

        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.output_layer = nn.Linear(d_model, feat_dim)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim

        # Initialize weights
        self.apply(self._init_weights)
        self.fix_init_weight()

    def forward(self, X, padding_masks=None):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            decoder_output: (seq_length, batch_size, d_model) tensor, output of the encoder
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """

        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, seq_length]
        inp = X.permute(1, 0, 2)  # (seq_length=360, batch_size=64, feat_dim=22)
        encoder_output = inp
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.pos_enc(inp)  # add positional encoding
        output = self.transformer_decoder(inp, inp, tgt_key_padding_mask=~padding_masks,
                                          memory_key_padding_mask=~padding_masks)  # (seq_length, batch_size, d_model)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)

        output = self.output_layer(output)  # (batch_size, seq_length, feat_dim)
        return output

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.transformer_decoder.layers):
            rescale(layer.self_attn.in_proj_weight.data, layer_id + 1)
            rescale(layer.self_attn.out_proj.weight.data, layer_id + 1)
            rescale(layer.linear1.weight.data, layer_id + 1)
            rescale(layer.linear2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            self._trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            self._trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _trunc_normal_(self, tensor, mean=0., std=1.):
        # trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)
        trunc_normal_(tensor, mean=mean, std=std)


class LinearDecoder(nn.Module):
    def __init__(self, feat_dim, max_len, d_model, dropout=0.1, activation='gelu', ):
        super(LinearDecoder, self).__init__()
        self.name = "LinearDecoder"
        self.max_len = max_len
        self.d_model = d_model

        self.project_inp = nn.Linear(feat_dim, d_model)

        self.output_layer = nn.Linear(d_model, feat_dim)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim

        # Initialize weights
        self.apply(self._init_weights)

    def forward(self, X, padding_mask=None):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            decoder_output: (seq_length, batch_size, d_model) tensor, output of the encoder
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """

        inp = X.permute(1, 0, 2)  # (seq_length=360, batch_size=64, feat_dim=22)
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        output = self.act(inp)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)

        output = self.output_layer(output)  # (batch_size, seq_length, feat_dim)
        return output

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            self._trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _trunc_normal_(self, tensor, mean=0., std=1.):
        # trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)
        trunc_normal_(tensor, mean=mean, std=std)


class FCNEncoder(nn.Module):
    def __init__(self, c_in, c_out, layers=None, kss=None):
        super(FCNEncoder, self).__init__()
        if kss is None:
            # kss = [7, 5, 3]
            kss = [5, 5, 3]
        if layers is None:
            layers = [128, 256, 128]
            # layers = [128,128]
        assert len(layers) == len(kss)
        self.convblock1 = ConvBlock(c_in, layers[0], kss[0])
        self.convblock2 = ConvBlock(layers[0], layers[1], kss[1])
        self.convblock3 = ConvBlock(layers[1], layers[2], kss[2])
        self.gap = GAP1d(1)
        self.fc = nn.Linear(layers[-1], c_out)
        self.name = "FCNEncoder"

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.gap(x)
        return self.fc(x)


class FCNPlusEncoder(nn.Module):
    def __init__(self, c_in, c_out):
        super(FCNPlusEncoder, self).__init__()
        self.name = "FCNPlusEncoder"
        self.encoder = FCNPlus(c_in=c_in, c_out=c_out, layers=[128, 256, 128], kss=[7, 5, 3], use_bn=True,
                               residual=True)

    def forward(self, x):
        return self.encoder(x)


class FCNPlusEMAEncoder(nn.Module):
    def __init__(self, c_in, c_out):
        super(FCNPlusEMAEncoder, self).__init__()
        self.name = "FCNPlusEMAEncoder"
        self.fcnplus = FCNPlus(c_in=c_in, c_out=c_out, layers=[128, 256, 128], kss=[7, 5, 3], use_bn=True,
                               residual=True)
        self.ema1d = EMA_1D(channels=c_out)

    def forward(self, x):
        x = self.fcnplus.backbone(x)
        x = self.ema1d(x)
        res = self.fcnplus.head(x)
        return res


class FCNPlusEncoderBackbone(nn.Module):
    def __init__(self, c_in, c_out):
        super(FCNPlusEncoderBackbone, self).__init__()
        self.name = "FCNPlusEncoderBackbone"
        self.encoder = FCNPlus(c_in=c_in, c_out=c_out, layers=[128, 256, 128], kss=[7, 5, 3], use_bn=True,
                               residual=True).backbone

    def forward(self, x):
        return self.encoder(x)


class ResnetEncoder(nn.Module):
    def __init__(self, c_in, c_out):
        super(ResnetEncoder, self).__init__()
        self.name = "ResnetEncoder"
        self.resnet = ResNet(c_in=c_in, c_out=c_out)
        self.head = FCNPlus(c_in=c_in, c_out=c_out, layers=[128, 256, 128], kss=[7, 5, 3], use_bn=True,
                            residual=True).head

    def forward(self, x):
        x = self.resnet.resblock1(x)
        x = self.resnet.resblock2(x)
        x = self.resnet.resblock3(x)
        output = self.head(x)
        return output


class ResnetEMAEncoder(nn.Module):
    def __init__(self, c_in, c_out):
        super(ResnetEMAEncoder, self).__init__()
        self.name = "ResnetEMAEncoder"
        self.resnet = ResNet(c_in=c_in, c_out=c_out)
        self.ema1d = EMA_1D(channels=c_out)
        self.head = FCNPlus(c_in=c_in, c_out=c_out, layers=[128, 256, 128], kss=[7, 5, 3], use_bn=True,
                            residual=True).head

    def forward(self, x):
        x = self.resnet.resblock1(x)
        x = self.resnet.resblock2(x)
        x = self.resnet.resblock3(x)
        x = self.ema1d(x)
        output = self.head(x)
        return output


class LSTMEncoder(nn.Module):
    def __init__(self, c_in, c_out, d_model, num_layers):
        super(LSTMEncoder, self).__init__()
        self.name = "LSTMEncoder"
        self.lstm = LSTM(c_in=c_in, c_out=c_out, hidden_size=d_model, n_layers=num_layers, rnn_dropout=0.5,
                         fc_dropout=0.5)
        head_layers = [Squeeze(-1), nn.BatchNorm1d(d_model), nn.Linear(d_model, c_out)]
        self.head = nn.Sequential(*head_layers)

    def forward(self, x):
        output = self.lstm(x)
        x = x.transpose(2, 1)  # [batch_size x n_vars x seq_len] --> [batch_size x seq_len x n_vars]
        output, _ = self.lstm.rnn(
            x)  # output from all sequence steps: [batch_size x seq_len x hidden_size * (1 + bidirectional)]
        output = output[:, -1]  # output from last sequence step : [batch_size x hidden_size * (1 + bidirectional)]
        output = self.head(output)
        return output


class LSTMEMAEncoder(nn.Module):
    def __init__(self, c_in, c_out, d_model, num_layers):
        super(LSTMEMAEncoder, self).__init__()
        self.name = "LSTMEMAEncoder"
        self.lstm = LSTM(c_in=c_in, c_out=c_out, hidden_size=d_model, n_layers=num_layers, rnn_dropout=0.5,
                         fc_dropout=0.5)
        self.ema1d = EMA_1D(channels=d_model)
        head_layers = [Squeeze(-1), nn.BatchNorm1d(d_model), nn.Linear(d_model, c_out)]
        self.head = nn.Sequential(*head_layers)

    def forward(self, x):
        x = x.transpose(2, 1)  # [batch_size x n_vars x seq_len] --> [batch_size x seq_len x n_vars]
        output, _ = self.lstm.rnn(
            x)  # output from all sequence steps: [batch_size x seq_len x hidden_size * (1 + bidirectional)]

        output = output.transpose(2, 1)  # [batch_size x seq_len x n_vars] --> [batch_size x n_vars x seq_len]
        output = self.ema1d(output)
        output = output.transpose(2, 1)  # [batch_size x n_vars x seq_len] --> [batch_size x seq_len x n_vars]

        output = output[:, -1]  # output from last sequence step : [batch_size x hidden_size * (1 + bidirectional)]
        output = self.lstm.fc(self.lstm.dropout(output))
        return output


class TransEncoder(nn.Module):
    def __init__(self, c_in, c_out, d_model, num_layers):
        super(TransEncoder, self).__init__()
        self.name = "TransformerEncoder"
        self.transformer = TransformerModel(c_in=c_in, c_out=c_out, d_model=d_model, n_layers=num_layers)
        head_layers = [Squeeze(-1), nn.BatchNorm1d(d_model), nn.Linear(d_model, c_out)]
        self.head = nn.Sequential(*head_layers)

    def forward(self, x):
        x = self.transformer.permute(x)  # bs x nvars x seq_len -> seq_len x bs x nvars
        x = self.transformer.inlinear(x)  # seq_len x bs x nvars -> seq_len x bs x d_model
        x = self.transformer.relu(x)
        x = self.transformer.transformer_encoder(x)
        x = self.transformer.transpose(x)  # seq_len x bs x d_model -> bs x seq_len x d_model
        x = self.transformer.max(x)
        output = self.head(x)
        return output


class TransEMAEncoder(nn.Module):
    def __init__(self, c_in, c_out, d_model, num_layers):
        super(TransEMAEncoder, self).__init__()
        self.name = "TransformerEMAEncoder"
        self.transformer = TransformerModel(c_in=c_in, c_out=c_out, d_model=d_model, n_layers=num_layers)
        self.ema1d = EMA_1D(channels=d_model)
        self.permute = Permute(0, 2, 1)
        head_layers = [Squeeze(-1), nn.BatchNorm1d(d_model), nn.Linear(d_model, c_out)]
        self.head = nn.Sequential(*head_layers)

    def forward(self, x):
        x = self.transformer.permute(x)  # bs x nvars x seq_len -> seq_len x bs x nvars
        x = self.transformer.inlinear(x)  # seq_len x bs x nvars -> seq_len x bs x d_model
        x = self.transformer.relu(x)
        x = self.transformer.transformer_encoder(x)
        x = self.transformer.transpose(x)  # seq_len x bs x d_model -> bs x seq_len x d_model

        x = self.permute(x)  # bs x seq_len x d_model -> bs x d_model x seq_len
        x = self.ema1d(x)
        x = self.permute(x)  # bs x d_model x seq_len -> bs x seq_len x d_model

        x = self.transformer.max(x)
        output = self.head(x)
        return output


class EMA_1D_v2(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA_1D_v2, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool1d(1)  # 使用1维平均池化
        self.pool = nn.AdaptiveAvgPool1d(1)  # 使用1维平均池化
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv1d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv1x3 = nn.Conv1d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1,
                                 padding=1)  # 使用1维卷积

    def forward(self, x):
        b, c, l = x.size()  # 注意这里的尺寸变化，l代表长度
        group_x = x.reshape(b * self.groups, -1, l)  # b*g,c//g,l
        # x_pooled = self.pool(group_x)
        # hw = self.conv1x1(x_pooled)
        # x1 = self.gn(group_x * hw.sigmoid())
        x1 = self.conv1x1(group_x)
        x2 = self.conv1x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, l
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, l
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, l)
        return (group_x * weights.sigmoid()).reshape(b, c, l)


class EMA_1D(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA_1D, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool1d(1)  # 使用1维平均池化
        self.pool = nn.AdaptiveAvgPool1d(1)  # 使用1维平均池化
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv1d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv1x3 = nn.Conv1d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)
        self.conv1x5 = nn.Conv1d(channels // self.groups, channels // self.groups, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        b, c, l = x.size()  # 注意这里的尺寸变化，l代表长度
        group_x = x.reshape(b * self.groups, -1, l)  # b*g,c//g,l
        x1 = self.conv1x1(group_x)
        x2 = self.conv1x3(group_x)
        x3 = self.conv1x5(group_x)

        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, l

        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, l

        x31 = self.softmax(self.agp(x3).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x32 = x3.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, l

        weights = (
                torch.matmul(x11, x22) + torch.matmul(x21, x12)
                + torch.matmul(x11, x32) + torch.matmul(x31, x12)
            # + torch.matmul(x21, x32) + torch.matmul(x31, x22)
        ).reshape(b * self.groups, 1, l)
        return (group_x * weights.sigmoid()).reshape(b, c, l)
