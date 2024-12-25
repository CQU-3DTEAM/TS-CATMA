import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# Model Config
_C.MODEL = CN()
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Model name
_C.MODEL.NAME = 'pretrain'
_C.MODEL.data_window_len = 360
_C.MODEL.max_seq_len = 360

# Label Classifier Config
_C.MODEL.LABEL_CLS = CN()
_C.MODEL.LABEL_CLS.input_dim = 22
_C.MODEL.LABEL_CLS.seq_length = 360
_C.MODEL.LABEL_CLS.num_classes = 2
_C.MODEL.LABEL_CLS.hidden_dim = 128

# Domain Classifier Config
_C.MODEL.DOMAIN_CLS = CN()
_C.MODEL.DOMAIN_CLS.input_dim = 22
_C.MODEL.DOMAIN_CLS.seq_length = 360
_C.MODEL.DOMAIN_CLS.num_classes = 2
_C.MODEL.DOMAIN_CLS.hidden_dim = 128
_C.MODEL.DOMAIN_CLS.gamma = 10

# Encoder
_C.MODEL.ENCODER = CN()
_C.MODEL.ENCODER.model = "transformer"
_C.MODEL.ENCODER.d_model = 64
_C.MODEL.ENCODER.num_heads = 8
_C.MODEL.ENCODER.num_layers = 3
_C.MODEL.ENCODER.dim_feedforward = 256
_C.MODEL.ENCODER.dropout = 0.1
_C.MODEL.ENCODER.pos_encoding = "fixed"
_C.MODEL.ENCODER.activation = "relu"
_C.MODEL.ENCODER.normalization_layer = "BatchNorm"
_C.MODEL.ENCODER.feat_dim = 22
_C.MODEL.ENCODER.out_dim = 128
_C.MODEL.ENCODER.freeze = False

# Decoder
_C.MODEL.DECODER = CN()
_C.MODEL.DECODER.model = "transformer"
_C.MODEL.DECODER.d_model = 64
_C.MODEL.DECODER.num_heads = 8
_C.MODEL.DECODER.num_layers = 3
_C.MODEL.DECODER.dim_feedforward = 256
_C.MODEL.DECODER.dropout = 0.1
_C.MODEL.DECODER.pos_encoding = "fixed"
_C.MODEL.DECODER.activation = "relu"
_C.MODEL.DECODER.normalization_layer = "BatchNorm"
_C.MODEL.DECODER.feat_dim = 22
_C.MODEL.DECODER.freeze = False

# Training settings
_C.TRAIN = CN()
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 200
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 3e-4
_C.TRAIN.WARMUP_LR = 2.5e-7
_C.TRAIN.MIN_LR = 2.5e-6
_C.TRAIN.LOOP = 1
_C.TRAIN.NO_PSEUDO_EPOCH = 100
_C.TRAIN.PSEUDO_THRESHOLDS = [0.9, 0.9]
_C.TRAIN.PSEUDO_FREQ = 10
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 3.0

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
# Gamma / Multi steps value, used in MultiStepLRScheduler
_C.TRAIN.LR_SCHEDULER.GAMMA = 0.1
_C.TRAIN.LR_SCHEDULER.MULTISTEPS = []

# Frequency to save checkpoint
_C.SAVE_FREQ = 10
# Frequency to logging info
_C.PRINT_FREQ = 10
# Frequency to test model
_C.TEST_FREQ = 1
# Path to pre-trained model
_C.PRETRAINED = ''
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    def _check_args(name):
        if hasattr(args, name) and eval(f'args.{name}'):
            return True
        return False

    # merge from specific arguments
    if _check_args('batch_size'):
        config.DATA.BATCH_SIZE = args.batch_size
    if _check_args('data_path'):
        config.DATA.DATA_PATH = args.data_path
    if _check_args('resume'):
        config.MODEL.RESUME = args.resume
    if _check_args('pretrained'):
        config.PRETRAINED = args.pretrained
    if _check_args('accumulation_steps'):
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if _check_args('use_checkpoint'):
        config.TRAIN.USE_CHECKPOINT = True
    if _check_args('amp_opt_level'):
        config.AMP_OPT_LEVEL = args.amp_opt_level
    if _check_args('output'):
        config.OUTPUT = args.output
    if _check_args('tag'):
        config.TAG = args.tag
    if _check_args('eval'):
        config.EVAL_MODE = True
    if _check_args('throughput'):
        config.THROUGHPUT_MODE = True

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config


def get_custom_config(cfg):
    config = _C.clone()
    _update_config_from_file(config, cfg)
    return config
