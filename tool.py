import os
import torch


def load_model(model, model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    del checkpoint
    torch.cuda.empty_cache()


def load_checkpoint(config, model, optimizers, lr_schedulers, logger):
    logger.info(f">>>>>>>>>> Resuming from {config.MODEL.RESUME} ..........")
    checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if 'lr_schedulers' in checkpoint:
        for scheduler_name, scheduler in lr_schedulers.items():
            if scheduler_name in checkpoint['lr_schedulers']:
                scheduler.load_state_dict(checkpoint['lr_schedulers'][scheduler_name])
    if 'optimizers' in checkpoint and 'epoch' in checkpoint:
        for optimizer_name, optimizer in optimizers.items():
            if optimizer_name in checkpoint['optimizers']:
                optimizer.load_state_dict(checkpoint['optimizers'][optimizer_name])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch']
        config.freeze()
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']
    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy


def save_checkpoint(config, epoch, model, max_accuracy, optimizers, lr_schedulers, logger):
    if lr_schedulers:
        save_state = {'model': model.state_dict(),
                      'optimizers': {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
                      'lr_schedulers': {name: lr_scheduler.state_dict() for name, lr_scheduler in lr_schedulers.items()},
                      'max_accuracy': max_accuracy,
                      'epoch': epoch,
                      'config': config}
    else:
        save_state = {'model': model.state_dict(),
                      'optimizers': {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
                      'max_accuracy': max_accuracy,
                      'epoch': epoch,
                      'config': config}
    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def auto_resume_helper(output_dir, logger):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    logger.info(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        logger.info(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file
