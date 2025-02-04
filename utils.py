import torchvision
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.manifold import TSNE

def optimizer_scheduler(optimizer, p):
    """
    Adjust the learning rate of optimizer 
    :param optimizer: optimizer for updating parameters 
    :param p: a variable for adjusting learning rate     
    :return: optimizer
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.01 / (1. + 10 * p) ** 0.75 

    return optimizer
