import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class LambdaLR:
    def __init__(self, epochs, offset, decay_start_epoch):
        assert (epochs - decay_start_epoch) > 0, "전체 epoch가 decay_start_epoch보다 커야함"

        self.num_epochs = epochs # 설정한 총 epoch
        self.offset = offset # (저장했었던) start epoch
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch): # epoch : 현재 epoch
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.num_epochs - self.decay_start_epoch)
























