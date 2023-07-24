# CACIOUS CODING
# Data     : 6/1/23  7:59 PM
# File name: reid_training_model
# Desc     : model for training, add fc

import torch.nn as nn


class TrainingModel(nn.Module):
    def __init__(self, model, opt):
        super(TrainingModel, self).__init__()
        self.model = model
        self.fc = nn.Linear(opt.reid_dim, opt.IDn)

    def forward(self, x):
        x = self.model(x)[0]['id']
        x_center = x[..., x.shape[2] // 2, x.shape[3] // 2]
        return self.fc(x_center)
