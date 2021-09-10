
import cv2
import os
import sys
import logging
import tempfile
from subprocess import call
import threading

import numpy as np
import torch
import torch.nn as nn
from scipy.io import wavfile
from sklearn.manifold import TSNE

from psbody.mesh import Mesh
from utils.rendering import render_mesh_helper
from utils import guo_network


class VOCAModel(nn.Module):
    def __init__(self, config):
        super(VOCAModel, self).__init__()
        self.config = config
        self.speech_encoder = guo_network.SpeechEncoder(self.config)
        self.decoder = guo_network.Decoder(self.config)
        self.recon_loss = nn.L1Loss()
        self.velocity_loss = guo_network.VelocityLoss(config)
        self.acc_loss = guo_network.AccelerationLoss(config)
        self.verts_reg_loss = guo_network.VertsRegLoss(config)

        self.optim_netE = torch.optim.Adam(self.speech_encoder.parameters(), lr=self.config['learning_rate'])
        self.optim_netD = torch.optim.Adam(self.decoder.parameters(), lr=self.config['learning_rate'])
        self.scheduler_E = torch.optim.lr_scheduler.StepLR(self.optim_netE, step_size=10, gamma = 0.5)
        self.scheduler_D = torch.optim.lr_scheduler.StepLR(self.optim_netD, step_size=10, gamma = 0.5)
    def save_model(self):
#        torch.save(self.state_dict(), PATH)
        pass

    def load_model(self):
        pass

