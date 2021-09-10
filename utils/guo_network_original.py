import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import torch.nn.init as init



class _conv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, use_bias=False):
        super(_conv2d, self).__init__()
        self.conv_block = self._build_conv_block(in_channel, out_channel, kernel_size, stride, use_bias)
        init.kaiming_normal_(self.conv_block.weight, mode='fan_out')
        init.constant_(self.conv_block.bias, 0)

    def _build_conv_block(self, in_channel, out_channel, kernel_size, stride, use_bias):
        conv_block = []
        conv_block += [nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding=(0, 1), padding_mode='circular'), nn.ELU(True)]
        

        return nn.Sequential(*conv_block)

    def forward(self, x):
        return self.conv_block(x)

class SpeechEncoder(nn.Module):
    def __init__(self, config):
        super(SpeechEncoder, self).__init__()
        
        self.speech_encoding_dim = config['expression_dim']
        self.condition_speech_features = config['condition_speech_features']
        self.speech_encoder_size_factor = config['speech_encoder_size_factor']

        self.batch_norm = nn.BatchNorm1d(num_features=26, eps=1e-5, momentum=0.9)
#        conv = []
#        conv += [
#            _conv2d(34, 32, (1, 3), (1, 2)),
#            _conv2d(32, 32, (1, 3), (1, 2)),
#            _conv2d(32, 64, (1, 3), (1, 2)),
#            _conv2d(64, 64, (1, 3), (1, 2))
#        ]
#        self.conv_block = nn.Sequential(*conv)
        
        self.conv1 = nn.Conv2d(in_channels=34,out_channels=32,kernel_size=(1,3),stride=(1,2), padding=(0, 1),padding_mode='circular')
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(1,3),stride=(1,2), padding=(0, 1),padding_mode='circular')
        self.conv3 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(1,3),stride=(1,2), padding=(0, 1),padding_mode='circular')
        self.conv4 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(1,3),stride=(1,2), padding=(0, 1),padding_mode='circular')
        
        

#        self.fc1 = nn.Sequential(*FC)
        self.fc1 = nn.Linear(72, 128)
#            nn.Linear(512, 128), nn.ELU()])
        self.fc2 = nn.Linear(128, self.speech_encoding_dim)
        


        self._initialize_weights()	

    def forward(self, speech_feature, condition, reuse=False):
        speech_feature = speech_feature.permute(0, 2, 1)
        speech_feature = self.batch_norm(speech_feature)
        speech_feature_reshaped = torch.unsqueeze(speech_feature, -1)
        speech_feature_reshaped = speech_feature_reshaped.permute(0, 1, 3, 2)

        condition_reshaped = condition.unsqueeze(-1).unsqueeze(-1)
        condition_reshaped = condition_reshaped.permute(0, 1, 3, 2)
        condition_reshaped = condition_reshaped.repeat(1, 1, 1, 16)
        feat = torch.cat((speech_feature_reshaped, condition_reshaped), 1)
        feat = F.elu(self.conv1(feat))
        feat = F.elu(self.conv2(feat))
        feat = F.elu(self.conv3(feat))
        feat = F.elu(self.conv4(feat))
        
        feat = feat.squeeze().squeeze()
        feat = torch.cat((feat, condition), 1)
        feat = F.elu(self.fc1(feat))
        feat = self.fc2(feat)
        return feat

    def _initialize_weights(self):
        init.kaiming_normal_(self.conv1.weight, mode='fan_out')
        init.constant_(self.conv1.bias, 0)

        init.kaiming_normal_(self.conv2.weight, mode='fan_out')
        init.constant_(self.conv2.bias, 0)

        init.kaiming_normal_(self.conv3.weight, mode='fan_out')
        init.constant_(self.conv3.bias, 0)

        init.kaiming_normal_(self.conv4.weight, mode='fan_out')
        init.constant_(self.conv4.bias, 0)

        init.kaiming_normal_(self.fc1.weight, mode='fan_out')
        init.constant_(self.fc1.bias, 0)

        init.kaiming_normal_(self.fc2.weight, mode='fan_out')
        init.constant_(self.fc2.bias, 0)


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.expression_basis_fname = config['expression_basis_fname']
        self.init_expression = config['init_expression']

        self.num_vertices = config['num_vertices']
        self.expression_dim = config['expression_dim']
        
        init_exp_basis = np.zeros((3*self.num_vertices, self.expression_dim))
        init_exp_basis[:, :min(self.expression_dim, 100)] = np.load(self.expression_basis_fname)[:, :min(self.expression_dim, 100)]
        decoder = []
        decoder += [nn.Linear(self.expression_dim, 3*self.num_vertices)]
        self.decoder = nn.Sequential(*decoder)
        if self.init_expression:
#            with torch.no_grad():
#                self.decoder.weight = nn.Parameter(torch.from_numpy(init_exp_basis.T).float()
#            pass
            self.decoder.data = torch.FloatTensor(init_exp_basis.T)

    def forward(self, x):
        x = self.decoder(x)
        expression_offset = x.reshape((-1, self.num_vertices, 3))
        return expression_offset


class VelocityLoss(nn.Module):
    def __init__(self, config):
        super(VelocityLoss, self).__init__()
        self.config = config

    def forward(self, predict, real):
        if self.config['velocity_weight'] > 0:
            assert (self.config['num_consecutive_frames'] >= 2)
            verts_predicted = torch.reshape(predict, (-1, self.config['num_consecutive_frames'],
                                                      self.config['num_vertices'], 3))
            x1_pred = torch.reshape(verts_predicted[:, -1, :], (-1, self.config['num_vertices'], 3, 1))
            x2_pred = torch.reshape(verts_predicted[:, -2, :], (-1, self.config['num_vertices'], 3, 1))
            velocity_pred = x1_pred - x2_pred

            verts_real = torch.reshape(real, (-1, self.config['num_consecutive_frames'],
                                                      self.config['num_vertices'], 3))
            x1_real = torch.reshape(verts_real[:, -1, :], (-1, self.config['num_vertices'], 3, 1))
            x2_real = torch.reshape(verts_real[:, -2, :], (-1, self.config['num_vertices'], 3, 1))
            velocity_real = x1_real - x2_real

            velocity_loss = self.config['velocity_weight']*F.l1_loss(velocity_pred, velocity_real)
            return velocity_loss
        else:
            return 0.0

class AccelerationLoss(nn.Module):
    def __init__(self, config):
        super(AccelerationLoss, self).__init__()
        self.config = config

    def forward(self, predict, real):
        if self.config['acceleration_weight'] > 0.0:
            assert (self.config['num_consecutive_frames'] >= 3)
            verts_predicted = torch.reshape(predict, (-1, self.config['num_consecutive_frames'],
                                                      self.config['num_vertices'], 3))
            x1_pred = torch.reshape(verts_predicted[:, -1, :], (-1, self.config['num_vertices'], 3, 1))
            x2_pred = torch.reshape(verts_predicted[:, -2, :], (-1, self.config['num_vertices'], 3, 1))
            x3_pred = torch.reshape(verts_predicted[:, -3, :], (-1, self.config['num_vertices'], 3, 1))
            acc_pred = x1_pred-2*x2_pred+x3_pred

            verts_real = torch.reshape(real, (-1, self.config['num_consecutive_frames'],
                                                      self.config['num_vertices'], 3))
            x1_real = torch.reshape(verts_real[:, -1, :], (-1, self.config['num_vertices'], 3, 1))
            x2_real = torch.reshape(verts_real[:, -2, :], (-1, self.config['num_vertices'], 3, 1))
            x3_real = torch.reshape(verts_real[:, -3, :], (-1, self.config['num_vertices'], 3, 1))
            acc_real = x1_real - 2 * x2_real + x3_real

            acc_loss = self.config['acceleration_weight']*F.l1_loss(acc_pred, acc_real)
            return acc_loss
        else:
            return 0.0

class VertsRegLoss(nn.Module):
    def __init__(self, config):
        super(VertsRegLoss, self).__init__()
        self.config = config

    def forward(self, expression_offset):
        if self.config['verts_regularizer_weight'] > 0.0:
            verts_reg_loss = self.config['verts_regularizer_weight']*torch.mean(torch.sum(
                torch.abs(expression_offset), dim=2))
            return verts_reg_loss
        else:
            return 0.0
