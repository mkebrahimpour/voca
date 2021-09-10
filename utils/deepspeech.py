import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.nn.init as init
from torchvision.models import resnet18
from torchvision.models import resnet34
from torchvision.models import resnet50
from torchvision.models import resnet101
import torch.utils.model_zoo as model_zoo
from torchaudio.models import DeepSpeech



class FullyConnected(torch.nn.Module):
    """
    Args:
        n_feature: Number of input features
        n_hidden: Internal hidden unit size.
    """

    def __init__(self,
                 n_feature: int,
                 n_hidden: int,
                 dropout: float,
                 relu_max_clip: int = 20) -> None:
        super(FullyConnected, self).__init__()
        self.fc = torch.nn.Linear(n_feature, n_hidden, bias=True)
        self.relu_max_clip = relu_max_clip
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.hardtanh(x, 0, self.relu_max_clip)
        if self.dropout:
            x = torch.nn.functional.dropout(x, self.dropout, self.training)
        return x


class deepspeech(torch.nn.Module):
    """
    DeepSpeech model architecture from :footcite:`hannun2014deep`.

    Args:
        n_feature: Number of input features
        n_hidden: Internal hidden unit size.
        n_class: Number of output classes
    """

    def __init__(
        self,
        n_feature: int,
        n_hidden: int = 2048,
        n_class: int = 29,
        dropout: float = 0.0,
        embedding_size: float = 512,
        norm = True,
        pretrained = True,
    ) -> None:
        super(deepspeech, self).__init__()

        self.model = DeepSpeech(pretrained)
#        self.model.out = torch.nn.Linear(n_hidden, n_class)
#        self._initialize_weights()

        for child in self.model.children():
            for param in child.parameters():
                param.requires_grad = False
        self.n_hidden = n_hidden
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Tensor of dimension (batch, channel, time, feature).
        Returns:
            Tensor: Predictor tensor of dimension (batch, time, class).
        """
        # N x C x T x F
        x = self.model.fc1(x)
        # N x C x T x H
        x = self.model.fc2(x)
        # N x C x T x H
        x = self.model.fc3(x)
        # N x C x T x H
        x = x.squeeze(1)
        # N x T x H
        x = x.transpose(0, 1)
        # T x N x H
        x, _ = self.model.bi_rnn(x)
        # The fifth (non-recurrent) layer takes both the forward and backward units as inputs
        x = x[:, :, :self.n_hidden] + x[:, :, self.n_hidden:]
        # T x N x H
        x = self.model.fc4(x)
        # T x N x H        
        x = self.model.out(x)
        # T x N x n_class
        x = x.permute(1, 0, 2)
        # N x T x n_class
        x = torch.nn.functional.log_softmax(x, dim=2)
        # N x T x n_class
        return x

    def _initialize_weights(self):
        init.kaiming_normal_(self.model.out.weight, mode='fan_out')
        init.constant_(self.model.out.bias, 0)
    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output
path = '/home/mebrahimpour/projects/speech_command/model/speechcommand_M5_best.pth'

class M(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)


    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)

class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32, embedding_size=512, pretrained=True, is_norm=True, bn_freeze = True):

        super().__init__()
        model = M()
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        self.model = model
        print('model loaded inside m5')
        self.embedding_size = embedding_size
        self.is_norm = is_norm

#        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride) 
#        self.bn1 = nn.BatchNorm1d(n_channel)
#        self.pool1 = nn.MaxPool1d(4)
#        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
#        self.bn2 = nn.BatchNorm1d(n_channel)
#        self.pool2 = nn.MaxPool1d(4)
#        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
#        self.bn3 = nn.BatchNorm1d(2 * n_channel)
#        self.pool3 = nn.MaxPool1d(4)
#        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
#        self.bn4 = nn.BatchNorm1d(2 * n_channel)
#        self.pool4 = nn.MaxPool1d(4)
#        
#        self.fc1 = nn.Linear(2 * n_channel, n_output)
        self.num_ftrs = self.model.fc1.out_features

#        self.model.fc = nn.Linear(self.num_ftrs, self.embedding_size)
        self.model.embedding = nn.Linear(self.num_ftrs, self.embedding_size)
        self._initialize_weights()
   

        if bn_freeze:
            for m in self.model.modules():
                print(m)
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)


    
    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output

    def forward(self, x):
        
        x = self.model.conv1(x)
        x = F.relu(self.model.bn1(x))
        x = self.model.pool1(x)
        x = self.model.conv2(x)
        x = F.relu(self.model.bn2(x))
        x = self.model.pool2(x)
        x = self.model.conv3(x)
        x = F.relu(self.model.bn3(x))
        x = self.model.pool3(x)
        x = self.model.conv4(x)
        x = F.relu(self.model.bn4(x))
        x = self.model.pool4(x)
        x = F.avg_pool1d(x,x.shape[-1])
        x = x.permute(0, 2, 1)
        
        x = self.model.fc1(x)
        x = x.squeeze()
#        x = F.relu(self.model.fc(x))
        x_embed = self.model.embedding(x)
        x_embed = x_embed.squeeze()
        
        if self.is_norm:
            x_embed = self.l2_norm(x_embed)
        return F.log_softmax(x, dim=1),x_embed

    def _initialize_weights(self):
        init.kaiming_normal_(self.model.embedding.weight, mode='fan_out')
        init.constant_(self.model.embedding.bias, 0)
        
#        init.kaiming_normal_(self.model.fc.weight, mode = 'fan_out')
#        init.constant_(self.model.fc.bias, 0)


class Resnet18(nn.Module):
    def __init__(self,embedding_size, pretrained=True, is_norm=True, bn_freeze = True):
        super(Resnet18, self).__init__()

        self.model = resnet18(pretrained)
        self.is_norm = is_norm
        self.embedding_size = embedding_size
        self.num_ftrs = self.model.fc.in_features
        self.model.gap = nn.AdaptiveAvgPool2d(1)
        self.model.gmp = nn.AdaptiveMaxPool2d(1)

        self.model.embedding = nn.Linear(self.num_ftrs, self.embedding_size)
        self._initialize_weights()

        if bn_freeze:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        avg_x = self.model.gap(x)
        max_x = self.model.gmp(x)

        x = max_x + avg_x
        
        x = x.view(x.size(0), -1)
        x = self.model.embedding(x)

        if self.is_norm:
            x = self.l2_norm(x)
            
        return x

    def _initialize_weights(self):
        init.kaiming_normal_(self.model.embedding.weight, mode='fan_out')
        init.constant_(self.model.embedding.bias, 0)

class Resnet34(nn.Module):
    def __init__(self,embedding_size, pretrained=True, is_norm=True, bn_freeze = True):
        super(Resnet34, self).__init__()

        self.model = resnet34(pretrained)
        self.is_norm = is_norm
        self.embedding_size = embedding_size
        self.num_ftrs = self.model.fc.in_features
        self.model.gap = nn.AdaptiveAvgPool2d(1)
        self.model.gmp = nn.AdaptiveMaxPool2d(1)

        self.model.embedding = nn.Linear(self.num_ftrs, self.embedding_size)
        self._initialize_weights()

        if bn_freeze:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        avg_x = self.model.gap(x)
        max_x = self.model.gmp(x)

        x = avg_x + max_x
        
        x = x.view(x.size(0), -1)
        x = self.model.embedding(x)

        if self.is_norm:
            x = self.l2_norm(x)

        return x

    def _initialize_weights(self):
        init.kaiming_normal_(self.model.embedding.weight, mode='fan_out')
        init.constant_(self.model.embedding.bias, 0)

class Resnet50(nn.Module):
    def __init__(self,embedding_size, pretrained=True, is_norm=True, bn_freeze = True):
        super(Resnet50, self).__init__()

        self.model = resnet50(pretrained)
        self.is_norm = is_norm
        self.embedding_size = embedding_size
        self.num_ftrs = self.model.fc.in_features
        self.model.gap = nn.AdaptiveAvgPool2d(1)
        self.model.gmp = nn.AdaptiveMaxPool2d(1)

        self.model.embedding = nn.Linear(self.num_ftrs, self.embedding_size)
        self._initialize_weights()

        if bn_freeze:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        avg_x = self.model.gap(x)
        max_x = self.model.gmp(x)

        x = max_x + avg_x
        x = x.view(x.size(0), -1)
        x = self.model.embedding(x)
        
        if self.is_norm:
            x = self.l2_norm(x)
        
        return x

    def _initialize_weights(self):
        init.kaiming_normal_(self.model.embedding.weight, mode='fan_out')
        init.constant_(self.model.embedding.bias, 0)

class Resnet101(nn.Module):
    def __init__(self,embedding_size, pretrained=True, is_norm=True, bn_freeze = True):
        super(Resnet101, self).__init__()

        self.model = resnet101(pretrained)
        self.is_norm = is_norm
        self.embedding_size = embedding_size
        self.num_ftrs = self.model.fc.in_features
        self.model.gap = nn.AdaptiveAvgPool2d(1)
        self.model.gmp = nn.AdaptiveMaxPool2d(1)

        self.model.embedding = nn.Linear(self.num_ftrs, self.embedding_size)
        self._initialize_weights()

        if bn_freeze:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        avg_x = self.model.gap(x)
        max_x = self.model.gmp(x)

        x = max_x + avg_x
        x = x.view(x.size(0), -1)
        x = self.model.embedding(x)
        
        if self.is_norm:
            x = self.l2_norm(x)
            
        return x

    def _initialize_weights(self):
        init.kaiming_normal_(self.model.embedding.weight, mode='fan_out')
        init.constant_(self.model.embedding.bias, 0)
