import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np 
from collections import OrderedDict

class Classifier(nn.Module):

    def __init__(self,embed_size,num_class,dropout):
        super(Classifier,self).__init__()
        self.num_class = num_class
        print(num_class)

        self.dr = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_size, num_class)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(self.dr(images))
        return features

    def load_state_dict(self, state_dict):
        """copies parameters. overwritting the default one to accept state_dict from Full model"""
        own_state = self.state_dict()
        new_state = OrderedDict()

        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(Classifier, self).load_state_dict(new_state)

class VSE(object):

    def __init__(self, opt):
        #build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = Classifier(opt.embed_size, 101, opt.dropout)

        if torch.cuda.is_available():
            self.img_enc.cuda()
            cudnn.benchmark = True

        #loss and Optimizer
        self.criterion = nn.CrossEntropyLoss()
        params = list(self.img_enc.parameters())
        self.params = params

        self.optimizer = torch.optim.Adam(params,lr=opt.learning_rate)

        self.Eiters = 0

    def state_dict(self):
        state_dict = self.img_enc.state_dict()
        return state_dict

    def load_state_dict(self,state_dict):
        self.img_enc.load_state_dict(state_dict)

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()

    def forward_emb(self,images):
        #forward
        img_emb = self.img_enc(images)
        return img_emb

    def forward_loss(self, img_emb, label, **kwargs):
        loss = self.criterion(img_emb, label)
        self.logger.update('Le', loss.data[0], img_emb.size(0))
        return loss

    def train_emb(self, images, label, *args):
        self.Eiters += 1
        if torch.cuda.is_available():
            images = images.cuda()
            label = label.cuda()

        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        #compute the embeddings
        img_emb = self.forward_emb(images)

        #measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, label)

        #compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            #梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()

        










    

    

