"""
AIO -- All Model in One
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform, xavier_normal, orthogonal


from models.multiTask import *
from models.subNets.BertTextEncoderFinetune import *
from models.subNets.VisionEncoderFinetune import *
from models.subNets.AudioEncoderFinetune import *

__all__ = ['AMIO']

MODEL_MAP = {
    'maf': MAF,
    'text': BertTextEncoderPretrain,
    'vision':VisionEncoderPretrain,
    'audio':AudioEncoderPretrain,
}

class AMIO(nn.Module):
    def __init__(self, args):
        super(AMIO, self).__init__()
        lastModel = MODEL_MAP[args.modelName]
        self.Model = lastModel(args)
        self.args = args
    def forward(self,text=None, audio=None, vision=None,mask=None,labels=None):
        if text != None and audio == None and vision == None:
            return self.Model(text,labels)

        if vision != None and audio == None and text == None:
            return self.Model(vision=vision, label=labels, key_padding_mask=mask)
        
        if audio != None and vision == None and text == None:
            return self.Model(audio=audio, label=labels, key_padding_mask=mask)

        if text != None and audio != None and vision != None:
            return self.Model(text,audio,vision)

        



    def save_model(self):
        self.Model.save_model()
        # # save all modules
        # path = self.args.model_save_path + f'{self.args.modelName}-{self.args.datasetName}-{self.args.train_mode}_fusion.pth'
        # print('model saved at:',path)
        # torch.save(path)

    def load_model(self,module):
        self.Model.load_model(module)