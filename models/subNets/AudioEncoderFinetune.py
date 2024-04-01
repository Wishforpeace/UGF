import torch
import torch.nn as nn
import torch.nn.functional as F
from models.subNets.BaseClassifier import BaseClassifier
from models.subNets.TfEncoder import  TfEncoder

import sys

from models.subNets.pooling import MultiheadSelfAttentionWithPooling

class AudioEncoder(nn.Module):
    def __init__(self,args,fea_size=None, encoder_fea_dim=None, nhead=None, dim_feedforward=None,
                 num_layers=None,
                 drop_out=0.5):
        super(AudioEncoder, self).__init__()
        self.args = args
        self.fc = nn.Linear(fea_size, encoder_fea_dim)
        self.encoder = TfEncoder(args=args,
                                fea_size=fea_size,
                                d_model=encoder_fea_dim,
                                nhead=nhead,
                                seq_lens=args.seq_lens[1], 
                                dim_feedforward=dim_feedforward,
                                num_layers=num_layers,
                                dropout=drop_out,
                                activation='gelu')
        self.device = self.args.device
        self.encoder.device = self.device
        self.activation = nn.Tanh()
        self.cls_embedding = nn.Parameter()
        self.layernorm = nn.LayerNorm(encoder_fea_dim)
        self.dense = nn.Linear(encoder_fea_dim, encoder_fea_dim)

    def forward(self, audio, key_padding_mask):
        
        x = self.encoder(src=audio, has_mask=False, src_key_padding_mask=key_padding_mask)
        x = self.layernorm(x)
        # x = torch.mean(x, dim=-2, keepdim=True)
        
       # 考虑多头注意力池化
        return x

    def set_train(self, train_module=None):
        if train_module is None:
            train_module = True
        for param in self.parameters():
            param.requires_grad = train_module



class AudioEncoderPretrain(nn.Module):
    def __init__(self,args):
        super(AudioEncoderPretrain, self).__init__()
        self.args = args
        self.device = args.device
        encoder_fea_dim = self.args.encoder_fea_dim
        
        drop_out = self.args.post_audio_dropout
       
        self.encoder = AudioEncoder(args=self.args,
                                    fea_size=self.args.feature_dims[1],
                                    encoder_fea_dim=self.args.encoder_fea_dim,
                                    nhead=self.args.audio_nhead,
                                    dim_feedforward=self.args.encoder_fea_dim,
                                    num_layers=self.args.audio_tf_num_layers,
                                    drop_out=self.args.post_audio_dropout)
        

        self.classifier = BaseClassifier(input_size=encoder_fea_dim,
                                         hidden_size=[int(encoder_fea_dim / 2), int(encoder_fea_dim / 4),
                                                      int(encoder_fea_dim / 8)],
                                         output_size=1, drop_out=drop_out)
        self.multiheadPooling = MultiheadSelfAttentionWithPooling(embed_size=encoder_fea_dim,num_heads=self.args.audio_nhead)
        
        self.criterion = torch.nn.MSELoss(reduction='none')

    def forward(self, audio, label, key_padding_mask):
        x = self.encoder(audio, key_padding_mask)
        x = self.multiheadPooling(x)

        pred = self.classifier(x).squeeze()
        loss = self.criterion(pred.squeeze(), label.squeeze())
        return pred, x, loss
       
    def save_model(self):
        # save all modules
        path = self.args.model_save_path + f'{self.args.datasetName}-audio'
        encoder_path = path + '-encoder.pth'
        decoder_path = path + '-decoder.pth'
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.classifier.state_dict(), decoder_path)
        print('model saved at:')

    def load_model(self, module=None):
        path = self.args.model_save_path + f'{self.args.datasetName}-audio'
        encoder_path =  path + '-encoder.pth'
        decoder_path =  path+ '-decoder.pth'

        print('model loaded from:')

        if module == 'encoder':
            self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
            print(encoder_path)
        if module == 'decoder':
            self.classifier.load_state_dict(torch.load(decoder_path, map_location=self.device))
            print(decoder_path)
        if module == 'all' or module is None:
            self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
            self.classifier.load_state_dict(torch.load(decoder_path, map_location=self.device))
            print(encoder_path)
            print(decoder_path)

    def set_train(self, train_module=None):
        if train_module is None:
            train_module = [True, True]
        self.encoder.set_train(train_module=train_module[0])
        for param in self.classifier.parameters():
            param.requires_grad = train_module[-1]

