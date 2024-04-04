import os
import sys
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from models.almt.almt_layer import Transformer, CrossTransformer, HhyperLearningEncoder
from einops import repeat

from models.subNets.BaseClassifier import BaseClassifier
from models.subNets.VisionEncoderFinetune import VisionEncoder
from models.subNets.AudioEncoderFinetune import AudioEncoder
from models.subNets.BertTextEncoderFinetune import BertTextEncoder
from models.FusionTransformer.transformer import LayerFusionTransformer
from models.subNets.pooling import MultiheadSelfAttentionWithPooling
from models.loss.bmc_loss import BMCLoss

__all__ = ['MAF']

class MAF(nn.Module):
    def __init__(self, args):
        super(MAF, self).__init__()

        self.scale_a = 1.0
        self.scale_t = 1.0
        self.scale_v = 1.0
        self.epsilon = 1.0
        self.args = args
        self.device = args.device
        # self.h_hyper = nn.Parameter(torch.ones(1, 8, args.post_fusion_dim))
        # text subnets
        # self.aligned = args.need_data_aligned
        # self.text_model = BertTextEncoder(language=args.language, use_finetune=args.use_finetune)

        # audio-vision subnets
        # text_in, audio_in, vision_in = args.feature_dims
        # text_patches, audio_patches, vision_patches = args.seq_lens
        

        # self.proj_t = Transformer(num_frames=50, save_hidden=False, token_len=8, dim=args.post_text_dim, depth=1, heads=8, mlp_dim=128)
        # self.proj_a = Transformer(num_frames=50, save_hidden=False, token_len=8, dim=args.post_audio_dim, depth=1, heads=8, mlp_dim=128)
        # self.proj_v = Transformer(num_frames=50, save_hidden=False, token_len=8, dim=args.post_vision_dim, depth=1, heads=8, mlp_dim=128)


        # self.text_attention = MultiheadSelfAttentionWithPooling(embed_size=args.post_text_dim,num_heads=args.nums_head)
        # self.vision_attention = MultiheadSelfAttentionWithPooling(embed_size=args.post_vision_dim,num_heads=args.nums_head)
        # self.audio_attention = MultiheadSelfAttentionWithPooling(embed_size=args.post_audio_dim,num_heads=args.nums_head)
        # self.fusion_attention= MultiheadSelfAttentionWithPooling(embed_size=args.post_fusion_dim,num_heads=args.nums_head)

        
        # self.text_encoder = Transformer(num_frames=8, save_hidden=True, token_len=None, dim= args.post_text_dim,depth=args.AHL_depth-1, heads=8, mlp_dim=128)
        
        
        # 直接拼接
        # if self.args.is_concat:
        #     args.post_fusion_dim = args.post_fusion_dim * 3
            
        # self.post_fusion_dropout = nn.Dropout(p=args.post_fusion_dropout)
        # self.post_fusion_layer_1 = nn.Linear(args.post_fusion_dim, args.post_fusion_dim)
        # self.post_fusion_layer_2 = nn.Linear(args.post_fusion_dim, 1)


        # the classify layer for text
        # self.post_text_dropout = nn.Dropout(p=args.post_text_dropout)
        # self.post_text_layer_1 = nn.Linear(text_in, args.post_text_dim)
        # self.post_text_layer_2 = nn.Linear(args.post_text_dim, args.post_text_dim)
        # self.post_text_layer_3 = nn.Linear(args.post_text_dim, 1)


        # the classify layer for audio
        # self.post_audio_dropout = nn.Dropout(p=args.post_audio_dropout)
        # self.post_audio_layer_1 = nn.Linear(audio_in, args.post_audio_dim)
        # self.post_audio_layer_2 = nn.Linear(args.post_audio_dim, args.post_audio_dim)
        # self.post_audio_layer_3 = nn.Linear(args.post_audio_dim, 1)

        # the classify layer for vision
        # self.post_vision_dropout = nn.Dropout(p=args.post_vision_dropout)
        # self.post_vision_layer_1 = nn.Linear(vision_in, args.post_vision_dim)
        # self.post_vision_layer_2 = nn.Linear(args.post_vision_dim, args.post_vision_dim)
        # self.post_vision_layer_3 = nn.Linear(args.post_vision_dim, 1)



        self.text_encoder = BertTextEncoder(self.args.language)
        self.vision_encoder = VisionEncoder(
                                    args=self.args,
                                    fea_size=self.args.feature_dims[2],
                                    encoder_fea_dim=self.args.encoder_fea_dim,
                                    nhead=self.args.vision_nhead,
                                    dim_feedforward=self.args.encoder_fea_dim,
                                    num_layers=self.args.vision_tf_num_layers,
                                    drop_out=self.args.post_vision_dropout)
        
        self.audio_encoder = AudioEncoder(
                                    args=self.args,
                                    fea_size=self.args.feature_dims[1],
                                    encoder_fea_dim=self.args.encoder_fea_dim,
                                    nhead=self.args.audio_nhead,
                                    dim_feedforward=self.args.encoder_fea_dim,
                                    num_layers=self.args.audio_tf_num_layers,
                                    drop_out=self.args.post_audio_dropout)
        
        encoder_fea_dim = self.args.encoder_fea_dim
        hidden_size = [encoder_fea_dim,int(encoder_fea_dim/2), int(encoder_fea_dim / 4), int(encoder_fea_dim / 8)]
        self.TVA_decoder = BaseClassifier(input_size=encoder_fea_dim*3,
                                          hidden_size=hidden_size,
                                          output_size=1, drop_out=self.args.post_fusion_dropout )

        self.t_mono_decoder = BaseClassifier(input_size=encoder_fea_dim,
                                           hidden_size=hidden_size[2:],
                                           output_size=1, drop_out=self.args.post_fusion_dropout )
        self.v_mono_decoder = BaseClassifier(input_size=encoder_fea_dim,
                                           hidden_size=hidden_size[2:],
                                           output_size=1, drop_out=self.args.post_fusion_dropout )

        self.a_mono_decoder = BaseClassifier(input_size=encoder_fea_dim,
                                           hidden_size=hidden_size[2:],
                                           output_size=1, drop_out=self.args.post_fusion_dropout )










        self.l2norm = Normalize(2)
        # self.criterion = BMCLoss(1.,self.args.device)
        self.criterion = nn.MSELoss(reduce='none')
        # 特征融合
        if self.args.is_almt:
            self.h_hyper_layer = HhyperLearningEncoder(dim=args.post_fusion_dim, depth=args.AHL_depth, heads=8, dim_head=16, dropout = 0.)
            self.fusion_layer = CrossTransformer(source_num_frames=8, tgt_num_frames=8, dim=args.post_fusion_dim, depth=args.fusion_layer_depth, heads=8, mlp_dim=args.post_fusion_dim)
        
        self.multiheadPooling = MultiheadSelfAttentionWithPooling(embed_size=encoder_fea_dim,num_heads=self.args.fusion_nhead)
        
        # 梯度更新
        if self.args.is_agm:
            self.m_t_o = Modality_out()
            self.m_a_o = Modality_out()
            self.m_v_o = Modality_out()

            self.m_t_o.register_full_backward_hook(self.hookt)
            self.m_a_o.register_full_backward_hook(self.hooka)
            self.m_v_o.register_full_backward_hook(self.hookv)

       
        
    
    def hookt(self,m,ginp,gout):
        gnew = ginp[0].clone()
        return gnew*self.scale_t,

    def hookv(self,m,ginp,gout):
        gnew = ginp[0].clone()
        return gnew * self.scale_v,


    def hooka(self,m,ginp,gout):
        gnew = ginp[0].clone()
        return gnew * self.scale_a,
    

    def update_scale(self,coeff_t,coeff_a,coeff_v):
        self.scale_a = coeff_a
        self.scale_t = coeff_t
        self.scale_v = coeff_v


    def forward(self, text, vision, audio, vision_padding_mask, audio_padding_mask,labels):
        x_t_embed = self.text_encoder(text)[0]
        x_v_embed = self.vision_encoder(vision, vision_padding_mask)
        x_a_embed = self.audio_encoder(audio, audio_padding_mask)
        
        x_t_embed = self.l2norm(x_t_embed)
        x_v_embed = self.l2norm(x_v_embed)
        x_a_embed = self.l2norm(x_a_embed)

        x_t_embed = x_t_embed[:,0,:].unsqueeze(1)
        x_v_embed = self.multiheadPooling(x_v_embed)
        x_a_embed = self.multiheadPooling(x_a_embed)
        

        x_fusion_embed = torch.cat((x_t_embed,x_v_embed,x_a_embed),dim=-1)
        
        if self.args.is_agm:
            x_t_embed = self.m_t_o(x_t_embed)
            x_v_embed = self.m_v_o(x_v_embed)
            x_a_embed = self.m_a_o(x_a_embed)

            dropout_rate = F.softmax(1-torch.tensor([self.scale_t, self.scale_v, self.scale_a]), dim=0)
            
            x_t_embed = F.dropout(x_t_embed, p=dropout_rate[0].item(), training=self.training)
            x_v_embed = F.dropout(x_t_embed, p=dropout_rate[1].item(), training=self.training)
            x_a_embed = F.dropout(x_t_embed, p=dropout_rate[2].item(), training=self.training)


        pred_t = self.t_mono_decoder(x_t_embed).squeeze(-1)
        pred_a = self.a_mono_decoder(x_a_embed).squeeze(-1)
        pred_v = self.v_mono_decoder(x_v_embed).squeeze(-1)
     
        pred_fusion = self.TVA_decoder(x_fusion_embed).squeeze(-1)

        
        
       
        
        # weighted_mono_loss = weighted[0].item()*loss_t + weighted[1].item()*loss_v + weighted[2].item()*loss_a


       
       



        return pred_fusion,pred_t,pred_a,pred_v

        # res = {
        #     'M': output_fusion, 
        #     'T': output_text,
        #     'A': output_audio,
        #     'V': output_vision,
        #     'Feature_t': text_feature,
        #     'Feature_a': audio_feature,
        #     'Feature_v': vision_feature,
        #     'Feature_f': fusion_feature,
        # }




    # def forward(self, text, audio, vision):

        
    #     # print(f"vision.size():{vision.size()}")
    #     # sys.exit(1)
    #     mask_len = torch.sum(text[:,1,:], dim=1, keepdim=True)
    #     text_lengths = mask_len.squeeze().int().detach().cpu()
    #     text = self.text_model(text)
        
    #     b = vision.size(0)
        
        
        
    #     # 线性变化，统一为[batch_size,128]
    #     text_h = F.relu(self.post_text_layer_1(text), inplace=False)
    #     vision_h = F.relu(self.post_vision_layer_1(vision), inplace=False)
    #     audio_h = F.relu(self.post_audio_layer_1(audio), inplace=False)
        
       

    #     proj_t = self.proj_t(text_h)
    #     proj_v = self.proj_v(vision_h)
    #     proj_a = self.proj_a(audio_h)
    #     if self.args.is_almt:
    #         h_hyper = repeat(self.h_hyper, '1 n d -> b n d', b = b)
    #         h_t = proj_t[:, :8]
    #         h_v = proj_v[:,:8]
    #         h_a = proj_a[:,:8]
    #         h_t = F.relu(self.post_audio_layer_2(h_t),inplace=False)
    #         h_v = F.relu(self.post_vision_layer_2(h_v),inplace=False)
    #         h_a = F.relu(self.post_audio_layer_2(h_a),inplace=False)
    #         h_t_list = self.text_encoder(h_t)
    #         h_hyper,attn_tt,attn_ta,attn_tv = self.h_hyper_layer(h_t_list, h_a, h_v, h_hyper)
    #         fusion_h = self.fusion_layer(h_hyper, h_t_list[-1])
    #     else:
    #         fusion_h = torch.cat((proj_t,proj_a[:,],proj_v),dim=-1)
       

    #     fusion_h = self.post_fusion_dropout(fusion_h)

    #     proj_t = self.post_text_dropout(proj_t)
    #     proj_v = self.post_vision_dropout(proj_v)
    #     proj_a = self.post_audio_dropout(proj_a)

       
        




    #     # attn_tt = torch.ones(attn_ta.size(0), attn_ta.size(1), attn_ta.size(2), attn_ta.size(3))
    #     # attn_tt = self.attend(attn_tt)
        
    #     # # 假设 attn_ta 的大小为 [batch_size, num_heads, seq_length, seq_length]
    #     # # 首先对 num_heads 维度进行最大池化
    #     # tt_max_over_heads = attn_tt.max(dim=1, keepdim=True)[0]
    #     # # 然后对 seq_length 维度进行最大池化
    #     # tt_max_over_seq_length = tt_max_over_heads.max(dim=2, keepdim=True)[0]
    #     # # 最后再次对 seq_length 维度进行最大池化
    #     # tt_attention_constant = tt_max_over_seq_length.max(dim=3, keepdim=True)[0]
    #     # tt_attention_constant = tt_attention_constant.squeeze(-1).squeeze(-1)


    #     # # 假设 attn_ta 的大小为 [batch_size, num_heads, seq_length, seq_length]
    #     # # 首先对 num_heads 维度进行最大池化
    #     # ta_max_over_heads = attn_ta.max(dim=1, keepdim=True)[0]
    #     # # 然后对 seq_length 维度进行最大池化
    #     # ta_max_over_seq_length = ta_max_over_heads.max(dim=2, keepdim=True)[0]
    #     # # 最后再次对 seq_length 维度进行最大池化
    #     # ta_attention_constant = ta_max_over_seq_length.max(dim=3, keepdim=True)[0]
    #     # ta_attention_constant = ta_attention_constant.squeeze(-1).squeeze(-1)


       
    #     # # 首先对 num_heads 维度进行最大池化
    #     # tv_max_over_heads = attn_tv.max(dim=1, keepdim=True)[0]
    #     # # 然后对 seq_length 维度进行最大池化
    #     # tv_max_over_seq_length = tv_max_over_heads.max(dim=2, keepdim=True)[0]
    #     # # 最后再次对 seq_length 维度进行最大池化
    #     # tv_attention_constant = tv_max_over_seq_length.max(dim=3, keepdim=True)[0]
    #     # tv_attention_constant = tv_attention_constant.squeeze(-1).squeeze(-1)


        
    #     # attention_constant = torch.cat((tt_attention_constant,ta_attention_constant,tv_attention_constant),dim=1)
        
    #     # attention_constant = self.attend(1-attention_constant)
        
    #     # t_attention_constant=attention_constant[:,0]
    #     # a_attention_constant=attention_constant[:,1]
    #     # v_attention_constant=attention_constant[:,2]
       
    #     # print(f"proj_t:{proj_t.size()},proj_a:{proj_a.size()},proj_v:{proj_v.size()}")




    #     fusion_feature = F.relu(self.post_fusion_layer_1(fusion_h), inplace=False)
    #     # fusion_feature = self.fusion_attention(fusion_h)
    #     fusion_feature = self.post_fusion_layer_1(fusion_feature[:,-1,:])





       
    #     output_fusion = self.post_fusion_layer_2(fusion_feature)
    #     # print(output_fusion)

        
        
       
    #     text_feature = self.post_text_layer_2(proj_t[:,-1,:])
    #     audio_feature = self.post_audio_layer_2(proj_t[:,-1,:])
    #     vision_feature = self.post_vision_layer_2(proj_v[:,-1,:])

    #     if self.args.is_agm:
    #         text_feature = self.m_t_o(text_feature)
    #         audio_feature = self.m_a_o(audio_feature)
    #         vision_feature = self.m_v_o(vision_feature)
            
    #     # text_feature = self.text_attention(proj_t)
    #     # vision_feature = self.vision_attention(proj_v)  
    #     # audio_feature = self.audio_attention(proj_a)
       
    
       
    #     output_text = self.post_text_layer_3(text_feature)
    #     output_audio = self.post_audio_layer_3(audio_feature)
    #     output_vision = self.post_vision_layer_3(vision_feature)


    #     res = {
    #         'M': output_fusion, 
    #         'T': output_text,
    #         'A': output_audio,
    #         'V': output_vision,
    #         'Feature_t': text_feature,
    #         'Feature_a': audio_feature,
    #         'Feature_v': vision_feature,
    #         'Feature_f': fusion_feature,
    #         # 't_attention_constant':t_attention_constant,
    #         # 'a_attention_constant':a_attention_constant,
    #         # 'v_attention_constant':v_attention_constant,
    #     }


        
    #     return res


    def load_model(self,load_pretrain=False):
        if load_pretrain:
            text_encoder_path = self.args.model_save_path + f'{self.args.datasetName}-text-encoder.pth'
            vision_encoder_path = self.args.model_save_path + f'{self.args.datasetName}-vision-encoder.pth'
            audio_encoder_path = self.args.model_save_path + f'{self.args.datasetName}-audio-encoder.pth'
            self.text_encoder.load_state_dict(torch.load(text_encoder_path, map_location=self.device))
            self.vision_encoder.load_state_dict(torch.load(vision_encoder_path, map_location=self.device))
            self.audio_encoder.load_state_dict(torch.load(audio_encoder_path, map_location=self.device))
        else:
            mode_path = self.args.model_save_path + f'{self.args.datasetName}-fusion.pth'
            self.load_state_dict(torch.load(mode_path, map_location=self.device))

    def set_train(self, train_module=None):
        if train_module is None:
            train_module = [False, True, True]
        for param in self.parameters():
            param.requires_grad = train_module[2]

        self.text_encoder.set_train(train_module=train_module[0])
        self.vision_encoder.set_train(train_module=train_module[1])
        self.audio_encoder.set_train(train_module=train_module[1])

    def save_model(self):
        # save all modules
        mode_path = self.args.model_save_path + f'{self.args.datasetName}-fusion.pth'

        print('model saved at:')
        print(mode_path)
        torch.save(self.state_dict(), mode_path)
        

    def align_sequences(self,sequences, max_seq_length=50):
        """
        Align a batch of sequences to the same length by padding or truncating.

        Args:
            sequences (Tensor): A tensor of sequences with shape (batch_size, seq_length, feature_dim).
            max_seq_length (int): The maximum sequence length to align to.

        Returns:
            Tensor: A tensor of aligned sequences with shape (batch_size, max_seq_length, feature_dim).
        """
        batch_size, seq_length, feature_dim = sequences.shape
        # Truncate sequences if they are longer than max_seq_length
        if seq_length > max_seq_length:
            sequences = sequences[:, :max_seq_length, :]

        # Pad sequences with the last element if they are shorter than max_seq_length
        if seq_length < max_seq_length:
            # Extract the last element from each sequence and repeat it
            last_elements = sequences[:, -1, :].unsqueeze(1).repeat(1, max_seq_length - seq_length, 1)
            sequences = torch.cat([sequences, last_elements], dim=1)

        return sequences



class Modality_out(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x



class projector(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.5):
        super(projector, self).__init__()

        self.fc = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            # nn.ReLU(),
            # nn.Linear(output_dim, output_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out
    



    


