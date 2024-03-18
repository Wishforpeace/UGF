import os
import sys
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from models.subNets.BertTextEncoder import BertTextEncoder
from models.almt.almt_layer import Transformer, CrossTransformer, HhyperLearningEncoder
from einops import repeat

__all__ = ['MAF']

class MAF(nn.Module):
    def __init__(self, args):
        super(MAF, self).__init__()

        self.scale_a = 1.0
        self.scale_t = 1.0
        self.scale_v = 1.0

        self.args = args
        self.device = args.device
        self.h_hyper = nn.Parameter(torch.ones(1, 8, args.post_fusion_dim))
        # text subnets
        self.aligned = args.need_data_aligned
        self.text_model = BertTextEncoder(language=args.language, use_finetune=args.use_finetune)

        # audio-vision subnets
        text_in,audio_in, video_in = args.feature_dims
        

        self.proj_t = Transformer(num_frames=50, save_hidden=False, token_len=8, dim=args.post_text_dim, depth=1, heads=8, mlp_dim=128)
        self.proj_a = Transformer(num_frames=50, save_hidden=False, token_len=8, dim=args.post_audio_dim, depth=1, heads=8, mlp_dim=128)
        self.proj_v = Transformer(num_frames=50, save_hidden=False, token_len=8, dim=args.post_video_dim, depth=1, heads=8, mlp_dim=128)


        # self.text_attention = MultiheadSelfAttentionWithPooling(embed_size=args.post_text_dim,num_heads=args.nums_head)
        # self.video_attention = MultiheadSelfAttentionWithPooling(embed_size=args.post_video_dim,num_heads=args.nums_head)
        # self.audio_attention = MultiheadSelfAttentionWithPooling(embed_size=args.post_audio_dim,num_heads=args.nums_head)
        # self.fusion_attention= MultiheadSelfAttentionWithPooling(embed_size=args.post_fusion_dim,num_heads=args.nums_head)

        
        self.text_encoder = Transformer(num_frames=8, save_hidden=True, token_len=None, dim= args.post_text_dim,depth=args.AHL_depth-1, heads=8, mlp_dim=128)
        
        
        # 直接拼接
        if self.args.is_concat:
            args.post_fusion_dim = args.post_fusion_dim * 3
            
        self.post_fusion_dropout = nn.Dropout(p=args.post_fusion_dropout)
        self.post_fusion_layer_1 = nn.Linear(args.post_fusion_dim, args.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(args.post_fusion_dim, 1)


        # the classify layer for text
        self.post_text_dropout = nn.Dropout(p=args.post_text_dropout)
        self.post_text_layer_1 = nn.Linear(text_in, args.post_text_dim)
        self.post_text_layer_2 = nn.Linear(args.post_text_dim, args.post_text_dim)
        self.post_text_layer_3 = nn.Linear(args.post_text_dim, 1)


        # the classify layer for audio
        self.post_audio_dropout = nn.Dropout(p=args.post_audio_dropout)
        self.post_audio_layer_1 = nn.Linear(audio_in, args.post_audio_dim)
        self.post_audio_layer_2 = nn.Linear(args.post_audio_dim, args.post_audio_dim)
        self.post_audio_layer_3 = nn.Linear(args.post_audio_dim, 1)

        # the classify layer for video
        self.post_video_dropout = nn.Dropout(p=args.post_video_dropout)
        self.post_video_layer_1 = nn.Linear(video_in, args.post_video_dim)
        self.post_video_layer_2 = nn.Linear(args.post_video_dim, args.post_video_dim)
        self.post_video_layer_3 = nn.Linear(args.post_video_dim, 1)

        # 特征融合
        if self.args.is_almt:
            self.h_hyper_layer = HhyperLearningEncoder(dim=args.post_fusion_dim, depth=args.AHL_depth, heads=8, dim_head=16, dropout = 0.)
            self.fusion_layer = CrossTransformer(source_num_frames=8, tgt_num_frames=8, dim=args.post_fusion_dim, depth=args.fusion_layer_depth, heads=8, mlp_dim=args.post_fusion_dim)
            
       
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




    def forward(self, text, audio, video):
        audio, audio_lengths = audio
        video, video_lengths = video

        
        # print(f"video.size():{video.size()}")
        # sys.exit(1)
        mask_len = torch.sum(text[:,1,:], dim=1, keepdim=True)
        text_lengths = mask_len.squeeze().int().detach().cpu()
        text = self.text_model(text)
        
        b = video.size(0)
        
        
        
        # 线性变化，统一为[batch_size,128]
        text_h = F.relu(self.post_text_layer_1(text), inplace=False)
        video_h = F.relu(self.post_video_layer_1(video), inplace=False)
        audio_h = F.relu(self.post_audio_layer_1(audio), inplace=False)
        
       

        proj_t = self.proj_t(text_h)
        proj_v = self.proj_v(video_h)
        proj_a = self.proj_a(audio_h)
        if self.args.is_almt:
            h_hyper = repeat(self.h_hyper, '1 n d -> b n d', b = b)
            h_t = proj_t[:, :8]
            h_v = proj_v[:,:8]
            h_a = proj_a[:,:8]
            h_t = F.relu(self.post_audio_layer_2(h_t),inplace=False)
            h_v = F.relu(self.post_video_layer_2(h_v),inplace=False)
            h_a = F.relu(self.post_audio_layer_2(h_a),inplace=False)
            h_t_list = self.text_encoder(h_t)
            h_hyper,attn_tt,attn_ta,attn_tv = self.h_hyper_layer(h_t_list, h_a, h_v, h_hyper)
            fusion_h = self.fusion_layer(h_hyper, h_t_list[-1])
                
        else:
            fusion_h = torch.cat((proj_t,proj_a[:,],proj_v),dim=-1)
       

        fusion_h = self.post_fusion_dropout(fusion_h)

        proj_t = self.post_text_dropout(proj_t)
        proj_v = self.post_video_dropout(proj_v)
        proj_a = self.post_audio_dropout(proj_a)

       
        




        # attn_tt = torch.ones(attn_ta.size(0), attn_ta.size(1), attn_ta.size(2), attn_ta.size(3))
        # attn_tt = self.attend(attn_tt)
        
        # # 假设 attn_ta 的大小为 [batch_size, num_heads, seq_length, seq_length]
        # # 首先对 num_heads 维度进行最大池化
        # tt_max_over_heads = attn_tt.max(dim=1, keepdim=True)[0]
        # # 然后对 seq_length 维度进行最大池化
        # tt_max_over_seq_length = tt_max_over_heads.max(dim=2, keepdim=True)[0]
        # # 最后再次对 seq_length 维度进行最大池化
        # tt_attention_constant = tt_max_over_seq_length.max(dim=3, keepdim=True)[0]
        # tt_attention_constant = tt_attention_constant.squeeze(-1).squeeze(-1)


        # # 假设 attn_ta 的大小为 [batch_size, num_heads, seq_length, seq_length]
        # # 首先对 num_heads 维度进行最大池化
        # ta_max_over_heads = attn_ta.max(dim=1, keepdim=True)[0]
        # # 然后对 seq_length 维度进行最大池化
        # ta_max_over_seq_length = ta_max_over_heads.max(dim=2, keepdim=True)[0]
        # # 最后再次对 seq_length 维度进行最大池化
        # ta_attention_constant = ta_max_over_seq_length.max(dim=3, keepdim=True)[0]
        # ta_attention_constant = ta_attention_constant.squeeze(-1).squeeze(-1)


       
        # # 首先对 num_heads 维度进行最大池化
        # tv_max_over_heads = attn_tv.max(dim=1, keepdim=True)[0]
        # # 然后对 seq_length 维度进行最大池化
        # tv_max_over_seq_length = tv_max_over_heads.max(dim=2, keepdim=True)[0]
        # # 最后再次对 seq_length 维度进行最大池化
        # tv_attention_constant = tv_max_over_seq_length.max(dim=3, keepdim=True)[0]
        # tv_attention_constant = tv_attention_constant.squeeze(-1).squeeze(-1)


        
        # attention_constant = torch.cat((tt_attention_constant,ta_attention_constant,tv_attention_constant),dim=1)
        
        # attention_constant = self.attend(1-attention_constant)
        
        # t_attention_constant=attention_constant[:,0]
        # a_attention_constant=attention_constant[:,1]
        # v_attention_constant=attention_constant[:,2]


       
        # print(f"proj_t:{proj_t.size()},proj_a:{proj_a.size()},proj_v:{proj_v.size()}")




        fusion_feature = F.relu(self.post_fusion_layer_1(fusion_h), inplace=False)
        # fusion_feature = self.fusion_attention(fusion_h)
        fusion_feature = self.post_fusion_layer_1(fusion_feature[:,-1,:])





       
        output_fusion = self.post_fusion_layer_2(fusion_feature)
        # print(output_fusion)

        
        
       
        text_feature = self.post_text_layer_2(proj_t[:,-1,:])
        audio_feature = self.post_audio_layer_2(proj_t[:,-1,:])
        video_feature = self.post_video_layer_2(proj_v[:,-1,:])

        if self.args.is_agm:
            text_feature = self.m_t_o(text_feature)
            audio_feature = self.m_a_o(audio_feature)
            video_feature = self.m_v_o(video_feature)
            
        # text_feature = self.text_attention(proj_t)
        # video_feature = self.video_attention(proj_v)  
        # audio_feature = self.audio_attention(proj_a)
       
    
       
        output_text = self.post_text_layer_3(text_feature)
        output_audio = self.post_audio_layer_3(audio_feature)
        output_video = self.post_video_layer_3(video_feature)


        res = {
            'M': output_fusion, 
            'T': output_text,
            'A': output_audio,
            'V': output_video,
            'Feature_t': text_feature,
            'Feature_a': audio_feature,
            'Feature_v': video_feature,
            'Feature_f': fusion_feature,
            # 't_attention_constant':t_attention_constant,
            # 'a_attention_constant':a_attention_constant,
            # 'v_attention_constant':v_attention_constant,
        }
        return res



class AuViSubNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(AuViSubNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)

    def forward(self, x, lengths):
        '''
        x: (batch_size, sequence_len, in_size)
        '''
        packed_sequence = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, final_states = self.rnn(packed_sequence)
        h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        return y_1



class MultiheadSelfAttentionWithPooling(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiheadSelfAttentionWithPooling, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads)
        self.embed_size = embed_size

    def forward(self, x):
        # Transpose x to match the input shape requirement for nn.MultiheadAttention: (sequence_length, batch_size, embed_size)
        x = x.transpose(0, 1)
        attention_output, _ = self.multihead_attention(x, x, x)
        # Transpose back to (batch_size, sequence_length, embed_size)
        attention_output = attention_output.transpose(0, 1)
        # Average pooling across the sequence dimension
        pooled_output = torch.mean(attention_output, dim=1)
        return pooled_output
    
class Modality_out(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x

