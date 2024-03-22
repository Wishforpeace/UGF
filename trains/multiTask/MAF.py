import os
import time
import logging
import math
import copy
import argparse
import numpy as np
import pickle as plk
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from tqdm import trange
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import optim
import sys
from utils.functions import dict_to_str
from utils.metricsTop import MetricsTop
from utils.scheduler import build_optimizer
from utils.common import check_and_save
from transformers import get_linear_schedule_with_warmup

logger = logging.getLogger('MSA')

class MAF():
    def __init__(self, args):
        # assert args.train_mode == 'regression'

        self.args = args
        self.args.tasks = "MTAV"
        self.metrics = MetricsTop(args.train_mode).getMetics(args.datasetName)
        self.warmup_steps = self.args.warmup_steps
        self.epochs = self.args.epochs
        self.feature_map = {
            'fusion': torch.zeros(args.train_samples, args.post_fusion_dim, requires_grad=False).to(args.device),
            'text': torch.zeros(args.train_samples, args.post_text_dim, requires_grad=False).to(args.device),
            'audio': torch.zeros(args.train_samples, args.post_audio_dim, requires_grad=False).to(args.device),
            'vision': torch.zeros(args.train_samples, args.post_video_dim, requires_grad=False).to(args.device),
        }

        self.center_map = {
            'fusion': {
                'pos': torch.zeros(args.post_fusion_dim, requires_grad=False).to(args.device),
                'neg': torch.zeros(args.post_fusion_dim, requires_grad=False).to(args.device),
            },
            'text': {
                'pos': torch.zeros(args.post_text_dim, requires_grad=False).to(args.device),
                'neg': torch.zeros(args.post_text_dim, requires_grad=False).to(args.device),
            },
            'audio': {
                'pos': torch.zeros(args.post_audio_dim, requires_grad=False).to(args.device),
                'neg': torch.zeros(args.post_audio_dim, requires_grad=False).to(args.device),
            },
            'vision': {
                'pos': torch.zeros(args.post_video_dim, requires_grad=False).to(args.device),
                'neg': torch.zeros(args.post_video_dim, requires_grad=False).to(args.device),
            }
        }

        self.dim_map = {
            'fusion': torch.tensor(args.post_fusion_dim).float(),
            'text': torch.tensor(args.post_text_dim).float(),
            'audio': torch.tensor(args.post_audio_dim).float(),
            'vision': torch.tensor(args.post_video_dim).float(),
        }


        # new labels
        self.label_map = {
            'fusion': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'text': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'audio': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'vision': torch.zeros(args.train_samples, requires_grad=False).to(args.device)
        }

        self.name_map = {
            'M': 'fusion',
            'T': 'text',
            'A': 'audio',
            'V': 'vision'
        }


    def do_train(self, model, dataloader):
        check = {'Loss': 10000, 'MAE': 100}
        # optimizer,scheduler = build_optimizer(args=self.args,model=model)
        
        
        # bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # bert_params = list(model.module.Model.text_model.named_parameters())

        # text_params = list(model.module.Model.proj_t .named_parameters())
        # audio_params = list(model.module.Model.proj_a.named_parameters())
        # video_params = list(model.module.Model.proj_v.named_parameters())

        # bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
        # bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
        # text_params = [p for n,p in text_params]
        # audio_params = [p for n, p in audio_params]
        # video_params = [p for n, p in video_params]

        # model_params_other = [p for n, p in list(model.module.Model.named_parameters()) if 'proj_t' not in n and \
        #                         'proj_a' not in n and 'proj_v' not in n and 'text_mode' not in n]

        # optimizer_grouped_parameters = [
        #     {'params': bert_params_decay, 'weight_decay': self.args.weight_decay_bert, 'lr': self.args.learning_rate_bert},
        #     {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': self.args.learning_rate_bert},
        #     {'params': text_params, 'weight_decay': self.args.weight_decay_text, 'lr': self.args.learning_rate_text},
        #     {'params': audio_params, 'weight_decay': self.args.weight_decay_audio, 'lr': self.args.learning_rate_audio},
        #     {'params': video_params, 'weight_decay': self.args.weight_decay_video, 'lr': self.args.learning_rate_video},
        #     {'params': model_params_other, 'weight_decay': self.args.weight_decay_other, 'lr': self.args.learning_rate_other}
        # ]
        # optimizer = optim.Adam(optimizer_grouped_parameters)
        if len(self.args.gpu_ids)>1:
            optimizer =  optim.Adam(model.module.Model.parameters(),lr=self.args.learning_rate)
        else:
            optimizer =  optim.Adam(model.Model.parameters(),lr=self.args.learning_rate)


        saved_labels = {}
        # init labels
        logger.info("Init labels...")
        with tqdm(dataloader['train']) as td:
            for batch_data in td:
                labels_m = batch_data['labels']['M'].view(-1).to(self.args.device)
                indexes = batch_data['index'].view(-1)
                self.init_labels(indexes, labels_m)

        # initilize results
        logger.info("Start training...")
        epoch, best_epoch = 0, 0
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        

        epoch_score_t = 0.
        epoch_score_a = 0.
        epoch_score_v = 0.
        epsilon = self.args.epsilon

        # 用于记录学习率的列表
        lr_history = []
        while True: 
        # for epoch in range(self.epochs):
            logger.info("TRAIN-(%s) epoch(%d)" % (self.args.modelName, epoch+1))
            
            # train
            y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
            y_true = {'M': [], 'T': [], 'A': [], 'V': []}
            model.train()
            train_loss = 0.0
            left_epochs = self.args.update_epochs
            ids = []
            epoch += 1 
            with tqdm(dataloader['train']) as td:
                for step,batch_data in enumerate(td):
                    iteration = (epoch-1)*self.args.batch_size + step +1
                    
                    # 衡量模态得分
                    score = {}
                    if self.args.is_ulgm:
                        if left_epochs == self.args.update_epochs:
                            optimizer.zero_grad()
                    else:
                        optimizer.zero_grad()

                    left_epochs -= 1

                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)

                    # print(f"vision.size():{vision.size()}")
                    # print(f"audio.size():{audio.size()}")
                    # print(f"text.size():{text.size()}")
                    # sys.exit(1)
                    indexes = batch_data['index'].view(-1)
                    cur_id = batch_data['id']
                    ids.extend(cur_id)

                    # if not self.args.need_data_aligned:
                    #     audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                    #     vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                    # else:
                    #     audio_lengths, vision_lengths = 0, 0
                   

                    outputs = model(text, audio, vision)
                    # update features
                    f_fusion = outputs['Feature_f'].detach()
                    f_text = outputs['Feature_t'].detach()
                    f_audio = outputs['Feature_a'].detach()
                    f_vision = outputs['Feature_v'].detach()

                    # 造伪标签
                    # if self.args.is_ulgm:
                    #     if epoch > 1:
                    #         self.update_labels(f_fusion, f_text, f_audio, f_vision, epoch, indexes, outputs)
                        
                    #     self.update_features(f_fusion, f_text, f_audio, f_vision, indexes)
                    #     self.update_centers()


                    # # store results
                    for m in self.args.tasks:
                        y_pred[m].append(outputs[m].cpu())
                        y_true[m].append(self.label_map[self.name_map[m]][indexes].cpu())
                        batch_size = self.label_map['fusion'][indexes].size(0)
                        # 使用梯度更新
                        if self.args.is_agm:
                            if m != 'M':
                                difference = torch.tanh(torch.abs(self.label_map[self.name_map[m]][indexes] - self.label_map['fusion'][indexes]))
                                score[m] = torch.sum(1/(difference+epsilon))/batch_size
                            
                           
                    # 梯度更新
                    if self.args.is_agm:
                        ratio_t = math.exp((2*score['T']-score['A']-score['V'])/2)
                        ratio_a = math.exp((2*score['A']-score['T']-score['V'])/2)
                        ratio_v = math.exp((2*score['V']-score['T']-score['A'])/2)

                        
                        optimal_ratio_t = math.exp(2*epoch_score_t-epoch_score_a-epoch_score_v)
                        optimal_ratio_a = math.exp(2*epoch_score_a-epoch_score_t-epoch_score_v)
                        optimal_ratio_v = math.exp(2*epoch_score_v-epoch_score_a-epoch_score_t)


                        coeff_t = math.exp(self.args.alpha*(optimal_ratio_t - ratio_t))
                        coeff_a = math.exp(self.args.alpha*(optimal_ratio_a - ratio_a))
                        coeff_v = math.exp(self.args.alpha*(optimal_ratio_v - ratio_v))



                    # compute loss
                    loss = 0.0
                    for m in self.args.tasks:
                        m = 'M'
                        loss += self.weighted_loss(y_pred=outputs[m],y_true = self.label_map[self.name_map[m]][indexes], \
                                                        indexes=indexes, mode=self.name_map[m])
                            



                    # backward
                    loss.backward()
                    if self.args.is_agm:
                        if len(self.args.gpu_ids)>1:
                            model.module.Model.update_scale(coeff_t,coeff_a,coeff_v)
                        else:
                            
                            model.Model.update_scale(coeff_t,coeff_a,coeff_v)

                        epoch_score_t = epoch_score_t * (iteration - 1) / iteration + score['T'] / iteration
                        epoch_score_a = epoch_score_a * (iteration - 1) / iteration + score['A'] / iteration
                        epoch_score_v = epoch_score_v * (iteration - 1) / iteration + score['V'] / iteration

                        if len(self.args.gpu_ids) > 1:
                            grad_max_fusion = torch.max(model.module.Model.post_fusion_layer_2.weight.grad)
                            grad_min_fusion = torch.min(model.module.Model.post_fusion_layer_2.weight.grad)
                            grad_max_text = torch.max(model.module.Model.post_text_layer_3.weight.grad)
                            grad_min_text = torch.min(model.module.Model.post_text_layer_3.weight.grad)
                            grad_max_audio = torch.max(model.module.Model.post_audio_layer_3.weight.grad)
                            grad_min_audio = torch.min(model.module.Model.post_audio_layer_3.weight.grad)
                            grad_max_video = torch.max(model.module.Model.post_video_layer_3.weight.grad)
                            grad_min_video = torch.min(model.module.Model.post_video_layer_3.weight.grad)
                        else:
                            grad_max_fusion = torch.max(model.Model.post_fusion_layer_2.weight.grad)
                            grad_min_fusion = torch.min(model.Model.post_fusion_layer_2.weight.grad)
                            grad_max_text = torch.max(model.Model.post_text_layer_3.weight.grad)
                            grad_min_text = torch.min(model.Model.post_text_layer_3.weight.grad)
                            grad_max_audio = torch.max(model.Model.post_audio_layer_3.weight.grad)
                            grad_min_audio = torch.min(model.Model.post_audio_layer_3.weight.grad)
                            grad_max_video = torch.max(model.Model.post_video_layer_3.weight.grad)
                            grad_min_video = torch.min(model.Model.post_video_layer_3.weight.grad)

                        grad_max = max(grad_max_fusion, grad_max_text, grad_max_audio, grad_max_video)
                        grad_min = min(grad_min_fusion, grad_min_text, grad_min_audio, grad_min_video)

                        if grad_max > 1.0 and grad_min < 1.0:
                            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)



                    train_loss += loss.item()
                    
                    
                    # update parameters
                    if self.args.is_ulgm:
                        if not left_epochs:
                            # update
                            optimizer.step()
                            left_epochs = self.args.update_epochs
                    else:
                        optimizer.step()
            

           
            # scheduler.step()
                        

            train_loss = train_loss / len(dataloader['train'])
            logger.info("TRAIN-(%s) (%d/%d/%d)>> loss: %.4f " % (self.args.modelName, \
                        epoch-best_epoch, epoch, self.args.cur_time, train_loss))
            

            for m in self.args.tasks:
                pred, true = torch.cat(y_pred[m]), torch.cat(y_true[m])
                train_results = self.metrics(pred, true)
                logger.info('%s: >> ' %(m) + dict_to_str(train_results))
            # validation
                
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            check = check_and_save(model, val_results, check, save_model=True, name='maf_fusion',parallel=self.args.parallel)

            
            # save labels
            if self.args.save_labels:
                tmp_save = {k: v.cpu().numpy() for k, v in self.label_map.items()}
                tmp_save['ids'] = ids
                saved_labels[epoch] = tmp_save
            # early stop
            # if epoch - best_epoch >= self.args.early_stop:
            #     print(f'--------best_epoch:{best_epoch}--------')
            if epoch == self.epochs:
                # plt.plot(lr_history)
                # plt.xlabel('Step')
                # plt.ylabel('Learning Rate')
                # plt.title('Learning Rate Schedule')
                # plt.savefig('ealystop_onecycle_lr_schedule.png')
                # if self.args.save_labels:
                #     with open(os.path.join(self.args.res_save_dir, f'{self.args.modelName}-{self.args.datasetName}-labels.pkl'), 'wb') as df:
                #         plk.dump(saved_labels, df, protocol=4)
                return 

    def do_test(self, model, dataloader, mode="VAL"):
        model.eval()
        y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
        y_true = {'M': [], 'T': [], 'A': [], 'V': []}
        eval_loss = 0.0
        # criterion = nn.L1Loss()
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    
                    # audio_lengths = audio_lengths.cpu().to(torch.int64)
                    # vision_lengths = vision_lengths.cpu().to(torch.int64)
                    labels_m = batch_data['labels']['M'].to(self.args.device).view(-1)
                    outputs = model(text, audio, vision)
                    loss = self.weighted_loss(y_pred=outputs['M'], y_true=labels_m,task='test')
                    eval_loss += loss.item()
                    y_pred['M'].append(outputs['M'].cpu())
                    y_true['M'].append(labels_m.cpu())
        eval_loss = eval_loss / len(dataloader)
        logger.info(mode+"-(%s)" % self.args.modelName + " >> loss: %.4f " % eval_loss)
        pred, true = torch.cat(y_pred['M']), torch.cat(y_true['M'])
        eval_results = self.metrics(pred, true)
        logger.info('M: >> ' + dict_to_str(eval_results))
        eval_results['Loss'] = eval_loss
        return eval_results
    
    def weighted_loss(self, y_pred, y_true, weighted=None, indexes=None, mode='fusion',task='train',temperature=0.1):
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        
        if self.args.is_ulgm:
            if mode == 'fusion':
                weighted = torch.ones_like(y_pred)
            else:
                weighted = torch.tanh(torch.abs(self.label_map[mode][indexes] - self.label_map['fusion'][indexes]))
        else:
            weighted = torch.ones_like(y_pred)
       
        mae_loss = torch.mean(weighted * torch.abs(y_pred - y_true))
        if torch.isnan(mae_loss):
            print(f"weighted:{weighted}")
            sys.eixt(1)


        
        return mae_loss
        if task == 'train':
            #获取正负样本的特征向量
            neg_indexes = self.label_map[mode][indexes] < 0

            pos_indexes = self.label_map[mode][indexes] > 0

            
            neg_features = self.feature_map[mode][indexes][neg_indexes]
            pos_features = self.feature_map[mode][indexes][pos_indexes]

           
            # 确保pos_features和neg_features是归一化的
            pos_features = F.normalize(pos_features, p=2, dim=1)
            neg_features = F.normalize(neg_features, p=2, dim=1)

            # 计算正样本之间的相似度
            pos_similarity = torch.matmul(pos_features, pos_features.t())
            
            pos_similarity.fill_diagonal_(0)  # 将对角线元素设置为0，排除自身相似度

            # 计算正样本和负样本之间的相似度
            neg_similarity = torch.matmul(pos_features, neg_features.t())
           

            # 将相似度转换为概率
            pos_prob = torch.exp(pos_similarity)/ temperature
            neg_prob = torch.exp(neg_similarity)/ temperature

            neg_row_sums = neg_prob.sum(dim=1)
            neg_denominator = neg_row_sums.unsqueeze(1).repeat(1, pos_prob.size(1))

            denominator = neg_denominator + pos_prob

            
            fraction = pos_prob / denominator
            mask = ~torch.eye(fraction.size(0), dtype=torch.bool).to(self.args.device)

            sim = -torch.log(fraction)

            sim =  sim * mask

            i_loss = sim.sum()/mask.sum()

            return (mae_loss + 0.1*i_loss)
            
        else:
            return mae_loss
    
    def update_features(self, f_fusion, f_text, f_audio, f_vision, indexes):
        self.feature_map['fusion'][indexes] = f_fusion
        self.feature_map['text'][indexes] = f_text
        self.feature_map['audio'][indexes] = f_audio
        self.feature_map['vision'][indexes] = f_vision

    def update_centers(self):
        def update_single_center(mode):
            neg_indexes = self.label_map[mode] < 0
            if self.args.excludeZero:
                pos_indexes = self.label_map[mode] > 0
            else:
                pos_indexes = self.label_map[mode] >= 0
            self.center_map[mode]['pos'] = torch.mean(self.feature_map[mode][pos_indexes], dim=0)
            self.center_map[mode]['neg'] = torch.mean(self.feature_map[mode][neg_indexes], dim=0)

        update_single_center(mode='fusion')
        update_single_center(mode='text')
        update_single_center(mode='audio')
        update_single_center(mode='vision')
    
    def init_labels(self, indexes, m_labels):
        self.label_map['fusion'][indexes] = m_labels
        self.label_map['text'][indexes] = m_labels
        self.label_map['audio'][indexes] = m_labels
        self.label_map['vision'][indexes] = m_labels
    
    def update_labels(self, f_fusion, f_text, f_audio, f_vision, cur_epoches, indexes, outputs):
        MIN = 1e-8
        def update_single_label(f_single, mode):
            d_sp = torch.norm(f_single - self.center_map[mode]['pos'], dim=-1) 
            d_sn = torch.norm(f_single - self.center_map[mode]['neg'], dim=-1) 
            delta_s = (d_sn - d_sp) / (d_sp + MIN)
            alpha = delta_s / (delta_f + MIN)

            new_labels = 0.5 * alpha * self.label_map['fusion'][indexes] + \
                        0.5 * (self.label_map['fusion'][indexes] + delta_s - delta_f)
            new_labels = torch.clamp(new_labels, min=-self.args.H, max=self.args.H)
            # new_labels = torch.tanh(new_labels) * self.args.H

            n = cur_epoches
            self.label_map[mode][indexes] = (n - 1) / (n + 1) * self.label_map[mode][indexes] + 2 / (n + 1) * new_labels

        d_fp = torch.norm(f_fusion - self.center_map['fusion']['pos'], dim=-1) 
        d_fn = torch.norm(f_fusion - self.center_map['fusion']['neg'], dim=-1) 
        # d_f_pn = torch.norm(self.center_map['fusion']['pos'] - self.center_map['fusion']['neg'], dim=-1)
        # delta_f = (d_fn - d_fp) / (d_f_pn + MIN)
        delta_f = (d_fn - d_fp) / (d_fp + MIN)
        
        update_single_label(f_text, mode='text')
        update_single_label(f_audio, mode='audio')
        update_single_label(f_vision, mode='vision')


        