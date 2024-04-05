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
from models.loss import BMCLoss
logger = logging.getLogger('MSA')

class TVA_Fusion():
    def __init__(self,args):
        self.args = args
        self.epochs = self.args.fusion_epochs
        self.finetune_epochs = self.args.finetune_epochs
        self.metrics = MetricsTop(args.train_mode).getMetics(args.datasetName)

    def do_train(self,model,dataloader,check,load_pretrain=True):

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": self.args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        }]
        optimizer,scheduler = build_optimizer(args=self.args,optimizer_grouped_parameters=optimizer_grouped_parameters,epochs=self.epochs)
        # if self.args.parallel:
        #     optimizer =  optim.Adam(model.module.Model.parameters(),lr=self.args.learning_rate,weight_decay=self.args.weight_decay)
        # else:
        #     optimizer =  optim.Adam(model.Model.parameters(),lr=self.args.learning_rate)

        epoch, best_epoch = 0, 0
        save_start_epoch = 1
        loss = 0.0
        train_loss = 0.0
        
        if self.args.parallel:
            model.module.Model.load_model(load_pretrain=True)
        else:
            model.Model.load_model(load_pretrain=True)

        epoch_score_t = 0.
        epoch_score_a = 0.
        epoch_score_v = 0.
        # criterion = nn.MSELoss(reduction='none')
        criterion = BMCLoss(init_noise_sigma=self.args.init_noise_sigma,device=self.args.device)
        self.epsilon = self.args.epsilon
        for epoch in range(1,self.epochs+1):
            model.train()
            if self.args.parallel:
                # model.module.Model.set_train([True, True, True])
                if epoch < self.finetune_epochs:
                    model.module.Model.set_train([False, True, True])
                else:
                    model.module.Model.set_train([True, True, True])
            else:
                # model.Model.set_train([True, True, True])
                if epoch < self.finetune_epochs:
                    model.Model.set_train([False, True, True])
                else:
                    model.Model.set_train([True, True, True])

            logger.info("TRAIN-(%s) epoch(%d)" % (self.args.modelName, epoch))
            y_pred =[]
            y_true =[]
           
            with tqdm(dataloader['train']) as td:
                for step,batch_data in enumerate(td):
                    step = step+1
                    iteration = (epoch-1)*self.args.batch_size + step
                    optimizer.zero_grad()
                    text = batch_data['text'].clone().detach().to(self.args.device)
                    vision = batch_data['vision'].clone().detach().to(self.args.device)
                    vision_mask = batch_data['vision_padding_mask'].clone().detach().to(self.args.device)
                    audio = batch_data['audio'].clone().detach().to(self.args.device)
                    audio_mask = batch_data['audio_padding_mask'].clone().detach().to(self.args.device)
                    labels = batch_data['labels']['M'].clone().detach().to(self.args.device)
                    pred_fusion,pred_t,pred_a,pred_v = model(text=text, vision=vision, audio=audio, vision_mask=vision_mask, audio_mask=audio_mask, labels = labels)
                    pred_loss = criterion(pred_fusion, labels)
                    loss_t = criterion(pred_t, labels)
                    loss_a = criterion(pred_a, labels)
                    loss_v = criterion(pred_v, labels)
                    mono_loss = loss_t + loss_a + loss_v
                    loss = torch.mean(pred_loss +  self.args.w * mono_loss)

                    t_difference = torch.tanh(torch.abs(pred_t - labels))
                    t_score = torch.sum(1/(t_difference+self.epsilon))/pred_t.size(0)

                    v_difference = torch.tanh(torch.abs(pred_v - labels))
                    v_score = torch.sum(1/(v_difference+self.epsilon))/pred_t.size(0)

                    a_difference = torch.tanh(torch.abs(pred_a - labels))
                    a_score = torch.sum(1/(a_difference+self.epsilon))/pred_t.size(0)
                    
                  


                    y_true.append(labels.cpu())
                    y_pred.append(pred_fusion.cpu())
                    loss.backward()
                    
                    if self.args.is_agm:
                        ratio_t = math.exp((2*t_score-a_score-v_score)/2)
                        ratio_a = math.exp((2*a_score-t_score-v_score)/2)
                        ratio_v = math.exp((2*v_score-t_score-a_score)/2)

                        
                        optimal_ratio_t = math.exp((2*epoch_score_t-epoch_score_a-epoch_score_v)/2)
                        optimal_ratio_a = math.exp((2*epoch_score_a-epoch_score_t-epoch_score_v)/2)
                        optimal_ratio_v = math.exp((2*epoch_score_v-epoch_score_a-epoch_score_t)/2)


                        coeff_t = math.exp(self.args.alpha*(optimal_ratio_t - ratio_t))
                        coeff_a = math.exp(self.args.alpha*(optimal_ratio_a - ratio_a))
                        coeff_v = math.exp(self.args.alpha*(optimal_ratio_v - ratio_v))


                        
                        epoch_score_t = epoch_score_t * (iteration - 1) / iteration + t_score / iteration
                        epoch_score_a = epoch_score_a * (iteration - 1) / iteration + a_score / iteration
                        epoch_score_v = epoch_score_v * (iteration - 1) / iteration + v_score / iteration
                        if self.args.parallel:
                            model.module.Model.update_scale(coeff_t,coeff_a,coeff_v)

                            grad_max_t = torch.max(model.module.Model.t_mono_decoder.MLP[-1].weight.grad)
                            grad_min_t = torch.min(model.module.Model.t_mono_decoder.MLP[-1].weight.grad)
                            grad_max_v = torch.max(model.module.Model.v_mono_decoder.MLP[-1].weight.grad)
                            grad_min_v = torch.min(model.module.Model.v_mono_decoder.MLP[-1].weight.grad)
                            grad_max_a = torch.max(model.module.Model.a_mono_decoder.MLP[-1].weight.grad)
                            grad_min_a = torch.min(model.module.Model.a_mono_decoder.MLP[-1].weight.grad)
                            grad_max = max(grad_max_t,grad_max_v,grad_max_a)
                            grad_min = min(grad_min_t,grad_min_v,grad_min_a)

                        else:
                            model.Model.update_scale(coeff_t,coeff_a,coeff_v)
                            
                            grad_max_t = torch.max(model.Model.t_mono_decoder.MLP[-1].weight.grad)
                            grad_min_t = torch.min(model.Model.t_mono_decoder.MLP[-1].weight.grad)
                            grad_max_v = torch.max(model.Model.v_mono_decoder.MLP[-1].weight.grad)
                            grad_min_v = torch.min(model.Model.v_mono_decoder.MLP[-1].weight.grad)
                            grad_max_a = torch.max(model.Model.a_mono_decoder.MLP[-1].weight.grad)
                            grad_min_a = torch.min(model.Model.a_mono_decoder.MLP[-1].weight.grad)
                            grad_max = max(grad_max_t,grad_max_v,grad_max_a)
                            grad_min = min(grad_min_t,grad_min_v,grad_min_a)

                        # print(f"grad_max:{grad_max}\ngrad_min:{grad_min}")
                        if grad_max > 1.0 and grad_min < -1.0:
                            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


                    train_loss += loss.item()
                    optimizer.step()


                    scheduler.step()
                    
           
            
                     
            train_loss = train_loss / len(dataloader['train'])
            logger.info("TRAIN-(%s) (%d/%d/%d)>> loss: %.4f " % (self.args.modelName, \
                            epoch-best_epoch, epoch, self.args.cur_time, train_loss))
            out_pred, true = torch.cat(y_pred), torch.cat(y_true)
            # if epoch == 10:
            #     print(out_pred)
            #     print(true)
            #     sys.exit(1)
            train_results = self.metrics(out_pred, true)
            logger.info('%s: >> ' %('fusion') + dict_to_str(train_results))

            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            # _ = check_and_save(model=model,result=val_results, check=check,parallel=self.args.parallel)

            test_results = self.do_test(model, dataloader['test'], mode="T")
            if epoch > save_start_epoch:
                check = check_and_save(model=model,result=test_results, check=check,parallel=self.args.parallel)
            
            torch.cuda.empty_cache()
                
                    
            
    
    def do_test(self,model,dataloader,mode='VAL'):
        
        if mode == 'VAL':
            model.eval()
        else:
            if self.args.parallel:
               model.module.load_model(module=False)
            else:   
                model.load_model(module=False)
        # criterion = nn.MSELoss(reduction='none')
        criterion = BMCLoss(init_noise_sigma=self.args.init_noise_sigma,device=self.args.device)
        with torch.no_grad():
            val_loss = 0.0
            y_pred =[]
            y_true =[]
            with tqdm(dataloader) as td:
                for batch_data in td:
                    text = batch_data['text'].clone().detach().to(self.args.device)
                    vision = batch_data['vision'].clone().detach().to(self.args.device)
                    vision_mask = batch_data['vision_padding_mask'].clone().detach().to(self.args.device)
                    audio = batch_data['audio'].clone().detach().to(self.args.device)
                    audio_mask = batch_data['audio_padding_mask'].clone().detach().to(self.args.device)
                    labels = batch_data['labels']['M'].clone().detach().to(self.args.device)
                    pred_fusion,pred_t,pred_a,pred_v = model(text=text, vision=vision, audio=audio, vision_mask=vision_mask, audio_mask=audio_mask, labels = labels)
                    pred_loss = criterion(pred_fusion, labels)
                    loss_t = criterion(pred_t, labels)
                    loss_a = criterion(pred_a, labels)
                    loss_v = criterion(pred_v, labels)
                    mono_loss = loss_t + loss_a + loss_v
                    loss = torch.mean(pred_loss + 0.1 * mono_loss)
                    val_loss += loss.item()
                    y_pred.append(pred_fusion)
                    y_true.append(labels)
        val_loss = val_loss / len(dataloader)
        logger.info(mode+"-(%s)" % self.args.modelName + " >> loss: %.4f " % val_loss)
        out_pred, true = torch.cat(y_pred), torch.cat(y_true)
        val_results = self.metrics(out_pred, true)
        val_results['Loss'] = val_loss
        logger.info('Fusion: >> ' + dict_to_str(val_results))
        return val_results
        
