import os
import gc
import time
import random
import torch
import pynvml
import logging
import argparse
import sys
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from models.AMIO import AMIO
from trains.ATIO import ATIO
from data.load_data import MMDataLoader
from config.config import ConfigPretrain



os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def run(args,seed):
    model_save_path = args.model_save_dir+f'/{str(seed)}/'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    args.model_save_path = model_save_path

    
    if len(args.gpu_ids) == 0 and torch.cuda.is_available():
        # load free-most gpu
        pynvml.nvmlInit()
        dst_gpu_id, min_mem_used = 0, 1e16
        for g_id in [0, 1, 2, 3]:
            handle = pynvml.nvmlDeviceGetHandleByIndex(g_id)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used = meminfo.used
            if mem_used < min_mem_used:
                min_mem_used = mem_used
                dst_gpu_id = g_id
        print(f'Find gpu: {dst_gpu_id}, use memory: {min_mem_used}!')
        logger.info(f'Find gpu: {dst_gpu_id}, with memory: {min_mem_used} left!')
        args.gpu_ids.append(dst_gpu_id)
    # device
    using_cuda = len(args.gpu_ids) > 0 and torch.cuda.is_available()
    logger.info("Let's use %d GPUs!" % len(args.gpu_ids))
    device = torch.device('cuda:%d' % int(args.gpu_ids[0]) if using_cuda else 'cpu')
    args.device = device
    # data
    dataloader = MMDataLoader(args)

    
    model = AMIO(args).to(device)
    

    def count_parameters(model):
        answer = 0
        for p in model.parameters():
            if p.requires_grad:
                answer += p.numel()
                # print(p)
        return answer
    logger.info(f'The model has {count_parameters(model)} trainable parameters')
    
    # using multiple gpus
    if using_cuda and len(args.gpu_ids) > 1:
        args.parallel = True
        model = torch.nn.DataParallel(model,
                                      device_ids=args.gpu_ids,
                                      output_device=args.gpu_ids[0])
    
    atio = ATIO().getTrain(args)
    # do train
    atio.do_train(model, dataloader)


    

    results = atio.do_test(model=model, dataloader=dataloader['test'],mode='TEST')
    # do test
   
    del model



    torch.cuda.empty_cache()
    gc.collect()

    return results


def run_finetune(args, tune_times=50):
    args.res_save_dir = os.path.join(args.res_save_dir, 'tunes')
    init_args = args
    has_debuged = [] # save used paras
    save_file_path = os.path.join(args.res_save_dir, \
                                f'{args.datasetName}-{args.modelName}-{args.train_mode}-tune.csv')
    if not os.path.exists(os.path.dirname(save_file_path)):
        os.makedirs(os.path.dirname(save_file_path))
    
    for i in range(tune_times):
        # load free-most gpus
        pynvml.nvmlInit()
        # cancel random seed
        setup_seed(int(time.time()))
        args = init_args
        config = ConfigTune(args)
        args = config.get_config()
        print(args)
        # print debugging params
        logger.info("#"*40 + '%s-(%d/%d)' %(args.modelName, i+1, tune_times) + '#'*40)
        for k,v in args.items():
            if k in args.d_paras:
                logger.info(k + ':' + str(v))
        logger.info("#"*90)
        logger.info('Start running %s...' %(args.modelName))
        # restore existed paras
        if i == 0 and os.path.exists(save_file_path):
            df = pd.read_csv(save_file_path)
            for i in range(len(df)):
                has_debuged.append([df.loc[i,k] for k in args.d_paras])
        # check paras
        cur_paras = [args[v] for v in args.d_paras]
        if cur_paras in has_debuged:
            logger.info('These paras have been used!')
            time.sleep(3)
            continue
        has_debuged.append(cur_paras)
        results = []
        for j, seed in enumerate([1111]):
            args.cur_time = j + 1
            setup_seed(seed)
            results.append(run(args))
        # save results to csv
        logger.info('Start saving results...')
        if os.path.exists(save_file_path):
            df = pd.read_csv(save_file_path)
        else:
            df = pd.DataFrame(columns = [k for k in args.d_paras] + [k for k in results[0].keys()])
        # stat results
        tmp = [args[c] for c in args.d_paras]
        
        for col in results[0].keys():
            values = [r[col] for r in results]
            tmp.append(round(sum(values) * 100 / len(values), 2))
        
        df.loc[len(df)] = tmp
        df.to_csv(save_file_path, index=None)
        logger.info('Results are saved to %s...' %(save_file_path))




def run_pretrain(args):
    args.res_save_dir = os.path.join(args.res_save_dir, args.train_mode)
    init_args = args
    model_results = {}
    seeds = args.seeds
    # run results
    for i, seed in enumerate(seeds):
        args = init_args
        # load config
        if args.train_mode == "pretrain":
            config = ConfigPretrain(args)
        args = config.get_config()
        setup_seed(seed)
        args.seed = seed
        logger.info('Start running %s...' %(args.modelName))
        logger.info(args)
        # runnning
        args.cur_time = i+1
        test_results = run(args,seed)
        # restore results
        model_results[seed]=test_results


    first_seed = list(model_results.keys())[0]
    criterions = list(model_results[first_seed].keys())

    
    # load other results
    save_path = os.path.join(args.res_save_dir,
                        args.modelName + '-' + args.datasetName + '-' + args.train_mode + '-' + datetime.datetime.now().strftime('%Y-%m-%d-%H%M') +'.csv')

    if not os.path.exists(args.res_save_dir):
        os.makedirs(args.res_save_dir)


    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
    else:
        df = pd.DataFrame(columns=["Model","Seed"] + criterions)

    # Populate the DataFrame with the new results
    for seed, results in model_results.items():
        row = {"Model": args.modelName, "Seed": seed}
        row.update(results)
        df = df.append(row, ignore_index=True)

    df.to_csv(save_path, index=False)

    logger.info('Results are added to %s...' %(save_path))

   


def set_log(args):
    log_file_path = f'logs/{args.modelName}-{args.datasetName}.log'
    # set logging
    logger = logging.getLogger() 
    logger.setLevel(logging.DEBUG)

    for ph in logger.handlers:
        logger.removeHandler(ph)
    # add FileHandler to log file
    formatter_file = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter_file)
    logger.addHandler(fh)
    # add StreamHandler to terminal outputs
    formatter_stream = logging.Formatter('%(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter_stream)
    logger.addHandler(ch)
    return logger


# def run_mono_modal(args):
    






def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_tune', type=bool, default=False,
                        help='tune parameters ?')
    parser.add_argument('--train_mode', type=str, default="pretrain",
                        help='pretrain / finetune')
    parser.add_argument('--modelName', type=str, default='maf',
                        help='support maf')
    parser.add_argument('--datasetName', type=str, default='sims',
                        help='support mosi/mosei/sims')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num workers of loading data')
    parser.add_argument('--model_save_dir', type=str, default='/mnt/disk1/wyx/MSA/Lab/ModalAdaptationMSA/results/models',
                        help='path to save results.')
    parser.add_argument('--res_save_dir', type=str, default='/mnt/disk1/wyx/MSA/Lab/ModalAdaptationMSA/results/results',
                        help='path to save results.')
    parser.add_argument('--gpu_ids', type=list, default=[0,1],
                        help='indicates the gpus will be used. If none, the most-free gpu will be used!')
    return parser.parse_args()

if __name__ == '__main__':
    start_time = time.time()
    args = parse_args()
    logger = set_log(args)
    # for data_name in ['mosi', 'mosei','sims']:


    # 组合 [concat,ulgm,almt,agm]

    ablation = [
        # 1个True
        # [True,False,False,False],
        [False,False,True,False],
        # #2个True
        # [True,True,False,False],
        # [False,True,True,False],
        # #3个True
        # [True,True,False,True],
        # [False,True,True,True]
        ]
    
    # for i in ablation:
    #     args.is_concat,args.is_ulgm,args.is_almt,args.is_agm = i
    #     for data_name in ['mosi']:
    #         args.datasetName = data_name
    #         args.seeds = [1111, 1112, 1113, 1114]
    #         if args.is_tune:
    #             run_finetune(args, tune_times=50)
    #         else:
    #             run_pretrain(args)
                # run_mono_modal(args)
    args.seeds = [1111, 1112,1113,1114]
    for i in ['text','vision','audio']:
        for data_name in ['mosi']:
            args.modelName = i
            args.datasetName = data_name
            if args.is_tune:
                run_finetune(args, tune_times=50)
            else:
                run_pretrain(args)




    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"代码运行时间: {elapsed_time} 秒")

