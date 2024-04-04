import os
import argparse

from utils.functions import Storage

class ConfigPretrain():
    def __init__(self, args):
        # hyper parameters for models
        HYPER_MODEL_MAP = {
            'maf': self.__MAF,
            'text':self.__MAF,
            'vision':self.__MAF,
            'audio':self.__MAF,
            'fusion':self.__MAF,

        }
        # hyper parameters for datasets
        HYPER_DATASET_MAP = self.__datasetCommonParams()

        # normalize
        model_name = str.lower(args.modelName)
        dataset_name = str.lower(args.datasetName)
        # load params
        commonArgs = HYPER_MODEL_MAP[model_name]()['commonParas']
        dataArgs = HYPER_DATASET_MAP[dataset_name]
        dataArgs = dataArgs['aligned'] if (commonArgs['need_data_aligned'] and 'aligned' in dataArgs) else dataArgs['unaligned']
        # integrate all parameters
        self.args = Storage(dict(vars(args),
                            **dataArgs,
                            **commonArgs,
                            **HYPER_MODEL_MAP[model_name]()['datasetParas'][dataset_name],
                            ))
    
    def __datasetCommonParams(self):
        root_dataset_dir = '/mnt/disk1/wyx/datasets/MSADatasets'
        tmp = {
            'mosi':{
                'aligned': {
                    # 'dataPath': os.path.join(root_dataset_dir, 'MOSI/new_mosi_data.pkl'),
                    'dataPath': os.path.join(root_dataset_dir, 'MOSI/Processed/aligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, vision)
                    'feature_dims': (768, 5, 20),
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss' 
                },
                'unaligned': {
                    # 'dataPath': os.path.join(root_dataset_dir, 'MOSI/new_mosi_data.pkl'),
                    'dataPath': os.path.join(root_dataset_dir, 'MOSI/Processed/unaligned_50.pkl'),
                    'seq_lens': (50, 375, 500),
                    # (text, audio, vision)
                    # 'feature_dims': (768, 64, 64),
                    'feature_dims': (768, 5, 20),
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss' 
                }
            },
            'mosei':{
                'aligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSEI/Processed/aligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, vision)
                    'feature_dims': (768, 74, 35),
                    'train_samples': 16326,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss'
                },
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSEI/Processed/unaligned_50.pkl'),
                    'seq_lens': (50, 500, 375),
                    # (text, audio, vision)
                    'feature_dims': (768, 74, 35),
                    'train_samples': 16326,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss'
                }
            },
            'sims':{
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'SIMS/Processed/unaligned_39.pkl'),
                    # (batch_size, seq_lens, feature_dim)
                    'seq_lens': (39, 400, 55), # (text, audio, vision)
                    'feature_dims': (768, 33, 709), # (text, audio, vision)
                    'train_samples': 1368,
                    'num_classes': 3,
                    'language': 'cn',
                    'KeyEval': 'Loss',
                }
            }
        }
        return tmp

    def __MAF(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': False,
                'need_model_aligned': False,
                'need_normalized': False,
                'use_bert': True,
                'use_finetune': True,
                'save_labels': True,
                'early_stop': 8,
                'update_epochs': 2,
                'warmup_steps':2,
                'AHL_depth':3,
                'fusion_layer_depth':2,
                'nums_head':8,
                'epsilon':1,
                'alpha':1,
                'modulation_starts ':0,
                'modulation_ends ':50,
                'lr_scalar':'onecyclewarmup',
                'optim':'adamw',
                'learning_rate':1e-4,
                'weight_decay':1e-3,
                'lr_decay_step':10,
                'lr_decay_ratio':0.1,
                'ln_threshold':2e-2,
                'init_noise_sigma':1.0
            },
            # dataset
            'datasetParas':{
                'mosi':{
                    # the batch_size of each epoch is update_epochs * batch_size
                    'batch_size': 64,
                    'text_epochs':100,
                    'vision_epochs':100,
                    'audio_epochs':100,
                    'fusion_epochs':50,
                    'finetune_epochs':25,
                    'encoder_fea_dim':768,
                    'text_out': 768, 
                    'audio_out': 16,
                    'vision_out': 32, 
                    'vision_nhead':8,
                    'vision_tf_num_layers': 2,
                    'audio_out': 32, 
                    'audio_nhead':8,
                    'audio_tf_num_layers': 2,
                    'proj_fea_dim':768,
                    'fusion_nhead':8,
                    # post feature
                    'post_fusion_dim': 128,
                    'post_text_dim':128,
                    'post_audio_dim': 128,
                    'post_vision_dim': 128,
                    'post_fusion_dropout': 0.3,
                    'post_text_dropout': 0.3,
                    'post_audio_dropout': 0.3,
                    'post_vision_dropout': 0.3,
                    # res
                    'H': 3.0
                },
                'mosei':{
                    'batch_size': 32,
                    'text_epochs':100,
                    'vision_epochs':100,
                    'audio_epochs':100,
                    'fusion_epochs':10,
                    'finetune_epochs':5,
                    'encoder_fea_dim':768,
                    'text_out': 768, 
                    'audio_out': 16,
                    'vision_out': 32, 
                    'vision_nhead':8,
                    'vision_tf_num_layers': 2,
                    'audio_out': 32, 
                    'audio_nhead':8,
                    'audio_tf_num_layers': 2,
                    'proj_fea_dim':768,
                    'fusion_nhead':8,
                    # post feature
                    'post_fusion_dim': 128,
                    'post_text_dim':128,
                    'post_audio_dim': 128,
                    'post_vision_dim': 128,
                    'post_fusion_dropout': 0.3,
                    'post_text_dropout': 0.3,
                    'post_audio_dropout': 0.3,
                    'post_vision_dropout': 0.3,
                  
                },
                'sims':{
                    'batch_size': 16,
                    'text_epochs':100,
                    'vision_epochs':100,
                    'audio_epochs':100,
                    'fusion_epochs':100,
                    'finetune_epochs':75,
                    'encoder_fea_dim':768,
                    'text_out': 768, 
                    'audio_out': 16,
                    'vision_out': 32, 
                    'vision_nhead':8,
                    'vision_tf_num_layers': 2,
                    'audio_out': 32, 
                    'audio_nhead':8,
                    'audio_tf_num_layers': 2,
                    'proj_fea_dim':768,
                    'fusion_nhead':8,
                    # post feature
                    'post_fusion_dim': 128,
                    'post_text_dim':128,
                    'post_audio_dim': 128,
                    'post_vision_dim': 128,
                    'post_fusion_dropout': 0.3,
                    'post_text_dropout': 0.3,
                    'post_audio_dropout': 0.3,
                    'post_vision_dropout': 0.3,
                },
            },
        }
        return tmp

    def get_config(self):
        return self.args