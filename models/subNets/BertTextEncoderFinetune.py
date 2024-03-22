import torch
import torch.nn as nn
import torch.nn.functional as F
from models.subNets.BaseClassifier import BaseClassifier
from transformers import BertTokenizer, BertModel
import sys

class BertTextEncoder(nn.Module):
    def __init__(self, language='en'):
        """
        language: en / cn
        """
        super(BertTextEncoder, self).__init__()

        assert language in ['en', 'cn']

        tokenizer_class = BertTokenizer
        model_class = BertModel


        # directory is fine
        # pretrained_weights = '/home/sharing/disk3/pretrained_embedding/Chinese/bert/pytorch'
        if language == 'en':
            self.tokenizer = tokenizer_class.from_pretrained('/mnt/disk1/wyx/pretrained/bert/pretrained_berts/bert_en', do_lower_case=True)
            self.extractor = model_class.from_pretrained('/mnt/disk1/wyx/pretrained/bert/pretrained_berts/bert_en')
        elif language == 'cn':
            self.tokenizer = tokenizer_class.from_pretrained('/mnt/disk1/wyx/pretrained/bert/pretrained_berts/bert_cn')
            self.extractor = model_class.from_pretrained('/mnt/disk1/wyx/pretrained/bert/pretrained_berts/bert_cn')
        

    def forward(self, text):
        """
        text: (batch_size, 3, seq_len)
        3: input_ids, input_mask, segment_ids
        input_ids: input_ids,
        input_mask: attention_mask,
        segment_ids: token_type_ids
        """
        input_ids, input_mask, segment_ids = text[:,0,:].long(), text[:,1,:].long(), text[:,2,:].long()
        
        last_hidden_states = self.extractor(input_ids=input_ids,
                                                attention_mask=input_mask,
                                                token_type_ids=segment_ids)
               
        last_hidden_states = last_hidden_states['pooler_output']
        return last_hidden_states



    def set_train(self, train_module=None):
        if train_module is None:
            train_module = True
        for name, param in self.extractor.named_parameters():
                param.requires_grad = train_module


class BertTextEncoderPretrain(nn.Module):
    def __init__(self,args):
        super(BertTextEncoderPretrain,self).__init__()
        self.args = args
        self.device = args.device
        drop_out = self.args.post_text_dropout
        self.encoder = BertTextEncoder()  # bert output 768

        self.proj_fea_dim = args.proj_fea_dim
        self.classifier = BaseClassifier(input_size=self.proj_fea_dim,
                                         hidden_size=[int(self.proj_fea_dim / 2), int(self.proj_fea_dim / 4),
                                                      int(self.proj_fea_dim / 8)],
                                         output_size=1, drop_out=drop_out)
        self.criterion = torch.nn.MSELoss()
        

    def forward(self, text, label):
        x = self.encoder(text)

        pred = self.classifier(x)
        loss = self.criterion(pred.squeeze(), label.squeeze())

       
        return pred, x, loss

    def save_model(self):
        # save all modules
        path = self.args.model_save_path + f'{self.args.datasetName}-text'
        encoder_path =  path + '-encoder.pth'
        decoder_path =  path+ '-decoder.pth'
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.classifier.state_dict(), decoder_path)
        print('model saved at:')
        print(encoder_path)
        print(decoder_path)


    def load_model(self, module='all'):
        path = self.args.model_save_path + f'{self.args.datasetName}-text'
        encoder_path =  path + '-encoder.pth'
        decoder_path =  path+ '-decoder.pth'
        print('model loaded from:')
        if module == 'encoder':
            print(encoder_path)
            self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
        if module == 'decoder':
            print(decoder_path)
            self.classifier.load_state_dict(torch.load(decoder_path, map_location=self.device))
        if module == 'all' or module is None:
            print(encoder_path)
            print(decoder_path)
            self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
            self.classifier.load_state_dict(torch.load(decoder_path, map_location=self.device))


    def set_train(self, train_module=None):
        if train_module is None:
            train_module = [True, True]
        self.encoder.set_train(train_module=train_module[0])
        for param in self.classifier.parameters():
            param.requires_grad = train_module[1]