import argparse
import logging
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, Dataset
from transformers import (BartForConditionalGeneration,
                          BartTokenizer)
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup

parser = argparse.ArgumentParser(description='BART Sarcasm')

parser.add_argument('--checkpoint_path',
                    type=str,
                    help='checkpoint path')

parser.add_argument('--test',
                    type=str,
                    help='for test data')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class ArgsBase():
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--train_file',
                            type=str,
                            default='./data/withc_train.txt',
                            help='train file')

        parser.add_argument('--valid_file',
                            type=str,
                            default='./data/withc_valid.txt',
                            help='valid file')

        parser.add_argument('--test_file',
                            type=str,
                            default='./data/withc_test.txt',
                            help='test file')
        
        parser.add_argument('--batch_size',
                            type=int,
                            default=14,
                            help='')

        parser.add_argument('--max_src_len',
                            type=int,
                            default=40,
                            help='max src len')
        
        parser.add_argument('--max_tgt_len',
                            type=int,
                            default=40,
                            help='max tgt len')

        return parser


class SARCDataset(Dataset):
    def __init__(self, filepath, max_src_len=104, max_tgt_len=42) -> None:
        self.filepath = filepath
        self.data = self.parse_data(self.filepath) 
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.sep_token = '<sep>'
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    def __len__(self):
        return len(self.data)

    def parse_data(self, filepath):
        data = []

        with open(filepath) as f:
            lines = f.readlines()
        
        for line in lines:
            data.append(line.strip().split('|'))

        return data

    def make_input_id_mask(self, tokens, max_len, index):
        input_id = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_id)
        if len(input_id) < max_len:
            while len(input_id) < max_len:
                input_id += [self.tokenizer.pad_token_id]
                attention_mask += [0]
        else:
            # logging.warning(f'exceed max_seq_len for given article : {index}')
            input_id = input_id[:max_len - 1] + [
                self.tokenizer.eos_token_id]
            attention_mask = attention_mask[:max_len]
        return input_id, attention_mask

    def __getitem__(self, index):
        record = self.data[index]
        #c, ns, s = record[0], record[1], record[2]

        #src_tokens = self.tokenizer.batch_encode_plus([[c, ns]], add_special_tokens=True, padding='max_length', max_length=self.max_src_len, truncation=True, return_tensors='pt')
        ns, s = record[0], record[1]
        
        src_tokens = self.tokenizer.batch_encode_plus([ns], add_special_tokens=True, padding='max_length', max_length=self.max_src_len, truncation=True, return_tensors='pt')
        tgt_tokens = self.tokenizer.batch_encode_plus([s], add_special_tokens=True, padding='max_length', max_length=self.max_tgt_len, truncation=True, return_tensors='pt')
        
        encoder_input_id, encoder_attention_mask = src_tokens['input_ids'].squeeze(), src_tokens['attention_mask'].squeeze()
        decoder_input_id, decoder_attention_mask = tgt_tokens['input_ids'].squeeze(), tgt_tokens['attention_mask'].squeeze()

        labels = np.array(tgt_tokens['input_ids'])[:,1:]
        labels[np.array(tgt_tokens['input_ids'])[:, 1:] == self.tokenizer.pad_token_id] = -100
        labels = np.append(labels, [-100])
        
        return {'input_ids': np.array(encoder_input_id, dtype=np.int_),
                'attention_mask': np.array(encoder_attention_mask, dtype=np.float_),
                'decoder_input_ids': np.array(decoder_input_id, dtype=np.int_),
                'decoder_attention_mask': np.array(decoder_attention_mask, dtype=np.float_),
                'labels': np.array(labels, dtype=np.int_)}


class SARCDataModule(pl.LightningDataModule):
    def __init__(self, train_file, 
                 valid_file,
                 test_file, 
                 max_src_len=104,
                 max_tgt_len=42,
                 batch_size=32,
                 num_workers=5):
        super().__init__()
        self.batch_size = batch_size
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.train_file_path = train_file
        self.valid_file_path = valid_file
        self.test_file_path = test_file
        self.num_workers = num_workers

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--num_workers',
                            type=int,
                            default=5,
                            help='num of worker for dataloader')
        return parser

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):
        # split dataset
        self.train = SARCDataset(self.train_file_path,
                                 self.max_src_len,
                                 self.max_tgt_len)
        
        self.valid = SARCDataset(self.valid_file_path,
                                 self.max_src_len,
                                 self.max_tgt_len)

        self.test = SARCDataset(self.test_file_path,
                                self.max_src_len,
                                self.max_tgt_len)

    def train_dataloader(self):
        train = DataLoader(self.train,
                           batch_size=self.batch_size,
                           num_workers=self.num_workers, shuffle=True)
        return train

    def val_dataloader(self):
        val = DataLoader(self.valid,
                         batch_size=self.batch_size,
                         num_workers=self.num_workers, shuffle=False)
        return val

    def test_dataloader(self):
        test = DataLoader(self.test,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False)
        return test


class Base(pl.LightningModule):
    def __init__(self, hparams, **kwargs) -> None:
        super(Base, self).__init__()
        self.save_hyperparameters(hparams)

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)

        parser.add_argument('--batch-size',
                            type=int,
                            default=14,
                            help='batch size for training (default: 96)')

        parser.add_argument('--lr',
                            type=float,
                            default=2e-5,
                            help='The initial learning rate')

        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.1,
                            help='warmup ratio')

        parser.add_argument('--model_path',
                            type=str,
                            default=None,
                            help='bart model path')
        return parser

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, correct_bias=False)
        # warm up lr
        num_workers = (self.hparams.gpus if self.hparams.gpus is not None else 1) * (self.hparams.num_nodes if self.hparams.num_nodes is not None else 1)
        data_len = len(self.train_dataloader().dataset)
        logging.info(f'number of workers {num_workers}, data length {data_len}')
        num_train_steps = int(data_len / (self.hparams.batch_size * num_workers) * self.hparams.max_epochs)
        logging.info(f'num_train_steps : {num_train_steps}')
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        logging.info(f'num_warmup_steps : {num_warmup_steps}')
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]


class BARTConditionalGeneration(Base):
    def __init__(self, hparams, **kwargs):
        super(BARTConditionalGeneration, self).__init__(hparams, **kwargs)
        self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        self.model.train()
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.sep_token = '<sep>'
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    def forward(self, inputs):
        return self.model(input_ids=inputs['input_ids'],
                          attention_mask=inputs['attention_mask'],
                          decoder_input_ids=inputs['decoder_input_ids'],
                          decoder_attention_mask=inputs['decoder_attention_mask'],
                          labels=inputs['labels'], return_dict=True)

    def training_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs.loss
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        '''
        print(batch['input_ids'].shape)
        print(batch['attention_mask'].shape)
        print(batch['decoder_input_ids'].shape)
        print(batch['decoder_attention_mask'].shape)
        print(batch['labels'].shape)
        '''
        outs = self(batch)
        loss = outs.loss
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
                    
    def test(self, src, tgt):
        #input_ids =  [self.tokenizer.bos_token_id] + self.tokenizer.encode(c) + [self.tokenizer.sep_token_id] + self.tokenizer.encode(ns) + [self.tokenizer.eos_token_id]
        
        res_ids = self.model.generate(src['input_ids'],
                                            max_length=self.hparams.max_tgt_len,
                                            num_beams=5,
                                            eos_token_id=self.tokenizer.eos_token_id,
                                            bad_words_ids=[[self.tokenizer.unk_token_id]])        
        a = self.tokenizer.batch_decode(res_ids.tolist())[0]
        return a.replace('<s>', '').replace('</s>', ''), tgt


if __name__ == '__main__':
    parser = Base.add_model_specific_args(parser)
    parser = ArgsBase.add_model_specific_args(parser)
    parser = SARCDataModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    logging.info(args)

    model = BARTConditionalGeneration(args)

    dm = SARCDataModule(args.train_file,
                        args.valid_file,
                        args.test_file,
                        max_src_len=args.max_src_len,
                        max_tgt_len=args.max_tgt_len,
                        num_workers=args.num_workers)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss',
                                                       dirpath=args.default_root_dir,
                                                       filename='./save/withc_{epoch:02d}_{val_loss:.3f}',
                                                       verbose=True,
                                                       save_last=True,
                                                       mode='min',
                                                       save_top_k=-1)
    
    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(args.default_root_dir, 'tb_logs'))
    lr_logger = pl.callbacks.LearningRateMonitor()
    trainer = pl.Trainer.from_argparse_args(args, logger=tb_logger,
                                            callbacks=[checkpoint_callback, lr_logger])
    trainer.fit(model, dm)
