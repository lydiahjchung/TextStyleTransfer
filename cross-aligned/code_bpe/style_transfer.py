import logging
import os
# from vocab import Vocabulary, build_vocab
from file_io import load_sent, write_sent, load_sent_csvgz
from options import load_arguments
import sys
import time
import math
import random
from datetime import datetime
import utils
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from pathlib import Path
import sentencepiece as spm
from model import Model
from torch.utils.tensorboard import SummaryWriter
import re

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def get_model(args, logger, sp):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#:"+str(args.cuda_device) if torch.cuda.is_available() else "cpu")
    # print("vocab size: ", vocab.size)

    # logger.info("vocab size: "+str(vocab.size))
    model = Model(args, sp.get_piece_size(), args.dim_emb, args.dim_z, 
    args.dim_z+args.dim_y, sp.get_piece_size(), device, logger)
    return model

def run_model(args):
    time = datetime.now().timestamp()
    train_filename = "data/sarc/sarc.train"
    sp_model_path = "tmp/sarc_bpe"

    sp = spm.SentencePieceProcessor()
    
    #####   data preparation   #####
    if args.train:
        
        logger = utils.init_logging(args, time)
        
        print("args: ", args)
        logger.info("args: "+str(args))
        no_of_epochs = args.max_epochs
        train0 = load_sent(args.train + '.0', args.max_train_size)
        train1 = load_sent(args.train + '.1', args.max_train_size)

        #train0, train1 = load_sent_csvgz(args.train, args.max_train_size)
        # if not os.path.isfile(train_filename):
        with open(train_filename, "w") as f:
            for sent in train0+train1:
                f.write(" ".join(sent)+"\n")
    
    
        # if not os.path.isfile(train_filename+".1"):
        #     with open(train_filename+".1", "w") as f:
        #         for sent in train1:
        #             f.write(" ".join(sent)+"\n")
        print('#sents of training file 0:', len(train0))
        print('#sents of training file 1:', len(train1))

        logger.info('#sents of training file 0: ' + str(len(train0)))
        logger.info('#sents of training file 1: ' + str(len(train1)))

        # if not os.path.isfile(args.vocab):
        #     build_vocab(train0 + train1, args.vocab)
    # if not os.path.isfile(sp_model_path+".model") or not os.path.isfile(sp_model_path+".vocab"):        
    if args.train:
        spm.SentencePieceTrainer.Train('--input='+train_filename+' --model_prefix='+sp_model_path+' \
            --vocab_size=10000 --hard_vocab_limit=false --bos_piece=<go> --eos_piece=<eos> --pad_id=0 \
                --bos_id=1 --eos_id=2 --unk_id=3 --user_defined_symbols=<url>,<at>,<hashtag>')        
    
    sp.Load(sp_model_path+".model")
    # vocab = Vocabulary(args.vocab, args.embedding, args.dim_emb)
    
    dev0 = []
    dev1 = []
    
    if args.dev:
        dev0 = load_sent(args.dev + '.0')
        dev1 = load_sent(args.dev + '.1')
    
    if args.predict:
        if args.model_path:
            # logger.info("Predicting a sample input\n---------------------\n")
            model = torch.load(args.model_path)
            model.training = False
            output = utils.predict(model, args.predict, args.target_sentiment, sp, args.beam)
            # output = output.replace(" ","")
            # output_new = ""      
            # # output = re.sub(r"(\s\s+)", " ", output)
            # for val in output:
            #     if val == "  ":
            #         output_new += " "
            #     elif val == " ":
            #         pass
            #     else:
            #         output_new += val
            print(f"Input given: {args.predict} \nTarget sentiment: {args.target_sentiment} \nTranslated output: {output}")
            # logger.info(f"Input given: {args.predict} \nTarget sentiment: {args.target_sentiment} \nTranslated output: {output}")
    if args.test:
        file0 = open(args.test+".0", "r")
        file1 = open(args.test+".1", "r")
        saves_path = os.path.join(args.saves_path, utils.get_filename(args, time, "model"))
        Path(saves_path).mkdir(parents=True, exist_ok=True)
        out_file_0 = open(os.path.join(saves_path, "test_outputs_neg_to_pos"), "w")
        out_file_1 = open(os.path.join(saves_path, "test_outputs_pos_to_neg"), "w")
        model = torch.load(args.model_path)
        model.training = False

        test_neg = file0.readlines()
        for line in test_neg:
            output = utils.predict(model, line, 1, sp, args.beam)
            # out_file_0.write(output+"\n")
        print("second")
        test_pos = file1.readlines()
        for line in test_pos:
            output = utils.predict(model, line, 0, sp, args.beam)
            out_file_1.write(output+"\n")

        # test0 = load_sent(args.test + '.0')
        # test1 = load_sent(args.test + '.1')
        # if args.model_path:
        #     saves_path = os.path.join(args.saves_path, utils.get_filename(args, time, "model"))
        #     Path(saves_path).mkdir(parents=True, exist_ok=True)
        #     model = torch.load(args.model_path)
        #     model.training = False
        #     batches0, batches1, _, _ = utils.get_batches(test0, test1, model.vocab.word2id, model.args.batch_size)

        #     output_file_0 = open(os.path.join(saves_path, "test_outputs_neg_to_pos"), "w")
        #     output_file_1 = open(os.path.join(saves_path, "test_outputs_pos_to_neg"), "w")

        #     for batch0, batch1 in zip(batches0, batches1):
        #         batch0 = batch0["enc_inputs"]
        #         batch1 = batch1["enc_inputs"]
        #         test_outputs_0 = utils.predict_batch(model, batch0, sentiment=1, beam_size=args.beam, plain_format=True)
        #         test_outputs_1 = utils.predict_batch(model, batch1, sentiment=0, beam_size=args.beam, plain_format=True)
        #         output_file_0.write('\n'.join(test_outputs_0) + '\n')
        #         output_file_1.write('\n'.join(test_outputs_1) + '\n')
                
    if args.train:
        summ_filename = 'runs/cross-alignment/'+utils.get_filename(args, time, "summary")
        writer = SummaryWriter(summ_filename)

        model = get_model(args, logger, sp)
        model.train_max_epochs(args, train0, train1, dev0, dev1, no_of_epochs, writer, time, sp,
        save_epochs_flag=True)
        
if __name__ == '__main__':
    args = load_arguments()
    # batch_sizes = [64, 256, 512]
    # for batch_size in batch_sizes:
    #     print(f"batch size: {batch_size}")
    #     args.batch_size = batch_size
    run_model(args)
