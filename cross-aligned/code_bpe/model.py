import logging
import os
# from vocab import Vocabulary, build_vocab
from file_io import load_sent, write_sent
from options import load_arguments
import sys
import time
import math
import random
from datetime import datetime
import _pickle as pickle
import utils
from torch import autograd

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from encoder import Encoder
from generator import Generator
from discriminator import Discriminator
from pathlib import Path
import math

class Model(nn.Module):
    def __init__(self, args, input_size_enc, embedding_size_enc, hidden_size_enc, 
    hidden_size_gen, output_size_gen, device, logger,
    pretrain_flag=False, autoencoder_train_flag=True, use_saved_models=True):
        super(Model, self).__init__()
        self.input_size_enc = input_size_enc
        self.hidden_size_enc = hidden_size_enc
        self.embedding_size_enc = embedding_size_enc
        self.hidden_size_gen = hidden_size_gen
        self.output_size_gen = output_size_gen
        self.args = args
        self.dropout_p = self.args.dropout_keep_prob  
        self.device = device
        # self.vocab = vocab
        self.dropout = nn.Dropout(self.dropout_p)
        

        # Loading the components of the pretrained model and assigning it to the current model
        if self.args.load_model == True:
            if self.args.model_path:
                saved_model = torch.load(args.model_path)
                self.generator = saved_model.generator.to(self.device)
                self.encoder = saved_model.encoder.to(self.device)    
                self.discriminator1 = saved_model.discriminator1.to(self.device)
                self.discriminator2 = saved_model.discriminator2.to(self.device)
            else:
                if self.args.autoencoder_path:
                    print("AE pretrained")
                    autoencoder_model = torch.load(self.args.autoencoder_path)
                    self.generator = autoencoder_model.generator.to(self.device)
                    self.encoder = autoencoder_model.encoder.to(self.device)    
                if self.args.discriminator_path:
                    print("DISC pretrained")
                    discriminator_model = torch.load(self.args.discriminator_path)
                    self.discriminator1 = discriminator_model.discriminator1.to(self.device)
                    self.discriminator2 = discriminator_model.discriminator2.to(self.device)
        else:
            self.generator = Generator(self.embedding_size_enc, self.hidden_size_gen, self.output_size_gen, self.dropout_p).to(self.device)
            self.encoder = Encoder(self.input_size_enc, self.embedding_size_enc, self.hidden_size_enc, self.dropout_p).to(self.device)
            self.discriminator1 = Discriminator(args, self.device, self.hidden_size_gen, 1, self.dropout_p).to(self.device)
            self.discriminator2 = Discriminator(args, self.device, self.hidden_size_gen, 1, self.dropout_p).to(self.device)

        self.softmax = torch.nn.Softmax(dim=1)
        # self.EOS_token = 2
        # self.GO_token = 1
        self.k = 5
        self.gamma = 0.001
        self.logger = logger
        # self.lambda_val = lambda_val
        self.beta1, self.beta2 = 0.5, 0.999
        self.grad_clip = 20.0
        

    
    def get_latent_reps(self, input_data):
        '''
        Summary: Takes x and finds the latent representation z
        Parameters:
        input_data: (length, batchsize)
        '''

        if torch.cuda.is_available():
            input_data = input_data.to(device=self.device)
        
        input_length = input_data.shape[0]
        batch_size = input_data.shape[1]

        encoder_hidden = self.encoder.initHidden(device=self.device)
        encoder_hidden = encoder_hidden.repeat(1,batch_size,1)
        input_tensor = self.encoder.embedding(input_data)
        input_tensor = self.dropout(input_tensor)
        # input_tensor = nn.Dropout(0.2)(input_tensor)

        if torch.cuda.is_available():
            input_tensor = input_tensor.to(device=self.device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[ei,:,:], encoder_hidden)

        return encoder_hidden

    def gumbel_softmax(self, logits, eps=1e-20):
        U = torch.rand_like(logits)
        G = -torch.log(-torch.log(U + eps) + eps)
        result = self.softmax((logits + G) / self.gamma)
        return result

    # Greedy search prediction used for testing some sample inputs
    def predict_greedy_search(self, input_data, target_sentiment, sp):
        latent_z = self.get_latent_reps(input_data)
        input_length = input_data.size()[0]
        if target_sentiment == 1:
            latent_z = torch.cat((latent_z, torch.ones([1,latent_z.shape[1],self.args.dim_y],
                            dtype=torch.float,device=self.device)), dim=2)
        else:
            latent_z = torch.cat((latent_z, torch.zeros([1,latent_z.shape[1],self.args.dim_y],
                            dtype=torch.float,device=self.device)), dim=2)
        
        gen_input = torch.tensor([sp.eos_id()], device=self.device)
        gen_hidden = latent_z

        outputs = []
        gen_output = torch.zeros(self.output_size_gen, device=self.device)
        
        count = 0
        while torch.argmax(gen_output) != sp.eos_id():
            if count >= input_length*2:
                break            
            gen_input = gen_input.unsqueeze(0)
            gen_input = gen_input.unsqueeze(2)
            gen_input = self.encoder.embedding(gen_input).squeeze(2)
            gen_input = self.dropout(gen_input)

            gen_output, gen_hidden = self.generator(
                gen_input, gen_hidden)
            gen_input = torch.argmax(gen_output, dim=1)            
            # outputs.append(self.vocab.id2word[gen_input])
            outputs.append(gen_input)
            count += 1
        outputs = [element.item() for element in outputs]
        # print("Output before decoding: ", outputs)
        outputs = sp.DecodeIds(outputs)
        print(outputs)
        return outputs

    # # Beam search prediction used for testing some sample inputs
    # def predict_beam_search(self, input_data, target_sentiment, beam_size, sp):
    #     latent_z = self.get_latent_reps(input_data)
    #     input_length = input_data.size()[0]
    #     if target_sentiment == 1:
    #         latent_z = torch.cat((latent_z, torch.ones([1,latent_z.shape[1],self.args.dim_y],
    #                         dtype=torch.float,device=self.device)), dim=2)
    #     else:
    #         latent_z = torch.cat((latent_z, torch.zeros([1,latent_z.shape[1],self.args.dim_y],
    #                         dtype=torch.float,device=self.device)), dim=2)
        
    #     gen_input = torch.tensor([self.GO_token], device=self.device)
    #     gen_hidden = latent_z
    #     outputs = []
    #     gen_output = torch.zeros(self.output_size_gen, device=self.device)
    #     result = []
    #     count = 0
    #     softmax = torch.nn.Softmax(dim=1)
    #     while torch.argmax(gen_output) != self.EOS_token:
    #         if len(result) == k or count >= input_length*2:
    #             break        
    #         if count == 0:
    #             gen_input = gen_input.unsqueeze(0)
    #             gen_input = gen_input.unsqueeze(2)
    #             gen_input = self.encoder.embedding(gen_input).squeeze(2)
    #             gen_input = self.dropout(gen_input)

    #             gen_output, gen_hidden = self.generator(
    #                 gen_input, gen_hidden)
    #             gen_output = softmax(gen_output)
    #             topv, topi = gen_output.topk(k)
    #             for i in range(topv.size()[1]):                
    #                 outputs.append({
    #                     "sequence": [topi[:,i].item()],
    #                     "score": np.log(topv[:,i].item())
    #                 })
    #                 if topi[:,i].item() == self.EOS_token:
    #                     result.append(outputs[-1])
    #                     del outputs[-1]
                
    #         else:
    #             outputs_old = outputs.copy()
    #             outputs = []
    #             for val in outputs_old:
    #                 gen_input = torch.tensor(val["sequence"][-1], device=self.device).view(1,1,1)
    #                 gen_input = self.encoder.embedding(gen_input).squeeze(2)
    #                 gen_input = self.dropout(gen_input)

    #                 gen_output, gen_hidden = self.generator(
    #                     gen_input, gen_hidden)
                    
    #                 topv, topi = gen_output.topk(k)

    #                 for i in range(topv.size()[1]):                          
    #                     outputs.append({
    #                         "sequence": val["sequence"] + [topi[:,i].item()],
    #                         "score": np.log(val["score"])+np.log(topv[:,i].item())
    #                     })
    #                     if topi[:,i].item() == self.EOS_token:
    #                         result.append(outputs[-1])
    #                         del outputs[-1]
                            
    #             outputs = sorted(outputs, key = lambda i: i['score'], reverse = True)
    #             outputs = outputs[:k]
    #         count += 1
    #     result = sorted(result, key = lambda i: i['score'], reverse = True)
        
    #     for output in result:
    #         # output["sentence"]=[self.vocab.id2word[val] for val in output["sequence"]]
    #         output["sentence"]=sp.DecodeIds(output["sequence"])
    #     return result


    # Generate X using [y,z] where y is style attribute and z is the latent content obtained from the encoder
    def generate_x(self, hidden_vec, true_outputs, criterion, sp, teacher_forcing=False):
        gen = self.generator
        
        batch_size = hidden_vec.shape[1]
        # gen_input = torch.tensor([self.GO_token], device=self.device)
        gen_input = torch.tensor([sp.bos_id()], device=self.device)
        gen_input = gen_input.repeat(batch_size)
        gen_hidden = hidden_vec

        gen_output = torch.zeros(self.output_size_gen, device=self.device)
        input_length = true_outputs.shape[0]
        loss = 0
        losses = torch.zeros(input_length, batch_size, device=self.device)
        gen_hid_states = torch.zeros(input_length, gen_hidden.shape[0], gen_hidden.shape[1], gen_hidden.shape[2], device=self.device)

        gen_input = gen_input.unsqueeze(0)
        gen_input = gen_input.unsqueeze(2)
        gen_input_new = self.encoder.embedding(gen_input).squeeze(2)
        gen_input = gen_input_new

        for i in range(input_length): 
            gen_output, gen_hidden = gen(
                gen_input, gen_hidden)            
            
            gen_output = gen_output+1e-8            
            gen_hid_states[i] = gen_hidden

            if teacher_forcing == True:
                loss = criterion(gen_output, true_outputs[i,:])
                losses[i] = loss
                gen_input = true_outputs[i,:]
                gen_input = gen_input.unsqueeze(0)
                gen_input = gen_input.unsqueeze(2)
                gen_input = self.encoder.embedding(gen_input).squeeze(2)
            else:                
                gen_input = self.gumbel_softmax(gen_output)
                gen_input = gen_input.unsqueeze(0)
                gen_input = torch.matmul(gen_input, self.encoder.embedding.weight)
        
        avg_loss = 0
        avg_loss = torch.mean(losses)
        return gen_hid_states, avg_loss

    
    def train_one_batch(self, training_data, target, sentiment, sp):
        '''
        Train one batch of positive or negative samples
        Returns h1 and h1~ or h2 and h2~
        '''
        # criterion = nn.CrossEntropyLoss(ignore_index=self.vocab.word2id['<pad>'], reduction='mean', size_average=True)
        criterion = nn.CrossEntropyLoss(ignore_index=sp.pad_id(), reduction='mean', size_average=True)
        latent_z = self.get_latent_reps(training_data)
        latent_z_original = []
        latent_z_translated = []
        if sentiment == 0:
            latent_z_original = torch.cat((latent_z, torch.zeros([1,latent_z.shape[1],self.args.dim_y],
                            dtype=torch.float,device=self.device)), 2)
            latent_z_translated = torch.cat((latent_z, torch.ones([1,latent_z.shape[1],self.args.dim_y],
                            dtype=torch.float,device=self.device)), 2)
        else:
            latent_z_original = torch.cat((latent_z, torch.ones([1,latent_z.shape[1],self.args.dim_y],
                            dtype=torch.float,device=self.device)), 2)
            latent_z_translated = torch.cat((latent_z, torch.zeros([1,latent_z.shape[1],self.args.dim_y],
                            dtype=torch.float,device=self.device)), 2)
        hidden_states_original, loss_original = self.generate_x(latent_z_original, target, criterion, sp, teacher_forcing=True)
        hidden_states_translated, _ = self.generate_x(latent_z_translated, target, criterion, sp, teacher_forcing=False)

        avg_loss = torch.mean(loss_original)

        return hidden_states_original, hidden_states_translated, avg_loss

    def compute_adv_loss(self, adv_output, adv_output_tilde, eps = 1e-8):
        return -torch.mean(torch.log(adv_output+eps))-torch.mean(torch.log(1-adv_output_tilde+eps))                           

    # Trains one set of positive and negative samples batch
    def train_util(self, batch0, batch1, epoch, writer, sp):
        correct_count = 0
        batch0_input = batch0["enc_inputs"]
        batch0_input = torch.tensor(batch0_input, device=self.device)
        batch0_input = batch0_input.t()
        target_batch0 = torch.tensor(batch0["targets"], device=self.device).t()
        # batch0_input = torch.cat((batch0_input, torch.zeros([batch0_input.shape[0],1],dtype=torch.long, device=self.device)), dim=1)

        batch1_input = batch1["enc_inputs"]                
        batch1_input = torch.tensor(batch1_input, device=self.device)
        batch1_input = batch1_input.t()
        target_batch1 = torch.tensor(batch1["targets"], device=self.device).t()
        # batch1_input = torch.cat((batch1_input, torch.ones([batch1_input.shape[0],1],dtype=torch.long,device=self.device)), dim=1)
        
        # 0 represents that the sample/batch has negative sentiment
        # 1 represents that the sample/batch has positive sentiment
        h1, h1_tilde, loss0 = self.train_one_batch(batch0_input, target_batch0, 0, sp)
        h2, h2_tilde, loss1 = self.train_one_batch(batch1_input, target_batch1, 1, sp)
        
        # Permuting so that the input for discriminator is in the 
        # format (Batch,Channels,H,W) (changed from (H,Channels,Batch,W))
        adv1_output = self.discriminator1(h1.permute(2,1,0,3))        
        adv1_output_tilde = self.discriminator1(h2_tilde.permute(2,1,0,3))

        adv2_output = self.discriminator2(h2.permute(2,1,0,3))
        adv2_output_tilde = self.discriminator2(h1_tilde.permute(2,1,0,3)) 

        correct_count += torch.sum((adv1_output>0.5) == True)
        correct_count += torch.sum((adv2_output>0.5) == True)
        correct_count += torch.sum((adv1_output_tilde<0.5) == True)
        correct_count += torch.sum((adv2_output_tilde<0.5) == True)
        correct_count = correct_count.item()
        correct_count /= (adv1_output.size()[0]+adv1_output_tilde.size()[0]+adv2_output.size()[0]+adv2_output_tilde.size()[0])

        # writer.add_scalars("Adv_Outputs", {
        #     "adv1_output": adv1_output[0].item(),
        #     "adv1_output_tilde": adv1_output_tilde[0].item(),
        #     "adv2_output": adv2_output[0].item(),
        #     "adv2_output_tilde": adv2_output_tilde[0].item()
        # }, epoch)

        loss_adv1 = self.compute_adv_loss(adv1_output, adv1_output_tilde)
        loss_adv2 = self.compute_adv_loss(adv2_output, adv2_output_tilde)

        loss_reconstruction = (loss0+loss1)/2
        loss_enc_gen = loss_reconstruction - self.args.rho*(loss_adv1+loss_adv2)

        return loss_adv1, loss_adv2, loss_enc_gen, loss_reconstruction, correct_count

    def train_max_epochs(self, args, train0, train1, dev0, dev1, no_of_epochs, writer, time, sp, save_epochs_flag=False, 
        save_batch_flag=False, save_batch=5):
        print("No of epochs: ", no_of_epochs)
        self.train()
        enc_optim = optim.AdamW(self.encoder.parameters(), lr=args.learning_rate, betas=(self.beta1, self.beta2))
        gen_optim = optim.AdamW(self.generator.parameters(), lr=args.learning_rate, betas=(self.beta1, self.beta2))
        discrim1_optim = optim.AdamW(self.discriminator1.parameters(), lr=args.learning_rate, betas=(self.beta1, self.beta2))
        discrim2_optim = optim.AdamW(self.discriminator2.parameters(), lr=args.learning_rate, betas=(self.beta1, self.beta2))
        
        Path(args.saves_path).mkdir(parents=True, exist_ok=True)        
        saves_path = os.path.join(args.saves_path, utils.get_filename(args, time, "model"))
        Path(saves_path).mkdir(parents=True, exist_ok=True)
        flag = True
        with autograd.detect_anomaly():
            for epoch in range(no_of_epochs):
                random.shuffle(train0)
                random.shuffle(train1)
                # batches0, batches1, _1, _2 = utils.get_batches(train0, train1, vocab.word2id,
                # args.batch_size, sp, noisy=True)
                batches0, batches1, _1, _2 = utils.get_batches_bpe(train0, train1, 
                args.batch_size, sp, noisy=True)
                
                random.shuffle(batches0)
                random.shuffle(batches1)
                print("Epoch: ", epoch)
                self.logger.info("Epoch: "+str(epoch))

                losses_enc_gen = []
                losses_adv1 = []
                losses_adv2 = []
                rec_losses = []

                losses_enc_gen_dev = []
                losses_adv1_dev = []
                losses_adv2_dev = []
                rec_losses_dev = []

                flag = True
                disc_tot_accuracy = 0
                for batch0, batch1 in tqdm(zip(batches0, batches1), total=len(batches0)):
                    enc_optim.zero_grad()
                    gen_optim.zero_grad()
                    discrim1_optim.zero_grad()
                    discrim2_optim.zero_grad()

                    loss_adv1, loss_adv2, loss_enc_gen, loss_reconstruction, disc_batch_acc = self.train_util(batch0, batch1, epoch, writer, sp)
                    disc_tot_accuracy += disc_batch_acc

                    if self.args.pretrain_flag == True:
                        if self.args.autoencoder_pretrain_flag == True:
                            # Train only autoencoder part
                            # Doing backprop on loss_reconstruction (not loss_enc_gen) because I think we just need to 
                            # train the autoencoder to reconstruct the input sentence, as part of the pretraining. 
                            # As a result, the autoencoder will be good at its task of reconstructing the input sentence
                            loss_reconstruction.backward()
                            torch.nn.utils.clip_grad_value_(self.encoder.parameters(), self.grad_clip)
                            torch.nn.utils.clip_grad_value_(self.generator.parameters(), self.grad_clip)
                            enc_optim.step()
                            gen_optim.step()
                        else:
                            # Train only discriminators
                            loss_adv1.backward(retain_graph=True)
                            loss_adv2.backward()
                            torch.nn.utils.clip_grad_value_(self.discriminator1.parameters(), self.grad_clip)
                            torch.nn.utils.clip_grad_value_(self.discriminator2.parameters(), self.grad_clip)
                            discrim1_optim.step()
                            discrim2_optim.step()
                    else:                        
                        if flag == True:
                            loss_enc_gen.backward()
                            torch.nn.utils.clip_grad_value_(self.encoder.parameters(), self.grad_clip)
                            torch.nn.utils.clip_grad_value_(self.generator.parameters(), self.grad_clip)
                            enc_optim.step()
                            gen_optim.step()
                        else:
                            loss_adv1.backward(retain_graph=True)
                            loss_adv2.backward()
                            torch.nn.utils.clip_grad_value_(self.discriminator1.parameters(), self.grad_clip)
                            torch.nn.utils.clip_grad_value_(self.discriminator2.parameters(), self.grad_clip)
                            discrim1_optim.step()
                            discrim2_optim.step()
                        flag = not flag
                    
                    losses_enc_gen.append(loss_enc_gen.detach())
                    losses_adv1.append(loss_adv1.detach())
                    losses_adv2.append(loss_adv2.detach())
                    rec_losses.append(loss_reconstruction.detach())                

                if save_epochs_flag == True and (epoch%args.epochs_per_checkpoint == args.epochs_per_checkpoint-1 or epoch == no_of_epochs-1):
                    filename = os.path.join(saves_path, str(epoch+1)+"_epochs")
                    torch.save(self, filename)

                disc_tot_accuracy = 1.0*disc_tot_accuracy/len(batches0)

                print("Avg Reconstruction Loss: ", torch.mean(torch.tensor(rec_losses)))
                print("Avg Loss of Encoder-Generator: ", torch.mean(torch.tensor(losses_enc_gen)))
                print("Avg Loss of D1: ", torch.mean(torch.tensor(losses_adv1)))
                print("Avg Loss of D2: ", torch.mean(torch.tensor(losses_adv2)))

                self.logger.info("Avg Reconstruction Loss: " + str(torch.mean(torch.tensor(rec_losses))))
                self.logger.info("Avg Loss of Encoder-Generator: " + str(torch.mean(torch.tensor(losses_enc_gen))))
                self.logger.info("Avg Loss of D1: " + str(torch.mean(torch.tensor(losses_adv1))))
                self.logger.info("Avg Loss of D2: " + str(torch.mean(torch.tensor(losses_adv2))))
                self.logger.info("Discriminator accuracy: " + str(disc_tot_accuracy))

                writer.add_scalar("discriminator_acc", disc_tot_accuracy, epoch)
                writer.add_scalars('All_losses', {
                    'recon-loss': torch.mean(torch.tensor(rec_losses)),
                    'enc-gen': torch.mean(torch.tensor(losses_enc_gen)),
                    'D1': torch.mean(torch.tensor(losses_adv1)),
                    'D2': torch.mean(torch.tensor(losses_adv2))
                }, epoch)

                if self.args.pretrain_flag == True:
                    if self.args.autoencoder_pretrain_flag == True:
                        if torch.mean(torch.tensor(rec_losses)) < 0.1:
                            filename = os.path.join(saves_path, str(epoch+1)+"_epochs")
                            torch.save(self, filename)
                            break
                    else:
                        if disc_tot_accuracy >= 0.8:
                            filename = os.path.join(saves_path, str(epoch+1)+"_epochs")
                            torch.save(self, filename)
                            break

                if self.args.dev:
                    
                    batches0, batches1, _, _ = utils.get_batches_bpe(dev0, dev1, args.batch_size, sp, noisy=True)

                    random.shuffle(batches0)
                    random.shuffle(batches1)

                    for batch0, batch1 in zip(batches0, batches1):
                        loss_adv1, loss_adv2, loss_enc_gen, loss_reconstruction, disc_batch_dev_acc = self.train_util(batch0, batch1, epoch, writer)

                        losses_adv1_dev.append(loss_adv1.detach())
                        losses_adv2_dev.append(loss_adv2.detach())

                        losses_enc_gen_dev.append(loss_enc_gen.detach())
                        rec_losses_dev.append(loss_reconstruction.detach())

                    print("\nDev loss")
                    print("Avg Reconstruction Loss: ", torch.mean(torch.tensor(rec_losses_dev)))
                    print("Avg Loss of Encoder-Generator: ", torch.mean(torch.tensor(losses_enc_gen_dev)))
                    print("Avg Loss of D1: ", torch.mean(torch.tensor(losses_adv1_dev)))
                    print("Avg Loss of D2: ", torch.mean(torch.tensor(losses_adv2_dev)))

                    self.logger.info("\nDev loss")
                    self.logger.info("Avg Reconstruction Loss: " + str(torch.mean(torch.tensor(rec_losses_dev))))
                    self.logger.info("Avg Loss of Encoder-Generator: " + str(torch.mean(torch.tensor(losses_enc_gen_dev))))
                    self.logger.info("Avg Loss of D1: " + str(torch.mean(torch.tensor(losses_adv1_dev))))
                    self.logger.info("Avg Loss of D2: " + str(torch.mean(torch.tensor(losses_adv2_dev))))

                    writer.add_scalars('All_losses_dev', {
                        'recon-loss': torch.mean(torch.tensor(rec_losses)),
                        'enc-gen': torch.mean(torch.tensor(losses_enc_gen_dev)),
                        'D1': torch.mean(torch.tensor(losses_adv1_dev)),
                        'D2': torch.mean(torch.tensor(losses_adv2_dev))
                    }, epoch)
                
                print("---------\n")
                self.logger.info("---------\n")
        # torch.save(self, saves_path)