import time
import copy
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from loader import SARCDataset
from torch.utils.data import DataLoader
from utils import epoch_time
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup

def define_args(parser):
    parser.add_argument('--max_input', type=int, default=128)
    parser.add_argument('--max_output', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-05)
    parser.add_argument('--warmup', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--data', type=str, default='./data/main')
    parser.add_argument('--path', type=str, default='./save')
    args = parser.parse_args()
    return args

def train(model, iterator, tokenizer, optimizer):
    total_loss, iter_num, train_acc = 0, 0, 0
    model.train()

    for step, batch in enumerate(iterator):
        outputs = model(
                intput_ids=batch['src_ids'].to(device),
                attention_mask=batch['src_mask'].to(device),
                decoder_input_ids=batch['tgt_ids'].to(device),
                decoder_attention_mask=batch['tgt_mask'].to(device), 
                labels=batch['labels'].to(device), return_dict=True)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc =  accuracy(outputs.logits, batch["tgt_ids"], tokenizer.pad_token_id)

        total_loss += loss
        iter_num += 1
        train_acc += acc

    return total_loss.data.cpu().numpy() / iter_num, train_acc.data.cpu().numpy() / iter_num


def valid(model, iterator, tokenizer, optimizer):
    total_loss, iter_num, valid_acc = 0, 0, 0
    model.eval()

    for step, batch in enumerate(iterator):
        outputs = model(
                intput_ids=batch['src_ids'].to(device),
                attention_mask=batch['src_mask'].to(device),
                decoder_input_ids=batch['tgt_ids'].to(device),
                decoder_attention_mask=batch['tgt_mask'].to(device),
                labels=batch['labels'].to(device), return_dict=True)
        
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc =  accuracy(outputs.logits, batch["tgt_ids"], tokenizer.pad_token_id)
        
        total_loss += loss
        iter_num += 1
        valid_acc += acc

     return total_loss.data.cpu().numpy() / iter_num, valid_acc.data.cpu().numpy() / iter_num

def main(train_loader, valid_loader, test_loader, tokenizer):
    earlystop_check = 0

    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn').to(device)
    optimizer = AdamW(params=model.parameters(), lr=args.lr, correct_bias=False)
    
    best_valid_loss = float('inf')
    checkpoint_path = f'{args.path}/_check.pt'

    # training
    for epoch in range(args.num_epochs):
        start_time = time.time()

        train_loss, train_acc = train(model, train_loader, tokenizer, optimizer)
        valid_loss, valid_acc = valid(model, valid_loader, tokenizer, optimizer)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if earlystop_check == args.patience:
            print("\n==== EARLY STOPPIN G====\n")
            break

        if valid_loss < best_valid_loss:
            earlystop_check = 0
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f'\n ==== SAVED Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.4f} ==== \n')
        else:
            earlystop_check += 1

        print(f'\n ==== EPOCH: {epoch} | TIME: {epoch_mins}m {epoch_secs}s ====')
        print(f'\n==== TRAIN Loss: {train_loss} | Acc: {train_acc} ====')
        print(f'\n==== VALID Loss: {valid_loss} | Acc: {valid_acc} ====\n')

    # test
    test_loss, test_acc = test(model, test_loader, optimizer)
    print(f'\n==== TEST Loss: {test_loss} | Acc: {test_acc} ====\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = define_args(parser)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    
    train_dataset = SARCDataset(args.data, "train",  tokenizer, args.max_input, args.max_output)
    valid_dataset = SARCDataset(args.data, "valid",  tokenizer, args.max_input, args.max_output)
    test_dataset = SARCDataset(args.data, "test",  tokenizer, args.max_input, args.max_output)

    print(len(train_dataset))
    print(len(valid_dataset), len(test_dataset))

    train_loader = DataLoader(train_dataset, shuffle=False, batch_size=args.batch_size)
    valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)

    print(len(train_loader))
    print(len(valid_loader), len(test_loader))

    main(train_loader, valid_loader, test_loader)
