import torch
import pytorch_lightning as pl
from main import BARTConditionalGeneration
from transformers import BartForConditionalGeneration, BartTokenizer

def parse_test(filepath):
    data = []
    with open(filepath) as f:
        lines = f.readlines()
        
    for line in lines:
        data.append(line.strip().split('|'))
        
    return data

if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = BARTConditionalGeneration.load_from_checkpoint('./logs/save/withc_epoch=04_val_loss=1.981.ckpt').to(device)
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    
    test_data = parse_test('./data/withc_test.txt')
    test_out = open('./logs/save/withc_1_981_test_out.txt', 'w')
    
    model.eval()
    
    count = 0
    for each in test_data:
        ns, s = each[0], each[1]
        src_tokens = tokenizer.batch_encode_plus([ns], add_special_tokens=True, padding='max_length', max_length=40, truncation=True, return_tensors='pt').to(device)
        pred, tgt = model.test(src_tokens, s)
        test_out.write(f'{ns}|{tgt}|{pred}\n')
        print(count)
        count += 1
    
    test_out.close()
