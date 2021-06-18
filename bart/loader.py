import torch
import numpy as np
from torch.utils.data import Dataset

def parse_data(filepath, type_):
    data = []
    
    with open(f'{filepath}_{type_}.tsv') as f:
        sents = f.readlines()

    for each in sents:
        sent = each.strip().split('\t')
        data.append(sent)

    return data

class SARCDataset(Dataset):
    def __init__(self, filepath, type_, tokenizer, max_input, max_output):
        self.dataset = parse_data(filepath, type_)
        self.tokenizer = tokenizer
        self.max_input = max_input
        self.max_output = max_output
    
    def __len__(self):
        return len(self.dataset)

    def convert(self, batch):
        c, ns, s = batch[0], batch[1], batch[2]

        source = self.tokenizer.batch_encode_plus([[c, ns]], add_special_tokens=True, padding='max_length', max_length=self.max_input, truncation=True, return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([s], add_special_tokens=True, padding='max_length', max_length=self.max_output, truncation=True, return_tensors='pt')

        return source, target

    def __getitem__(self, index):
        source, target = self.convert(self.dataset[index])

        src_ids = source["input_ids"].squeeze()
        src_mask = source["attention_mask"].squeeze()

        tgt_ids = target["input_ids"].squeeze()
        tgt_mask = target["attention_mask"].squeeze()

        labels = tgt_ids[1:].clone().detach()
        labels[tgt_ids[1:] == self.tokenizer.pad_token_id] = -100
        labels = torch.from_numpy(np.append(labels.cpu().numpy(), [-100]))


        return {"src_ids": src_ids, "src_mask": src_mask, "tgt_ids": tgt_ids, "tgt_mask": tgt_mask, "labels": labels}

if __name__ == "__main__":
    pairs = tokenizer("Hello world", "Bye world", add_special_tokens=True)
    print(pairs)
    print(tokenizer.get_special_tokens_mask(pairs, already_has_special_tokens=True))
    print(tokenizer.encode("Hello world", "Bye world", add_special_tokens=True))
    print(tokenizer.encode("Hello world", "Bye world"))
    
    print(parse_data('./data/main', 'test')[100])
    data = SARCDataset('./data/main', 'test', 128, 64)

    test_dataloader = DataLoader(data, batch_size=32, shuffle=False)
    print(len(test_dataloader))
    
    for step, batch in enumerate(test_dataloader):
        print(step)
        print("*******")
        print(batch)
        print()


    data = data[100]
    print()
    print("Shape", data['src_ids'].shape)
    print()
    print("Decode Input", tokenizer.decode(data['src_ids']))
    print()
    print("Decode Output", tokenizer.decode(data["tgt_ids"]))
    print(data["tgt_mask"])
