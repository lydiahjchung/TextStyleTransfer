import time
import numpy as np
import torchtext
from torchtext import data

from utils import tensor2text

import torch


class DatasetIterator(object):
    def __init__(self, ns_iter, s_iter):
        self.ns_iter = ns_iter
        self.s_iter = s_iter

    def __iter__(self):
        for batch_ns, batch_s in zip(iter(self.ns_iter), iter(self.s_iter)):
            if batch_ns.text.size(0) == batch_s.text.size(0):
                yield batch_ns.text, batch_s.text

def load_dataset(config, train_ns='cns_train.ns', train_s='cs_train.s',
                 dev_ns='cns_valid.ns', dev_s='cs_valid.s',
                 test_ns='cns_test.ns', test_s='cs_test.s'):

    root = config.data_path
    TEXT = data.Field(batch_first=True, eos_token='<eos>')
    #TEXT = data.Field(batch_first=True, eos_token='<eos>', pad_token='<pad>', fix_length=60)

    dataset_fn = lambda name: data.TabularDataset(
        path=root + name,
        format='tsv',
        fields=[('text', TEXT)]
    )

    train_ns_set, train_s_set = map(dataset_fn, [train_ns, train_s])
    dev_ns_set, dev_s_set = map(dataset_fn, [dev_ns, dev_s])
    test_ns_set, test_s_set = map(dataset_fn, [test_ns, test_s])

    TEXT.build_vocab(train_ns_set, train_s_set, min_freq=3)#config.min_freq)

    # if config.load_pretrained_embed:
    #     start = time.time()
        
    #     vectors=torchtext.vocab.GloVe('6B', dim=config.embed_size, cache=config.pretrained_embed_path)
    #     TEXT.vocab.set_vectors(vectors.stoi, vectors.vectors, vectors.dim)
    #     print('vectors', TEXT.vocab.vectors.size())
        
    #     print('load embedding took {:.2f} s.'.format(time.time() - start))

    vocab = TEXT.vocab
        
    dataiter_fn = lambda dataset, train: data.BucketIterator(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=train,
        repeat=train,
        sort_key=lambda x: len(x.text),
        sort_within_batch=False,
        device=config.device
    )

    train_ns_iter, train_s_iter = map(lambda x: dataiter_fn(x, True), [train_ns_set, train_s_set])
    dev_ns_iter, dev_s_iter = map(lambda x: dataiter_fn(x, False), [dev_ns_set, dev_s_set])
    test_ns_iter, test_s_iter = map(lambda x: dataiter_fn(x, False), [test_ns_set, test_ns_set])
    train_iters = DatasetIterator(train_ns_iter, train_s_iter)
    dev_iters = DatasetIterator(dev_ns_iter, dev_s_iter)
    test_iters = DatasetIterator(test_ns_iter, test_s_iter)
    return train_iters, dev_iters, test_iters, vocab


if __name__ == '__main__':
    train_iter, dev_iter, _, vocab = load_dataset('./data/sarc/')

    for i, batch in enumerate(dev_iter):
        print(i)

    print("end")
