from transformers import BartTokenizer
import numpy as np

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

with open('cns_test.ns') as f:
    nsdata = f.readlines()

with open('cs_test.s') as f:
    sdata = f.readlines()

wrt = open('withc_test.txt', 'w')

for ns, s in zip(nsdata, sdata):
    wrt.write(f'{ns.strip()} | {s.strip()} \n')

wrt.close()
