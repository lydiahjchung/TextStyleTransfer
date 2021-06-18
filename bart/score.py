import os
import nltk
from transformers import BartTokenizer

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

with open('./logs/save/withc_1_981_test_out.txt') as f:
    data = f.readlines()

def common_start(sa, sb):
    """ returns the longest common substring from the beginning of sa and sb """
    def _iter():
        for a, b in zip(sa, sb):
            if a == b:
                yield a
            else:
                return

    return ''.join(_iter())

bleu, count = 0, 0
for line in data:
    each = line.strip().split('|')
    ref = each[1]#tokenizer.tokenize(each[1])
    cand = each[2]#tokenizer.tokenize(each[2])

    common = common_start(ref, cand)
     
    ref = tokenizer.tokenize(ref.replace(common, '').strip())
    cand = tokenizer.tokenize(cand.replace(common, '').strip())
    #print(ref)
    #print(cand)
    #exit()
    bleu += nltk.translate.bleu_score.sentence_bleu([ref], cand, weights=[1])
    print(count)
    count += 1


print(bleu)
print(bleu / len(data))    
