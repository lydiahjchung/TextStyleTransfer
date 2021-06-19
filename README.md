# Text Style Transfer
SWCON401-00 Capstone Design

Generating sarcastic utterances via text style transfer. The original SARC dataset with instructions can be found [here](https://github.com/NLPrinceton/SARC).

## Using pre-trained BART
Referenced *KoBART Chit-Chat model* to implemt. Referenced code can be found [here](https://github.com/haven-jeon/KoBART-chatbot).

**Training**
```python
mkdir logs
python bart/main.py --gradient_clip_val 1.0 --max_epochs 10 --default_root_dir logs --gpus 1
```
## Using Cross-aligned Auto Encoder
Model proposed in *[Style Transfer from Non-Parallel Text by Cross-Alignment](https://arxiv.org/pdf/1905.05621.pdf), NIPS, 2017*.

**Training**
```python
mkdir save
python cross-aligned/style_transfer.py --saves_path ./save --train ./sarc/sarc.train --max_epochs 120 --vocab ./tmp/sarc.vocab

```
## Using Style Transformer
Model proposed in *[Style Transformer: Unpaired Text Style Transfer without Disentangled Latent Representation](https://arxiv.org/pdf/1705.09655.pdf), ACL, 2019*.

**Training**
```python
mkdir save
python main.py
```
