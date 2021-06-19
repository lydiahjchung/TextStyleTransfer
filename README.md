# Text Style Transfer
SWCON401-00 Capstone Design

Generating sarcastic utterances via text style transfer.

## Using pre-trained BART
Training
```python
mkdir logs
python bart/main.py --gradient_clip_val 1.0 --max_epochs 10 --default_root_dir logs --gpus 1
```
## Using Cross-aligned Auto Encoder
Training
```python
mkdir save
python cross-aligned/style_transfer.py --saves_path ./save --train ./sarc/sarc.train --max_epochs 120 --vocab ./tmp/sarc.vocab

```
## Using Style Transformer
Training
```python
mkdir save
python main.py
```
