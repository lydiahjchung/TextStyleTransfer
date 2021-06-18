import torch

def accuracy(yhat, y, pad):
    with torch.no_grad():
        yhat = yhat.max(dim=-1)[1]
        acc = (yhat == y).float()[y != pad].mean()

    return acc


def epoch_time(start, end):
    total = end - start
    mins = int(total / 60)
    secs = int(total - mins * 60)

    return mins, secs
