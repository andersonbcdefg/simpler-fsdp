import torch
import torch.nn.functional as F

@torch.compile
def compiled_cross_entropy(embs, classifier, targets):
    logits = F.linear(embs, classifier)
    return F.cross_entropy(logits, targets)
