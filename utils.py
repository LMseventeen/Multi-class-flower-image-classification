import torch

def accuracy(output, target, topk=(1,)):
    """计算top-k准确率"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1.0 / batch_size).item())
    return res

def get_param_groups(model, base_lr=1e-3, fc_lr=1e-2):
    """为不同部分设置不同学习率"""
    params = [
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('fc')], 'lr': base_lr},
        {'params': model.fc.parameters(), 'lr': fc_lr}
    ]
    return params 