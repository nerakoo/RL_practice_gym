import torch
from torch.optim import Adam

class SharedAdam(Adam):
    def __init__(self, params, lr=1e-4, betas=(0.92, 0.999), eps=1e-8, weight_decay=0):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    state['step'] = torch.zeros(1)
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['exp_avg'].share_memory_()
                    state['exp_avg_sq'].share_memory_()
                    state['step'].share_memory_()
