import torch
from torch.optim import Optimizer
from collections import defaultdict
from math import sqrt
import time
import torch.nn.functional as F

class SharedAdam(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-4,
        betas=(0.9, 0.999), # 一阶矩和二阶矩估计
        eps=1e-3,
        weight_decay=0,
        amsgrad=True,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad
        )
        super(SharedAdam, self).__init__(params, defaults)
        defaults["ONE"] = torch.ones(())
        defaults["TWO"] = torch.ones(()) * 2.0
        defaults["torch_eps"] = torch.tensor(eps).float()
        defaults["beta1"], defaults["beta2"] = betas
        defaults["beta2T"] = torch.tensor(defaults["beta2"]).float()
        defaults["stepNum"] = torch.zeros(())
        defaults["OneMinusBeta1"] = defaults["ONE"].sub(defaults["beta1"]).float()
        defaults["OneMinusBeta2"] = defaults["ONE"].sub(defaults["beta2T"])
        defaults["NegLR"] = -lr
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
                state["max_exp_avg_sq"] = torch.zeros_like(p) + defaults["torch_eps"]

    def share_memory(self):
        self.defaults["stepNum"].share_memory_()
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["exp_avg"].share_memory_()
                state["exp_avg_sq"].share_memory_()
                state["max_exp_avg_sq"].share_memory_()

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        stepFlag = 1
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                if stepFlag:
                    defaults = self.defaults
                    amsgrad = defaults["amsgrad"]
                    OneMinusBeta1 = defaults["OneMinusBeta1"]
                    OneMinusBeta2 = defaults["OneMinusBeta2"]
                    beta2T = defaults["beta2T"]
                    TWO = defaults["TWO"]
                    defaults["stepNum"].add_(defaults["ONE"])
                    step_t = defaults["stepNum"].item()
                    bias_correction1 = 1 - defaults["beta1"] ** step_t
                    bias_correction2 = 1 - defaults["beta2"] ** step_t
                    bias_correction2_sqrt = sqrt(bias_correction2)
                    step_size_neg = (
                        defaults["NegLR"] * bias_correction2_sqrt / bias_correction1
                    )
                    stepFlag = 0

                grad = p.grad
                state = self.state[p]

                exp_avg = state["exp_avg"]
                exp_avg.lerp_(grad, OneMinusBeta1)
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg_sq.mul_(beta2T).addcmul_(grad, grad, value=OneMinusBeta2)

                exp_avg_sq.add_(defaults["torch_eps"])
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the mean of max. and 2nd moment running avg for normalizing running avg. of gradient
                    denom = exp_avg_sq.add(max_exp_avg_sq).div(TWO).sqrt()
                else:
                    denom = exp_avg_sq.add(defaults["torch_eps"]).sqrt()

                p.data.addcdiv_(exp_avg, denom, value=step_size_neg)
        return loss