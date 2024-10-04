import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from parser import parser

# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)
args = parser.parse_args()

# 定义AC网络
class ActorCritic(nn.Module):
    def __init__(self, input_size=args.input_size, hidden_size=args.hidden_size, output_size=args.output_size):
        super(ActorCritic, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size*2) # 公共网络部分
        self.l2 = nn.Linear(hidden_size*2, hidden_size)
        self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        self.input_norm = nn.LayerNorm(input_size)

        # Actor 分支，输出均值和标准差的对数
        self.act_mean = nn.Linear(hidden_size, output_size)  # 动作均值输出层
        self.act_sigma = nn.Linear(hidden_size, output_size)  # 动作标准差的对数输出层

        # Critic 分支
        self.cri_l1 = nn.Linear(hidden_size, hidden_size)
        self.cri_output = nn.Linear(hidden_size, 1)

        nn.init.kaiming_uniform_(self.l1.weight, a=0.25, mode="fan_in", nonlinearity="leaky_relu") # 使用 Kaiming 均匀初始化
        nn.init.kaiming_uniform_(self.l2.weight, a=0.25, mode="fan_out", nonlinearity="leaky_relu")
        nn.init.kaiming_uniform_(self.cri_l1.weight,a=0.25,mode="fan_out",nonlinearity="leaky_relu")

        nn.init.xavier_uniform_(self.act_mean.weight, gain=1.0) # 使用 Xavier 均匀初始化
        nn.init.zeros_(self.act_mean.bias)

        nn.init.xavier_uniform_(self.act_sigma.weight, gain=1.0)
        nn.init.zeros_(self.act_sigma.bias)

        nn.init.xavier_uniform_(self.cri_output.weight)
        nn.init.zeros_(self.cri_output.bias)

        self.distribution = torch.distributions.Normal

    def forward(self, input, hx, cx):
        x = self.input_norm(input)
        # x = input
        x = F.leaky_relu(self.l1(x), 0.25)
        x = F.leaky_relu(self.l2(x), 0.25)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        mean = 2 * torch.tanh(self.act_mean(x)) # 输出动作的均值
        sigma = F.softplus(self.act_sigma(x))+0.001 # 输出动作的标准差

        c = F.leaky_relu(self.cri_l1(x) ,0.25)
        critic = self.cri_output(c)

        return mean, sigma, critic, hx, cx

    def action(self, input, hx, cx):
        mean, sigma, critic, hx, cx = self.forward(input, hx, cx)

        dist = torch.distributions.Normal(mean, sigma) # 创建一个正态分布对象，均值和标准差是层的输出。
        raw_action = dist.sample() # 生成一个动作
        action = torch.tanh(raw_action) # 将动作映射到 -1~1 范围内

        return action, critic, hx, cx

    def loss_func(self, state, action, reward, hx, cx):
        self.train()
        mu, sigma, values, hx, cx= self.forward(state, hx, cx)
        td = reward - values
        c_loss = td.pow(2)

        m = self.distribution(mu, sigma)
        log_prob = m.log_prob(action)
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(m.scale)
        exp_v = log_prob * td.detach() + args.entropy_coef * entropy
        a_loss = -exp_v
        total_loss = (a_loss + args.clc * c_loss).mean()
        return total_loss
