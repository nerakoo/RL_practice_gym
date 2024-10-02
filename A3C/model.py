import torch
import torch.nn as nn
import torch.nn.functional as F

# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)

# 定义AC网络
class ActorCritic(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(ActorCritic, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size*2) # 公共网络部分
        self.l2 = nn.Linear(hidden_size*2, hidden_size)
        self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        self.input_norm = nn.LayerNorm(input_size)

        # Actor 分支，输出均值和标准差的对数
        self.act_mean = nn.Linear(hidden_size, 4)  # 动作均值输出层
        self.act_log_std = nn.Linear(hidden_size, 4)  # 动作标准差的对数输出层

        # Critic 分支
        # self.cri_l1 = nn.Linear(hidden_size, hidden_size)
        self.cri_output = nn.Linear(hidden_size, 1)

        nn.init.kaiming_uniform_(self.l1.weight, a=0.25, mode="fan_in", nonlinearity="leaky_relu") # 使用 Kaiming 均匀初始化
        nn.init.kaiming_uniform_(self.l2.weight, a=0.25, mode="fan_out", nonlinearity="leaky_relu")
        # nn.init.kaiming_uniform_(self.cri_l1.weight,a=0.25,mode="fan_out",nonlinearity="leaky_relu",)

        nn.init.xavier_uniform_(self.act_mean.weight, gain=1.0) # 使用 Xavier 均匀初始化
        nn.init.zeros_(self.act_mean.bias)

        nn.init.xavier_uniform_(self.act_log_std.weight, gain=1.0)
        nn.init.zeros_(self.act_log_std.bias)

        nn.init.xavier_uniform_(self.cri_output.weight)
        nn.init.zeros_(self.cri_output.bias)

        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, input, hx, cx):
        # x = F.normalize(input, dim=0)
        x = self.input_norm(input)
        x = F.leaky_relu(self.l1(x), 0.25)
        x = F.leaky_relu(self.l2(x), 0.25)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        # 事实上，会有这样的疑问：mean和log看起来是两个完全没有关联的层，那么如何保证这两个层表示的是同一个概率密度的均值和标准差呢？
        # 其实这是一种隐式关联，首先，mean和log_std共用了一个头，这样就可以保证他们的输出和概率密度相关，同时，在训练的时候，也是进行
        # 联合训练的，他们共享一个优化目标:dist = torch.distributions.Normal(mean, std)，这使得他们的输出会逐渐变成需要的数值。

        # 使用两个不同的线性层，可以让网络更灵活地调整均值和标准差的映射关系。mean 和 log_std 的参数是独立的，这意味着网络可以在需要的时候
        # 调整其中一个而不影响另一个，适应不同的状态和策略需求。在许多实际应用中，这种方法已经被证明是有效的，能够成功训练出高性能的智能体。
        # PS:说白了，这俩玩意本来关系就不大，是你拟合的概率密度函数两个部分罢了。
        mean = torch.tanh(self.act_mean(x)) # 输出动作的均值
        log_std = self.act_log_std(x) # 输出动作的标准差的对数
        log_std = torch.clamp(log_std, min=-20, max=2)  # 根据经验值设定范围
        std = torch.exp(log_std) # 确保输出为正

        # c = F.leaky_relu(self.cri_l1(x) ,0.25)
        # critic = self.cri_output(c)
        critic = self.cri_output(x)

        return mean, std, critic, hx, cx

    def action(self, input, hx, cx):
        mean, std, critic, hx, cx = self.forward(input, hx, cx)

        dist = torch.distributions.Normal(mean, std) # 创建一个正态分布对象，均值和标准差是层的输出。
        raw_action = dist.sample() # 生成一个动作
        action = torch.tanh(raw_action) # 将动作映射到 -1~1 范围内

        # 计算经过 tanh 变换后的对数概率密度
        # tanh 函数的导数不是恒定的，因此在策略优化中，我们需要考虑动作通过 tanh 变换后，概率密度函数的变化。这是通过 1 - action^2 的修正项实现的。
        log_prob = dist.log_prob(raw_action) # 计算生成动作的对数概率
        log_prob -= torch.log(1 - action.pow(2) + 1e-6) # 计算 tanh 变换的雅可比行列式的对数。将修正项从 log_prob 中减去，得到修正后的对数概率密度。
        log_prob = log_prob.sum(dim=1, keepdim=True) # 对所有动作维度进行求和

        return action, log_prob, critic, hx, cx