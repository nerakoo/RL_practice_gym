import gym
import numpy as np
from gym import envs
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

# --------------------parameter----------------------------------------
learning_rate = 0.009
MAX_DUR = 500
MAX_EPISODES = 1000
gamma = 0.99 # 由于奖励是负的，所以是增强。
# gamma = 1.01
score = [] # 记录训练期间轮次长度
expectation = 0.0
# print(envs.registry.all())

# ---------------------model-------------------------------------------
class Agent(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Agent, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.leaky_relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

model = Agent(4, 64, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 对数概率
# 定义损失函数：对于计算机计算（只能精确到小数点后若干位），使用对数函数优化
# 由于概率的值范围为[0,1],对数函数可以将其放大到(-∞,0)
def loss_fn(preds, r):
    return -1 * torch.sum(r * torch.log(preds)) # 计算概率的对数，乘以贴现奖励

# 信用分配
# 计算贴现奖励：奖励之前的最后一个动作要比第一个动作获得更多信用（越往后对赢得奖励的贡献越大）。
# 分配方式：对于获得奖励前的n个时间步，依次分配指数衰减的贴现因子 [γ^n, ..., γ^3, γ^2, γ^1, 1]，再进行归一化（除以最大值将值限制在0~1之内）。
def discount_rewards(rewards, gamma=0.99):
    lenr = len(rewards)
    disc_return = torch.pow(gamma, torch.arange(lenr).float()) * rewards # 将获得的奖励依次乘以贴现因子的指数次幂。

    disc_return /= disc_return.max() # 归一化
    return disc_return

# 环境准备
# 创建CartPole-v1环境
env = gym.make('CartPole-v1')
# state_now = env.reset()
# print(env.step(1))
# action = env.action_space.sample()
# state, reward, done, info = env.step(action)
def running_mean(x, N=50):
    kernel = np.ones(N)
    conv_len = x.shape[0]-N
    y = np.zeros(conv_len)
    for i in range(conv_len):
        y[i] = kernel @ x[i:i+N]
        y[i] /= N
    return y

# ---------------------train-------------------------------------------
# REINFORCE 算法
for episode in range(MAX_EPISODES):
    state_now = env.reset()
    done = False
    transitions = []

    # 通过当前的网络完整进行一次游戏，并且记录游戏过程中的所有信息
    for i in range(MAX_DUR):
        act_prob = model( torch.from_numpy(state_now).float().unsqueeze(0) ) # 模型预估一个当前动作概率。
        action = np.random.choice(np.array([0,1]), p=act_prob.data.numpy().squeeze())  # 根据动作概率选择动作
        prev_state = state_now
        state_now, reward, done, info = env.step(action) # 执行动作后，获得下一状态，记录对应信息
        transitions.append((prev_state, action, i + 1)) # 存储轮次信息

        if done: # 游戏结束，退出循环
            break

    ep_len = len(transitions) # 记录轮次长度
    score.append(ep_len)
    reward_batch = torch.Tensor([r for (s, a, r) in transitions]).flip(dims=(0,)) # 将该轮次所有信息进行批量化
    state_batch = torch.Tensor([s for (s, a, r) in transitions])
    action_batch = torch.Tensor([a for (s, a, r) in transitions])

    disc_returns = discount_rewards(reward_batch) # 信用分配
    # print(disc_returns)
    pred_batch = model(state_batch) # 获得模型对所有状态的预测值
    prob_batch = pred_batch.gather(dim=1, index=action_batch.long().view(-1, 1)).squeeze()
    loss = loss_fn(prob_batch, disc_returns)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

score = np.array(score)
avg_score = running_mean(score, 50)

# ---------------------ploting-------------------------------------------
plt.figure(figsize=(10,7))
plt.ylabel("Episode Duration",fontsize=22)
plt.xlabel("Training Epochs",fontsize=22)
plt.plot(avg_score, color='green')
plt.show()

# ---------------------ploting-------------------------------------------
state_now = env.reset()
done = False
while not(done):
    env.render()  # 可视化环境
    act_prob = model(torch.from_numpy(state_now).float().unsqueeze(0))
    action = np.random.choice(np.array([0, 1]), p=act_prob.data.numpy().squeeze())
    prev_state = state_now
    state_now, reward, done, info = env.step(action)
env.close()

