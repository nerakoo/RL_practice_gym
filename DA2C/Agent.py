import multiprocessing as mp
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch import optim
import numpy as np
from torch.nn import functional as F
import gym
import torch.multiprocessing as mp
from tqdm import tqdm

# -------------------------ignore------------------------------------------
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# code algorithm:DA2C(Distributed Advantage Actor-Critic)
# -------------------------params-------------------------------------------
params = {
    'epochs':1200,
    'n_workers':7,
}
processes = []
train_keep_time = []
# -------------------------model-------------------------------------------
# 演员-评论家模型是一个“双头模型"，
#                                                   log_softmax层 -> R^2   actor网络
#   R^4 -> L1(RELU) -> R^25 -> L2(RELU) -> R^50 ->
#              (detach创建该张量的一个“离断”版本)        L3(RELU) -> R^25 -> tanh层 -> R^1   critic网络
# 事实上，actor和critic共用同一个头”l1,l2“，critic模型和actor模型共用头，但是critic只更新l3的参数，不会影响到l1和l2
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.l1 = nn.Linear(4,25)
        self.l2 = nn.Linear(25,50)
        self.actor_lin1 = nn.Linear(50,2)
        self.l3 = nn.Linear(50,25)
        self.critic_lin1 = nn.Linear(25,1)
    def forward(self,x):
        x = F.normalize(x,dim=0) # 将输入数据利用L-p范数归一化,L-p范数是向量里所有元素的模。例如(1,2,3)->(1/√14,2/√14,3/√14),L-p范数是√14
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))
        actor = F.log_softmax(self.actor_lin1(y),dim=0) # actor网络返回两个动作的对数概率
        c = F.relu(self.l3(y.detach())) # .detach()方法用于从计算图中分离一个张量，这意味着它创建了一个新的张量，与原始张量共享数据，但不再参与任何计算图。这意味着这个新的张量不依赖于过去的计算值。
        critic = torch.tanh(self.critic_lin1(c)) # critic网络返回一个tanh为界的数字-1~1.(因为CartPole的奖励范围是-1~1,很契合，所以选择tanh)
        return actor, critic

# -------------------------run_episode-------------------------------------------
# 运行轮次并收集数据
def run_episode(worker_env, worker_model):
    state = torch.from_numpy(worker_env.env.state).float() # 将环境状态从Numpy数组转换为Pytorch张量
    values, logprobs, rewards = [],[],[] # 分别存储计算的状态值（critic），对数概率（actor）和奖励
    done = False
    j = 0
    while (done == False):
        j += 1
        policy, value = worker_model(state) # policy输出两个行为的对数概率，例如（[-0.2542, -1.4941]）,value输出对当前状态得分的估计。
        values.append(value)

        logits = policy.view(-1) # 将原张量转变成一维的结构。
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample() # Categorical函数:按照传入的probs中给定的概率，在相应的位置处进行取样，取样返回的是该位置的整数索引。
        logprob_ = policy.view(-1)[action]
        logprobs.append(logprob_) # 存储随机抽取的动作的概率

        state_, _, done, info = worker_env.step(action.detach().numpy())
        state = torch.from_numpy(state_).float()
        if done:
            reward = -10
            worker_env.reset()
        else:
            reward = 1.0
        rewards.append(reward)
    return values, logprobs, rewards, j

# -------------------------update_params-------------------------------------------
# 更新参数
def update_params(worker_opt, values, logprobs, rewards, clc=0.1, gamma=0.55):
    # 处理数据，确定是一维的结构
    rewards = torch.Tensor(rewards).flip(dims=(0,)).view(-1)
    logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1)
    values = torch.stack(values).flip(dims=(0,)).view(-1)

    # 信用分配：计算贴现奖励
    Returns = []
    ret_ = torch.Tensor([0])
    for r in range(rewards.shape[0]):
        ret_ = rewards[r] + gamma * ret_
        Returns.append(ret_)
    Returns = torch.stack(Returns).view(-1)
    Returns = F.normalize(Returns, dim=0)

    actor_loss = -1 * logprobs * (Returns - values.detach())  # actor的更新公式是: -log(π(a|S)) * (R-V_π(S)) ,V_π(S))是critic对于当前状态的得分估计
    critic_loss = torch.pow(values - Returns, 2)  # 利用最终得分直接更新critic
    loss = actor_loss.sum() + clc * critic_loss.sum()
    loss.backward()
    worker_opt.step()
    return actor_loss, critic_loss, len(rewards)

# -------------------------worker-------------------------------------------
def worker(t, worker_model, counter, params):
    worker_env = gym.make("CartPole-v1")
    worker_env.reset()
    worker_opt = optim.Adam(lr=1e-4,params=worker_model.parameters()) # 每个模型使用独立的优化器，但是共享模型
    worker_opt.zero_grad() # 将模型的参数梯度初始化为0

    for i in range(params['epochs']):
        worker_opt.zero_grad()

        # A2C算法，运行一个轮次并存储，然后根据公式更新参数
        values, logprobs, rewards, keep_time = run_episode(worker_env, worker_model) # 运行轮次并收集数据
        actor_loss,critic_loss,eplen = update_params(worker_opt, values, logprobs, rewards) #C

        counter.value = counter.value + 1 # 全局变量加一，表示又训练了一个轮次
        train_keep_time.append(keep_time)

if __name__ == "__main__":
    # -------------------------多线程-------------------------------------------
    # 环境准备与多线程
    # 多线程的优势：打破经验之间的耦合，起到类似于经验回放的作用，但又避免了 Repaly Buffer 内存占用过大的问题
    env = gym.make('CartPole-v1') # 创建CartPole-v1环境
    MasterNode = ActorCritic() # 创建一个全局共享的模型
    MasterNode.share_memory()  # 将tensor从底层内存转移到共享内存，可以在进程之间共享，但是大小不能改变

    counter = mp.Value('i',0) # 创建一个全局共享的计数器，类型为i(整数)

    for i in range(params['n_workers']):
        p = mp.Process(target=worker, args=(i, MasterNode, counter, params)) # 启用一个运行worker函数的新进程
        p.start()
        processes.append(p)
    for p in processes:  # "连接"每个进程
        p.join()
    for p in processes:  # 确保每个进程都已经终止
        p.terminate()

    # # ---------------------cheaking-------------------------------------------
    # env = gym.make("CartPole-v1")
    # env.reset()
    # done = False
    # state_ = np.array(env.env.state)
    #
    # while not (done):
    #     state = torch.from_numpy(state_).float()
    #     env.render()  # 可视化环境
    #     logits, value = MasterNode(state)
    #     action_dist = torch.distributions.Categorical(logits=logits)
    #     action = action_dist.sample()
    #     # action = np.random.choice(np.array([0, 1]), p=logits.data.numpy().squeeze())
    #     state_, reward, done, info = env.step(action.detach().numpy())

    env.close()
    torch.save(MasterNode, 'DA2C.model')

    # ---------------------ploting-------------------------------------------
    # plt.figure(figsize=(10, 7))
    # plt.ylabel("Episode Duration", fontsize=22)
    # plt.xlabel("Training Epochs", fontsize=22)
    # plt.plot(train_keep_time, color='green')
    # plt.show()