import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import gym
import numpy as np
from torch import optim
import matplotlib.pyplot as plt
from model import ActorCritic
from parser import parser

# --------------------------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 指定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BipedalWalkerHardcore-v3 环境结构如下：
# 环境信息为24维度，分别为：
# 机器人的身体状态：横向和纵向速度, 身体角度和角速度
# 机器人的腿部状态：两条腿的关节角度, 两条腿的关节速度, 每条腿的接地传感器（即是否接触到地面）
# 激光传感器（LIDAR）数据：用于感知机器人前方的地形，帮助它了解前方的地面形状

# 输入是四个信号：
# 第一，二，三，四条腿的髋关节扭矩，通常在 [-1, 1] 的范围内。

processes = []
args = parser.parse_args()
hx_initial = torch.zeros(1, args.hidden_size).to(device)
cx_initial = torch.zeros(1, args.hidden_size).to(device)
epi_times = mp.Queue()

def run_episode(worker_env, worker_model):
    state_data = worker_env.reset()
    state = torch.from_numpy(state_data).float().unsqueeze(0).to(device)
    hx = hx_initial.clone().to(device)
    cx = cx_initial.clone().to(device)

    memory = []
    done = False
    epi_time = 0

    while (done == False and epi_time <= 500):
        # Agent玩游戏
        action, log_prob, value, hx, cx = worker_model.action(state, hx, cx)
        action_np = action.detach().cpu().numpy()[0]

        # 与环境交互，获取下一个状态、奖励和完成标志
        next_state_data, reward, done, info = worker_env.step(action_np)
        next_state = torch.from_numpy(next_state_data).float().unsqueeze(0).to(device)

        memory.append({
            'state': state,
            'action': action,
            'log_prob': log_prob,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'hx': hx.detach(),
            'cx': cx.detach()
        })

        state = next_state # 更新状态
        epi_time = epi_time + 1

    return memory, epi_time

def update_params(memory, worker_model, worker_opt, clc=args.clc, gamma=args.gamma):
    outcome = []

    # 处理数据，确定是一维的结构
    states = torch.cat([item['state'] for item in memory], dim=0).to(device)
    actions = torch.cat([item['action'] for item in memory], dim=0).to(device)
    rewards = [item['reward'] for item in memory]
    hx_list = [item['hx'] for item in memory]
    cx_list = [item['cx'] for item in memory]

    # 信用分配：计算贴现奖励
    returns = []
    R = worker_model.forward(memory[-1]['next_state'], hx_list[-1], cx_list[-1])[2].detach()
    R = R.squeeze()  # 确保 R 是标量
    for reward in reversed(rewards):
        R = reward + gamma * R
        returns.insert(0, R)
    returns = torch.stack(returns).to(device)
    returns = returns.squeeze()  # 去除多余的维度
    returns = returns.unsqueeze(1)  # 形状调整为 [batch_size, 1]

    # 标准化回报（可选）
    returns = (returns - returns.mean()) / (returns.std() + 1e-6)

    # 重新计算价值估计和对数概率
    values = []
    new_log_probs = []

    # 不需要重新初始化 hx 和 cx，因为我们从 hx_list 和 cx_list 中获取
    for state, action, hx, cx in zip(states, actions, hx_list, cx_list):
        state = state.unsqueeze(0)
        action = action.unsqueeze(0)

        mean, std, value, hx, cx = worker_model.forward(state, hx, cx)
        dist = torch.distributions.Normal(mean, std)

        # 计算对应的 raw_action
        # 计算对应的 raw_action
        # 事实上，action是一个-1~1的值，所以使用log1p来保证输入为正
        action_clipped = action.clamp(-0.999, 0.999)
        raw_action = torch.atanh(action_clipped)

        # 计算新的对数概率,同action
        log_prob = dist.log_prob(raw_action)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        values.append(value)
        new_log_probs.append(log_prob)

    values = torch.cat(values, dim=0)
    new_log_probs = torch.cat(new_log_probs, dim=0)

    # 确保 values 和 returns 的形状一致
    values = values.squeeze()
    values = values.unsqueeze(1)
    returns = returns.squeeze()
    returns = returns.unsqueeze(1)

    # 计算优势函数
    advantages = returns - values.detach()

    # 计算策略熵
    entropy = dist.entropy().mean()

    # 计算损失
    policy_loss = (-new_log_probs * advantages).mean()
    value_loss = F.mse_loss(values, returns)
    loss = policy_loss - args.entropy_coef * entropy + clc * value_loss  # args.clc 是 Critic Loss Coefficient

    # 反向传播和更新参数
    worker_opt.zero_grad()
    loss.backward()
    worker_opt.step()

    # 返回一些指标用于监控
    outcome = {
        'loss': loss.item(),
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'returns': returns.mean().item()
    }

    return outcome

def worker(t, worker_model, counter, params):
    worker_env = gym.make("BipedalWalkerHardcore-v3")
    worker_env.reset()
    worker_model = worker_model.to(device)
    worker_opt = optim.Adam(lr=params.lr, params=worker_model.parameters())
    worker_opt.zero_grad()

    for i in range(params.epochs):
        worker_opt.zero_grad()
        memory, epi_time = run_episode(worker_env, worker_model)
        outcome = update_params(memory, worker_model, worker_opt)
        counter.value = counter.value + 1
        if counter.value % 100 == 1:
            print("core:",t,counter.value, epi_time)
        if counter.value % 3000 == 1:
            save_index = (counter.value // 3000) + 1
            filename = f'A3C_LSTM{save_index}.model'
            torch.save(worker_model, filename)
       # epi_times.put(epi_time)

if __name__ == '__main__':
    # 全局共享的模型
    global_model = ActorCritic(args.input_size).to(device)
    global_model.share_memory()

    # 计数器
    episode = mp.Value('i', 0)

    for i in range(args.num_processes):
        p = mp.Process(target=worker, args=(i, global_model, episode, args)) # 启用一个运行worker函数的新进程
        p.start()
        processes.append(p)
    for p in processes:  # "连接"每个进程
        p.join()

    print("All processes have been joined.")
    for p in processes:  # 确保每个进程都已经终止
        p.terminate()

    torch.save(global_model, 'A3C_LSTM.model')

    # ---------------------cheaking-------------------------------------------
    done = False
    hx = hx_initial.clone().to(device)
    cx = cx_initial.clone().to(device)
    env = gym.make('BipedalWalkerHardcore-v3')
    state_data = env.reset()
    state = torch.from_numpy(state_data).float().unsqueeze(0).to(device)
    while not done:
        env.render()
        action, log_prob, value, hx, cx = global_model.action(state, hx, cx)
        action_np = action.detach().cpu().numpy()[0]

        # print(f"next_state: {next_state}")
        # print(f"Action: {action}")
        # print(f"Reward: {reward}")
        # print(f"Done: {done}")

        # 与环境交互，获取下一个状态、奖励和完成标志
        next_state_data, reward, done, info = env.step(action_np)
        next_state = torch.from_numpy(next_state_data).float().unsqueeze(0).to(device)
        state = next_state

    # 绘图
    plt.plot(epi_times)
    plt.xlabel('Episode')
    plt.ylabel('epoch times')
    plt.title('A3C Training on BipedalWalkerHardcore-v3')
    plt.show()

    env.close()
