import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import gym
import numpy as np
from torch import optim
from model import ActorCritic
from parser import parser
import os
from SharedAdam import SharedAdam
# --------------------------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 指定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processes = []
args = parser.parse_args()
args.N_steps = 5  # 设置 N_steps 的默认值为 5
hx_initial = torch.zeros(1, args.hidden_size).to(device)
cx_initial = torch.zeros(1, args.hidden_size).to(device)

def run_episode(worker_env, worker_model, N_steps=17):
    state_data = worker_env.reset()
    state = torch.from_numpy(state_data).float().unsqueeze(0).to(device)
    hx = hx_initial.clone().to(device)
    cx = cx_initial.clone().to(device)

    memory = []
    done = False
    epi_time = 0

    while not done:
        # Agent 玩游戏
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

        state = next_state  # 更新状态
        epi_time += 1

        # 每 N 步或 Episode 结束时，yield memory
        if len(memory) >= N_steps or done:
            yield memory, epi_time
            memory = []

    # 如果还有剩余的 memory，没有达到 N_steps，也要进行更新
    if memory:
        yield memory, epi_time

def update_params(memory, worker_model, global_model, worker_opt, clc=args.clc, gamma=args.gamma):
    # 提取数据
    states = [item['state'] for item in memory]
    actions = [item['action'] for item in memory]
    rewards = [item['reward'] for item in memory]
    hx_list = [item['hx'] for item in memory]
    cx_list = [item['cx'] for item in memory]

    # 计算 N 步回报
    returns = []
    with torch.no_grad():
        if memory[-1]['done']:
            R = torch.zeros(1).to(device)
        else:
            next_state = memory[-1]['next_state']
            hx = hx_list[-1]
            cx = cx_list[-1]
            R = worker_model.forward(next_state, hx, cx)[2]
            R = R.squeeze()
    for reward in reversed(rewards):
        R = reward + gamma * R
        R = R.squeeze()
        returns.insert(0, R)
    returns = torch.stack(returns).to(device)
    returns = returns.unsqueeze(1)  # 形状调整为 [N, 1]

    # 计算优势函数
    values = []
    new_log_probs = []
    entropies = []
    for state, action, hx, cx in zip(states, actions, hx_list, cx_list):
        mean, std, value, hx, cx = worker_model.forward(state, hx, cx)
        dist = torch.distributions.Normal(mean, std)
        action_clipped = action.clamp(-0.999, 0.999)
        raw_action = torch.atanh(action_clipped)
        log_prob = dist.log_prob(raw_action)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        value = value.squeeze()
        values.append(value)
        new_log_probs.append(log_prob)
        entropy = dist.entropy()
        entropies.append(entropy)

    values = torch.stack(values).to(device).unsqueeze(1)  # 确保形状为 [N, 1]
    new_log_probs = torch.cat(new_log_probs, dim=0)
    entropy = torch.cat(entropies).mean()

    # 计算优势
    advantages = returns - values

    # 计算损失
    policy_loss = (-new_log_probs * advantages.detach()).mean()
    value_loss = F.mse_loss(values, returns)
    loss = policy_loss - args.entropy_coef * entropy + clc * value_loss

    # 更新参数
    worker_opt.zero_grad()
    loss.backward()
    # 调整梯度裁剪阈值
    torch.nn.utils.clip_grad_norm_(worker_model.parameters(), max_norm=5)

    for local_param, global_param in zip(worker_model.parameters(), global_model.parameters()):
        global_param._grad = local_param.grad

    worker_opt.step()
    worker_model.load_state_dict(global_model.state_dict())

    # 返回一些指标用于监控
    outcome = {
        'loss': loss.item(),
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'returns': returns.mean().item()
    }

    return outcome ,loss

def worker(t, global_model, counter, params):
    worker_env = gym.make("BipedalWalkerHardcore-v3")
    worker_env.reset()
    # 创建局部模型
    worker_model = ActorCritic(args.input_size).to(device)
    worker_model.load_state_dict(global_model.state_dict())

    # 使用 SharedAdam 优化器，优化全局模型的参数
    worker_opt = SharedAdam(global_model.parameters(), lr=params.lr)
    worker_opt.zero_grad()

    for i in range(params.epochs):
        for memory, epi_time in run_episode(worker_env, worker_model, N_steps=args.N_steps):
            outcome ,loss = update_params(memory, worker_model, global_model, worker_opt)
            counter.value += 1
            if counter.value % 100 == 1:
                print("core:", t, "counter:", counter.value, "epi_time:", epi_time, "loss:", loss)
            if counter.value % 30000 == 1:
                save_index = (counter.value // 30000) + 1
                filename = f'A3C_LSTM{save_index}.model'
                torch.save(worker_model.state_dict(), filename)

if __name__ == '__main__':
    # 全局共享的模型
    global_model = ActorCritic(args.input_size).to(device)
    global_model.share_memory()

    model_path = 'A3C_LSTMbest.model'
    if os.path.exists(model_path):
        global_model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model parameters from {model_path}")

    # 计数器
    episode = mp.Value('i', 0)

    for i in range(args.num_processes):
        p = mp.Process(target=worker, args=(i, global_model, episode, args))  # 启用一个运行worker函数的新进程
        p.start()
        processes.append(p)
    for p in processes:  # 等待每个进程结束
        p.join()

    print("All processes have been joined.")
    for p in processes:  # 确保每个进程都已经终止
        p.terminate()

    torch.save(global_model.state_dict(), 'A3C_LSTM.model')

    # ---------------------测试模型-------------------------------------------
    done = False
    hx = hx_initial.clone().to(device)
    cx = cx_initial.clone().to(device)
    env = gym.make('BipedalWalkerHardcore-v3')
    state_data = env.reset()
    state = torch.from_numpy(state_data).float().unsqueeze(0).to(device)
    global_model.load_state_dict(torch.load('A3C_LSTM.model', map_location=device))
    global_model.eval()
    while not done:
        env.render()
        with torch.no_grad():
            action, log_prob, value, hx, cx = global_model.action(state, hx, cx)
        action_np = action.detach().cpu().numpy()[0]

        # 与环境交互，获取下一个状态、奖励和完成标志
        next_state_data, reward, done, info = env.step(action_np)
        next_state = torch.from_numpy(next_state_data).float().unsqueeze(0).to(device)
        state = next_state

    env.close()