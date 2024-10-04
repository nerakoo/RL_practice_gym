import time

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
import utils

# --------------------------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 指定设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
processes = []
args = parser.parse_args()
hx_initial = torch.zeros(1, args.hidden_size).to(device)
cx_initial = torch.zeros(1, args.hidden_size).to(device)

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def update_params(memory, global_model, model, optimizer, next_state, next_s_r, done, gamma=args.gamma):
    states = [item['state'] for item in memory]
    states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in states])
    actions = [item['action'] for item in memory]
    actions = torch.stack([torch.tensor(s, dtype=torch.float32) for s in actions])
    rewards = [item['reward'] for item in memory]
    hx_list = [item['hx'] for item in memory]
    hx_list = torch.stack([torch.tensor(s, dtype=torch.float32) for s in hx_list])
    cx_list = [item['cx'] for item in memory]
    cx_list = torch.stack([torch.tensor(s, dtype=torch.float32) for s in cx_list])

    # 贴现价值
    returns = []
    for r in rewards[::-1]:
        next_s_r = r + gamma * next_s_r
        returns.append(next_s_r)
    returns.reverse()
    returns = torch.tensor(returns ,requires_grad=True)

    for i in range(len(returns)):
        # print(states.shape, actions.shape, returns.shape, hx_list.shape, cx_list.shape)
        loss = model.loss_func(states[i], actions[i], returns[i], hx_list[i], cx_list[i])
        optimizer.zero_grad()
        loss.backward()

        ensure_shared_grads(model, global_model)

        optimizer.step()

    return loss

def run_episode(env, global_model, model, optimizer, N_steps=35):
    state = env.reset()
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    hx = hx_initial.clone().to(device)
    cx = cx_initial.clone().to(device)
    model.load_state_dict(global_model.state_dict())

    memory = []

    done = False
    epi_time = 0
    epi_reward = 0

    while not done:
        epi_time = epi_time + 1
        action, critic, hx, cx = model.action(state, hx, cx)
        action = action.detach().cpu().numpy()[0]
        action = torch.tensor(action).to(device)
        next_state, reward, done, info = env.step(action)
        next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(device)

        epi_reward += reward

        if done:
            next_s_r = 0.  # terminal
        else:
            _, _, next_s_r, _, _ = model(next_state, hx, cx)

        memory.append({
            'action': action,
            'reward': (reward + 1.0)/1.0,
            'state': next_state,
            'hx': hx,
            'cx': cx
        })

        if epi_time % N_steps == 0 or done:
            loss = update_params(memory, global_model, model, optimizer, next_state, next_s_r, done)
            memory = []

            if done:
                break

    return epi_time, epi_reward, loss
# worker
def worker(t, global_model, counter, params, optimizer):
    env = gym.make("BipedalWalker-v3")
    local_model = ActorCritic(args.input_size, args.hidden_size, args.output_size)
    local_model.train()

    for i in range(params.epochs):
        epi_time, epi_reward, loss = run_episode(env, global_model, local_model, optimizer)
        counter.value += 1
        print("counter.value:",counter.value,"epi_time:", epi_time, "epi_reward_avg:", epi_reward / epi_time, "loss:", loss)
        if counter.value % 700 == 1:
            save_index = (counter.value // 700) + 1
            filename = f'A3C_LSTM{save_index}.model'
            torch.save(global_model.state_dict(), filename)

    # print("loss_func:",global_model.loss_func(state,action,reward,hx_initial,cx_initial))

if __name__ == '__main__':
    worker_env = gym.make("BipedalWalker-v3")
    worker_env.reset()

    shared_model = ActorCritic(args.input_size, args.hidden_size, args.output_size)

    # 读档
    model_path = 'A3C_LSTMbest.model'
    if os.path.exists(model_path):
        shared_model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model parameters from {model_path}")

    shared_model.share_memory()
    optimizer = SharedAdam(
        shared_model.parameters(),
        lr=args.lr,
        eps=args.opt_eps,
        weight_decay=0,
        amsgrad=args.amsgrad,
    )
    optimizer.share_memory()

    # 计数器
    episode = mp.Value('i', 0)

    for i in range(args.num_processes):
        p = mp.Process(target=worker, args=(i, shared_model, episode, args, optimizer))  # 启用一个运行worker函数的新进程
        p.start()
        time.sleep(0.01)
        processes.append(p)

    for p in processes:  # 等待每个进程结束
        p.join()

    print("All processes have been joined.")
    for p in processes:  # 确保每个进程都已经终止
        p.terminate()

    time.sleep(0.001)