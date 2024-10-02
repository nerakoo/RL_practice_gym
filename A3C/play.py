import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
from model import ActorCritic  # 导入模型定义

# --------------------------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    # 创建环境并获取输入维度
    env = gym.make('BipedalWalkerHardcore-v3')
    input_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    hidden_size = 128

    hx_initial = torch.zeros(1, hidden_size).to(device)
    cx_initial = torch.zeros(1, hidden_size).to(device)

    model = ActorCritic(input_size=input_size, hidden_size=hidden_size).to(device)
    model.load_state_dict(torch.load('A3C_LSTM4.model', map_location=device))
    model.eval()

    # 初始化环境
    done = False
    hx = hx_initial.clone().to(device)
    cx = cx_initial.clone().to(device)
    state_data = env.reset()
    state = torch.from_numpy(state_data).float().unsqueeze(0).to(device)

    total_reward = 0  # 记录总奖励

    while not done:
        env.render()
        with torch.no_grad():
            action, log_prob, value, hx, cx = model.action(state, hx, cx)
        action_np = action.detach().cpu().numpy()[0]

        # 与环境交互，获取下一个状态、奖励和完成标志
        next_state_data, reward, done, info = env.step(action_np)
        next_state = torch.from_numpy(next_state_data).float().unsqueeze(0).to(device)
        state = next_state
        print(reward)

        total_reward += reward  # 累加奖励

    # 打印结果
    if total_reward < 0:
        print("lost", total_reward)
    else:
        print("win", total_reward)
    env.close()
