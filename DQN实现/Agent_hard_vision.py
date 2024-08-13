import numpy as np
import torch
from Gridworld import Gridworld
import random
from collections import deque
from matplotlib import pylab as plt
from tqdm import tqdm
import copy

# --------------- reference --------------------
epochs = 12000
losses = []
mem_size = 1000 # 经验回放内存大小
batch_size = 200 # 经验回放小批量数量，内存满了的时候随机选取该数量的子集训练
replay = deque(maxlen = mem_size)

h = 0
epsilon= 1.0
alpha = 1.0 # 贝尔曼方程参数
gamma = 0.9 # 贝尔曼方程参数

input_size = 5*5*4
max_moves = 80
sync_freq = 500 # 更新频率
run_times = 0 # 当前Q网络已经被更新的次数

action_set = {
    0: 'u',
    1: 'd',
    2: 'l',
    3: 'r'
}

# --------------- model -------------------------
model = torch.nn.Sequential(
    torch.nn.Linear(input_size,128),
    torch.nn.ReLU(),
    torch.nn.Linear(128,64),
    torch.nn.ReLU(),
    torch.nn.Linear(64,32),
    torch.nn.ReLU(),
    torch.nn.Linear(32,4)
)

model2 = copy.deepcopy(model)
model2.load_state_dict(model.state_dict()) # 创建目标网络

loss_fn = torch.nn.MSELoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#---------------- train -------------------------
# epochs = 8000
for i in tqdm(range(epochs)):
    game = Gridworld(size=5, mode='random')

    # 获取游戏信息
    state_now_ran = game.board.render_np().reshape(1, input_size) + np.random.rand(1, input_size) / 100.0 # 增加噪音
    state_now = torch.from_numpy(state_now_ran).float()
    status = 1
    mov_times = 0

    # 每局游戏
    while (status == 1):
        run_times += 1
        mov_times += 1

        # Q-learning
        Q_val_tens = model(state_now)
        Q_val = Q_val_tens.data.numpy()

        if (random.random() < epsilon): # 部分概率随机取一个方向走，部分概率取预测的最大值走
            action_ = np.random.randint(0, 4)
        else:
            action_ = np.argmax(Q_val)

        action = action_set[action_]
        game.makeMove(action) # 下一步

        state_next_ran = game.board.render_np().reshape(1, input_size) + np.random.rand(1, input_size) / 100.0
        state_next = torch.from_numpy(state_next_ran).float()

        reward = game.reward()
        done = True if reward != -1 else False # 记录本步结束后是否游戏结束，如果本步结束后游戏结束，那么就不存在下一状态。(落入坑中或者到达终点)
        exp = (state_now, action_, reward, state_next, done) # 记录贝尔曼方程里的 Q(s,a) R 和 Q(s',a') 其中action_是a ,state_now是s
        replay.append(exp)
        state_now = state_next # 下一步变成本步了

        # 经验回放算法
        # 经验回放算法，选取batch_size数量的子集训练
        if len(replay) > batch_size:
            minibatch = random.sample(replay, batch_size) # 从内存条中取batch_size个数据
            # 分别提取五个记录
            state_now_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch])
            action_batch = torch.Tensor([a for (s1, a, r, s2, d) in minibatch])
            reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in minibatch])
            state_next_batch = torch.cat([s2 for (s1, a, r, s2, d) in minibatch])
            done_batch = torch.Tensor([d for (s1, a, r, s2, d) in minibatch])

            Q_now = model(state_now_batch) # 当前状态Q值
            with torch.no_grad(): # 不计算梯度开关，节约内存
                Q_next = model2(state_next_batch) # 下一状态Q值

            # Q-learning
            # 贝尔曼方程更新的新值,(1-done_batch)见注释
            # Y = Q_now + alpha * (reward_batch + gamma * torch.max(Q_next,dim=1)[0] * (1-done_batch) - Q_now)
            Y = reward_batch + gamma * ((1 - done_batch) * torch.max(Q_next, dim=1)[0])
            # 源Q值
            X = Q_now.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()

            loss = loss_fn(X, Y.detach())
            # print(i, loss.item())
            # clear_output(wait=True)
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()

            # 目标网络算法
            # 目标网络算法，每sync_freq步将网络更新一次，防止最新的更新对网络的影响
            if run_times % sync_freq == 0:  #
                model2.load_state_dict(model.state_dict())

        if reward != -1 or mov_times > max_moves: # 如果游戏结束，退出循环
            status = 0
            mov_times = 0

    if epsilon > 0.1:  # 调整每次随机选取的概率
        epsilon -= (1 / epochs)

losses = np.array(losses)

#---------------- ploting -------------------------
plt.figure(figsize=(10,7))
plt.plot(losses)
plt.xlabel("Epochs",fontsize=22)
plt.ylabel("Loss",fontsize=22)
plt.show()

#---------------- test -------------------------
def test_model(model,size = 5, mode='random', display=True):
    i = 0
    test_game = Gridworld(size=size ,mode=mode)
    state_ = test_game.board.render_np().reshape(1, input_size) + np.random.rand(1, input_size) / 10.0
    state = torch.from_numpy(state_).float()
    if display:
        print("Initial State:")
        print(test_game.display())
    status = 1
    score = 0
    while (status == 1):  # 每局游戏期间
        qval = model(state)
        qval_ = qval.data.numpy()
        action_ = np.argmax(qval_)  # 采取Q值最高的动作
        action = action_set[action_]
        if display:
            print('Move #: %s; Taking action: %s' % (i, action))
        test_game.makeMove(action)
        state_ = test_game.board.render_np().reshape(1, input_size) + np.random.rand(1, input_size) / 10.0
        state = torch.from_numpy(state_).float()
        if display:
            print(test_game.display())
        reward = test_game.reward()
        score += reward
        if reward != -1:
            if reward > 0:
                status = 2
                if display:
                    print("Game won! score: %s" % (score,))
            else:
                status = 0
                if display:
                    print("Game LOST. score: %s" % (score,))
        i += 1
        if (i > 15):
            if display:
                print("Game lost; too many moves.")
            break

    win = True if status == 2 else False
    return win

# 测试训练结果
max_games = 1000
wins = 0
for i in range(max_games):
    win = test_model(model, mode='random', display=False)
    if win:
        wins += 1

win_perc = float(wins) / float(max_games)
print("Games played: {0}, # of wins: {1}".format(max_games,wins))
print("Win percentage: {}%".format(100.0*win_perc))

for i in range(20):
    test_model(model, mode='random')