from Agent import ActorCritic
import torch
import gym
import numpy as np

if __name__ == "__main__":
    # MasterNode = torch.load('DA2C.model')
    MasterNode = torch.load('DA2C_N_step.model')
    env = gym.make("CartPole-v1")
    env.reset()
    done = False
    state_ = np.array(env.env.state)

    while not (done):
        state = torch.from_numpy(state_).float()
        env.render()  # 可视化环境
        logits, value = MasterNode(state)
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()
        # action = np.random.choice(np.array([0, 1]), p=logits.data.numpy().squeeze())
        state_, reward, done, info = env.step(action.detach().numpy())

    env.close()
    torch.save(MasterNode, 'DA2C.model')