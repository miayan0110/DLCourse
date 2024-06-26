'''DLP DQN Lab'''
__author__ = 'chengscott'
__copyright__ = 'Copyright 2020, NCTU CGI Lab'
import argparse
from collections import deque
import itertools
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


class ReplayMemory:
    __slots__ = ['buffer']

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, *transition):
        # (state, action, reward, next_state, done)
        self.buffer.append(tuple(map(tuple, transition)))

    def sample(self, batch_size, device):
        '''sample a batch of transition tensors'''
        transitions = random.sample(self.buffer, batch_size)
        return (torch.tensor(x, dtype=torch.float, device=device)
                for x in zip(*transitions))


class Net(nn.Module):
    def __init__(self, state_dim=8, action_dim=4, hidden_dim=32):
        super().__init__()
        ## TODO ##
        self.layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        # raise NotImplementedError

    def forward(self, x):
        ## TODO ##
        return self.layer(x)
        # raise NotImplementedError


class DQN:
    def __init__(self, args):
        self._behavior_net = Net().to(args.device)
        self._target_net = Net().to(args.device)
        # initialize target network
        self._target_net.load_state_dict(self._behavior_net.state_dict())
        ## TODO ##
        self._optimizer = optim.Adam(self._behavior_net.parameters(),lr=args.lr)
        # raise NotImplementedError
        # memory
        self._memory = ReplayMemory(capacity=args.capacity)

        ## config ##
        self.device = args.device
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.freq = args.freq
        self.target_freq = args.target_freq

    def select_action(self, state, epsilon, action_space):
        '''epsilon-greedy based on behavior network'''
        ## TODO ##
        if random.uniform(0, 1) < epsilon:
            return action_space.sample()
        else:
            with torch.no_grad():
                action = self._behavior_net(torch.from_numpy(state).unsqueeze(0).to(self.device))
                return action.cpu().detach().numpy().argmax()  # 轉換成numpy array
        # raise NotImplementedError

    def append(self, state, action, reward, next_state, done):
        self._memory.append(state, [action], [reward / 10], next_state,
                            [int(done)])

    def update(self, total_steps):
        if total_steps % self.freq == 0:
            self._update_behavior_network(self.gamma)
        if total_steps % self.target_freq == 0:
            self._update_target_network()

    def _update_behavior_network(self, gamma):
        # sample a minibatch of transitions
        state, action, reward, next_state, done = self._memory.sample(
            self.batch_size, self.device)                       # state、action、reward等，每個都是一個batch的資料 (tensor)

        ## TODO ##
        q_value = self._behavior_net(state).gather(1, action.long())   # 根據選擇的action取得state對應的q_vlaue
        with torch.no_grad():
           q_next = self._target_net(next_state).max(1).values  # 取得各個state可獲得的最大q_value作為q_next
           q_target = gamma*q_next*(1-done) + reward
        criterion = nn.MSELoss()
        loss = criterion(q_value, q_target)
        # raise NotImplementedError
        # optimize
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._behavior_net.parameters(), 5)
        self._optimizer.step()

    def _update_target_network(self):
        '''update target network by copying from behavior network'''
        ## TODO ##
        self._target_net.load_state_dict(self._behavior_net.state_dict())
        # raise NotImplementedError

    def save(self, model_path, checkpoint=False):
        if checkpoint:
            torch.save(
                {
                    'behavior_net': self._behavior_net.state_dict(),
                    'target_net': self._target_net.state_dict(),
                    'optimizer': self._optimizer.state_dict(),
                }, model_path)
        else:
            torch.save({
                'behavior_net': self._behavior_net.state_dict(),
            }, model_path)

    def load(self, model_path, checkpoint=False):
        model = torch.load(model_path)
        self._behavior_net.load_state_dict(model['behavior_net'])
        if checkpoint:
            self._target_net.load_state_dict(model['target_net'])
            self._optimizer.load_state_dict(model['optimizer'])


def train(args, env, agent, writer):
    print('Start Training')
    action_space = env.action_space
    total_steps, epsilon = 0, 1.
    ewma_reward = 0
    max_reward = 0

    for episode in range(args.episode):
        total_reward = 0
        state = env.reset()[0]
        for t in itertools.count(start=1):
            # select action
            if total_steps < args.warmup:
                action = action_space.sample()
            else:
                action = agent.select_action(state, epsilon, action_space)
                epsilon = max(epsilon * args.eps_decay, args.eps_min)
            # execute action
            next_state, reward, done, _, _ = env.step(action)  # type(action)必須為numpy
            # store transition
            agent.append(state, action, reward, next_state, done)
            if total_steps >= args.warmup:
                agent.update(total_steps)

            state = next_state
            total_reward += reward
            total_steps += 1
            if done or t > 1000:
                ewma_reward = 0.05 * total_reward + (1 - 0.05) * ewma_reward
                writer.add_scalar('Train/Episode Reward', total_reward,
                                  total_steps)
                writer.add_scalar('Train/Ewma Reward', ewma_reward,
                                  total_steps)
                print(
                    'Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEwma reward: {:.2f}\tEpsilon: {:.3f}'
                    .format(total_steps, episode, t, total_reward, ewma_reward,
                            epsilon))
                
                if total_reward > max_reward:
                    max_reward = total_reward
                    print('Saving temp model...')
                    agent.save(args.tempModel)
                break
    env.close()


def test(args, env, agent, writer):
    print('Start Testing')
    action_space = env.action_space
    epsilon = args.test_epsilon
    seeds = (args.seed + i for i in range(10))
    rewards = []
    for n_episode, seed in enumerate(seeds):
        total_reward = 0
        state = env.reset(seed=seed)[0] # 初始化state
        
        ## TODO ##
        # 有了state之後，開始action
        for t in itertools.count(start=1):  # while not done:
            env.render()

            # 選擇action
            action = agent.select_action(state, epsilon, action_space)
            # 執行action，得到下一個state和reward => 儲存reward
            state, reward, done, _, _ = env.step(action)
            total_reward += reward

            if done:
                rewards.append(total_reward)
                writer.add_scalar('Test/Episode Reward', total_reward, n_episode)
                print(f'Episode: {n_episode}\tTotal reward: {total_reward}')
                break
        # raise NotImplementedError
    env.close()
    avg_reward = np.mean(rewards)
    print('Average Reward', avg_reward)
    return avg_reward

def is_save_model(args, avg_reward):
    with open(args.max_reward, 'r+') as f:
            max_reward = float(f.read())
            f.seek(0)                       # 將讀寫頭移動到檔案最初的位置
            if avg_reward > max_reward:
                f.truncate(0)               # 清除檔案內資料
                f.write(str(avg_reward))
                return True
            return False

def main():
    ## arguments ##
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-m', '--model', default='train/dqn/dqn.pth')
    parser.add_argument('--logdir', default='log/dqn')
    # train
    parser.add_argument('--warmup', default=10000, type=int)
    parser.add_argument('--episode', default=1200, type=int)
    parser.add_argument('--capacity', default=10000, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=.0005, type=float)
    parser.add_argument('--eps_decay', default=.995, type=float)
    parser.add_argument('--eps_min', default=.01, type=float)
    parser.add_argument('--gamma', default=.99, type=float)
    parser.add_argument('--freq', default=4, type=int)
    parser.add_argument('--target_freq', default=100, type=int)
    # test
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--test_temp_only', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--seed', default=20200519, type=int)
    parser.add_argument('--test_epsilon', default=.001, type=float)
    # save model
    parser.add_argument('--tempModel', default='temp/dqn.pth')
    parser.add_argument('--max_reward', default='train/dqn/max_reward.txt')
    args = parser.parse_args()

    ## main ##
    env = gym.make('LunarLander-v2', render_mode='human')
    # env = gym.make('LunarLander-v2')
    agent = DQN(args)
    writer = SummaryWriter(args.logdir)
    if args.test_only:
        agent.load(args.model)
        test(args, env, agent, writer)
    elif args.test_temp_only:
        agent.load(args.tempModel)
        test(args, env, agent, writer)
    else:
        agent.load(args.model)
        train(args, env, agent, writer)
        agent.load(args.tempModel)
        avg_reward = test(args, env, agent, writer)

        if is_save_model(args, avg_reward):
            print('Saving model...')
            agent.save(args.model)


if __name__ == '__main__':
    main()
