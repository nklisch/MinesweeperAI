import math

import MinesweeperEnv.Minesweeper as ms
import random
from itertools import count
from collections import namedtuple
import torch as torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import matplotlib

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, height, width, finalOutputs, kernel_size, stride, padding, convLayerChannels,
                 fullyConnectedLayers, device,
                 parameters=None):
        super(DQN, self).__init__()
        self.convLayers = nn.ModuleList()
        self.batchNormal = nn.ModuleList()
        self.fullyConnectedLayers = nn.ModuleList()
        self.device = device

        def conv2d_size_out(size, kernel_size=kernel_size, stride=stride, padding=padding):
            return (((size + 2 * padding - kernel_size)) // stride) + 1

        convw = width
        convh = height
        for index, channels in enumerate(convLayerChannels[:-1]):
            self.convLayers.append(
                nn.Conv2d(channels, convLayerChannels[index + 1], kernel_size=kernel_size, stride=stride,
                          padding=padding))
            self.batchNormal.append(nn.BatchNorm2d(convLayerChannels[index + 1]))
            convw = conv2d_size_out(convw)
            convh = conv2d_size_out(convh)

        self.convLayers.append(
            nn.Conv2d(convLayerChannels[-1], convLayerChannels[-1], kernel_size=kernel_size, stride=stride,
                      padding=padding))
        self.batchNormal.append(nn.BatchNorm2d(convLayerChannels[-1]))
        convw = conv2d_size_out(convw)
        convh = conv2d_size_out(convh)
        linearInputSize = convw * convh * convLayerChannels[-1]

        for output in fullyConnectedLayers:
            self.fullyConnectedLayers.append(nn.Linear(linearInputSize, output))
            linearInputSize = output

        self.outputLayer = nn.Linear(linearInputSize, finalOutputs)
        if parameters:
            self.load_state_dict(parameters)
        self.to(device)

    def forward(self, x):

        for convLayer, batchNorm in zip(self.convLayers, self.batchNormal):
            x = F.relu(batchNorm(convLayer(x)))

        for layer in self.fullyConnectedLayers:
            x = F.relu(layer(x.view(x.shape[0], -1)))

        return self.outputLayer(x.view(x.shape[0], -1))


class TrainingModel:

    def __init__(self, batchSize=200, gamma=0.99, epsilonStart=0.9, epsilonEnd=0.05, epsilonDecay=200, device='cpu'):
        self.batch_size, self.gamma, self.epsilonStart = batchSize, gamma, epsilonStart
        self.epsilonEnd, self.epsilonDecay = epsilonEnd, epsilonDecay
        self.device = device
        self.policyNet = self.targetNet = None
        self.stepsDone = 0

    def createDQN(self, height, width, numberOfActions, kernel_size=3, stride=1, padding=1,
                  convLayerChannels=[9, 18, 36],
                  fullyConnectedLayers=[]):
        self.policyNet = DQN(height, width, numberOfActions, kernel_size, stride, padding, convLayerChannels,
                             fullyConnectedLayers, self.device)
        self.targetNet = DQN(height, width, numberOfActions, kernel_size, stride, padding, convLayerChannels,
                             fullyConnectedLayers, self.device,
                             self.policyNet.state_dict())

        self.numberOfActions = numberOfActions
        self.targetNet.eval()
        self.optimizer = optim.RMSprop(self.policyNet.parameters())
        return self

    def epsThreshold(self):
        return self.epsilonEnd + (self.epsilonStart - self.epsilonEnd) * \
               math.exp(-1. * self.stepsDone / self.epsilonDecay)

    def selectAction(self, state):
        # Maybe return actions as tensors on the GPU if it is actually faster
        sample = random.random()
        epsThreshold = self.epsThreshold()
        self.stepsDone += 1
        if sample > epsThreshold:
            with torch.no_grad():
                return self.policyNet(state).max(1)[1].view(1, 1)[0][0].item()
        else:
            return random.randrange(self.numberOfActions)

    def updateTargetNet(self, ):
        self.targetNet.load_state_dict(self.policyNet.state_dict())

    def optimize(self, memory):
        if len(memory) < self.batch_size:
            return
        transitions = memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        nonFinalMask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=self.device, dtype=torch.uint8)
        nonFinalNextStates = torch.cat([s for s in batch.next_state
                                        if s is not None])

        stateBatch = torch.cat(batch.state, 0)
        actionBatch = torch.cat(batch.action)
        rewardBatch = torch.cat(batch.reward)
        temp = self.policyNet(stateBatch)
        stateActionValues = temp.gather(1, actionBatch.view((-1, 1)))

        nextStateValues = torch.zeros(self.batch_size, device=self.device)
        nextStateValues[nonFinalMask] = self.targetNet(nonFinalNextStates).max(1)[0].detach()

        expectedStateActionValues = (nextStateValues * self.gamma) + rewardBatch

        loss = F.smooth_l1_loss(stateActionValues, expectedStateActionValues.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policyNet.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


def trainAModel(boardShape, percentOfBombs, numberOfEpisodes, batchSize, gamma, epsilonStart, epsilonEnd, epsilonDecay,
                kernelSize, stride, padding, targetUpdate, convLayerChannels,
                fullyConnectedLayers, device, fileOutputPath):
    memory = ReplayMemory(50000)
    env = ms.MinesweeperEnv(boardShape, percentOfBombs, device)
    wins = 0
    steps = 0
    totalReward = 0
    model = None
    dfRewards = pd.DataFrame(columns=['Episode', 'Average Total Reward', 'Average Win', 'Average Steps'])
    print('Episode: ', end='')
    for i_episode in range(numberOfEpisodes):
        env.reset()
        state = env.startingState().view((1, 9, boardShape[0], boardShape[1]))
        numberOfActions = env.action_space.n
        model = TrainingModel(batchSize=batchSize, gamma=gamma, epsilonStart=epsilonStart, epsilonEnd=epsilonEnd,
                              epsilonDecay=epsilonDecay, device=device)
        model.createDQN(boardShape[0], boardShape[1], numberOfActions, kernelSize, stride, padding, convLayerChannels,
                        fullyConnectedLayers)

        for t in count():
            action = model.selectAction(state)
            next_state, reward, done, _ = env.step(action)
            totalReward += reward
            steps += 1
            next_state = next_state.view((1, next_state.shape[0], next_state.shape[1], next_state.shape[2]))
            if reward == env.winReward():
                env.render()
                wins += 1

            reward = torch.tensor([reward], device=device, dtype=torch.float32)
            action = torch.tensor([action], device=device)
            if done:
                next_state = None

            memory.push(state, action, next_state, reward)
            state = next_state

            model.optimize(memory)
            if done:
                break

        if i_episode % targetUpdate == 0 and i_episode != 0:
            dfRewards = dfRewards.append(
                {'Episode': i_episode, 'Average Total Reward': totalReward/targetUpdate, 'Average Win': wins/targetUpdate, 'Average Steps': steps/targetUpdate},
                ignore_index=True)
            model.updateTargetNet()
            print(i_episode, end=' ')
            wins = 0
            steps = 0
            totalReward = 0
        if i_episode % (targetUpdate * 10) == 0:
            f = open(fileOutputPath, 'a')
            dfRewards.to_csv(fileOutputPath)
            f.close()

    return model, dfRewards