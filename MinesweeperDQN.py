import math

import MinesweeperEnv.Minesweeper as ms
import random
import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

env = ms.MinesweeperEnv((8, 8), 0.2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def __init__(self, height, width, finalOutputs, kernel_size, stride, convLayerChannels, fullyConnectedLayers,
                 parameters=None):
        super(DQN, self).__init__()
        self.convLayers = [];
        self.batchNormal = [];
        self.fullyConnectedLayers = []
        convLayerChannels.append(convLayerChannels[-1])
        for index, channels in enumerate(convLayerChannels):
            self.convLayers.append(
                nn.Conv2d(channels, convLayerChannels[index + 1], kernel_size=kernel_size, stride=stride))
            self.bn.append(nn.BatchNorm2d(channels))

        def conv2d_size_out(size, kernel_size=kernel_size, stride=stride):
            return ((size - kernel_size - 1) - 1) // (stride + 1)

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(width)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(height)))
        linearInputSize = convw * convh * convLayerChannels[-1]

        for output in fullyConnectedLayers:
            self.append(nn.Linear(linearInputSize, output))
            linearInputSize = output

        self.outputLayer = nn.Linear(linearInputSize, finalOutputs)
        if parameters:
            self.load_state_dict(parameters)

    def forward(self, x):
        for convLayer, batchNorm in zip(self.convLayers, self.batchNormal):
            x = F.relu(batchNorm(convLayer(x)))

        for layer in self.fullyConnectedLayers:
            x = F.relu(layer(x))

        return self.outputLayer(x.view(x.size(0), -1))


class TrainingModel:

    def __init__(self, batchSize=200, gamma=0.99, epsilonStart=0.9, epsilonEnd=0.05, epsilonDecay=200, targetUpdate=10):
        self.batch_size, self.gamma, self.epsilonStart = batchSize, gamma, epsilonStart
        self.epsilonEnd, self.epsilonDecay, self.targetUpdate = epsilonEnd, epsilonDecay, targetUpdate
        self.policyNet = self.targetNet = None
        self.stepsDone = 0
        return self

    def createDQN(self, height, width, numberOfActions, kernel_size=3, stride=2, convLayerChannels=[9, 18, 36],
                  fullyConnectedLayers=[]):
        self.policyNet = DQN(height, width, numberOfActions, kernel_size, stride, convLayerChannels,
                             fullyConnectedLayers)
        self.targetNet = DQN(height, width, numberOfActions, kernel_size, stride, convLayerChannels,
                             fullyConnectedLayers,
                             self.policyNet.state_dict())
        self.numberOfActions = numberOfActions
        self.targetNet.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        return self

    def epsThreshold(self):
        return self.epsilonEnd + (self.epsilonStart - self.epsilonEnd) * \
               math.exp(-1. * self.stepsDone / self.epsilonDecay)

    def selectAction(self, state):
        sample = random.random()
        epsThreshold = self.epsThreshold()
        self.stepsDone += 1
        if sample > epsThreshold:
            with torch.no_grad():
                return self.policyNet(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.numberOfActions)]], device=device, dtype=torch.long)

    def updateTargetNet(self, parameters):
        self.targetNet.load_state_dict(parameters)

    def optimize(self, memory):
        if len(self.memory) < self.batch_size:
            return
        transitions = memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        nonFinalMask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.bool)
        nonFinalNextStates = torch.cat([s for s in batch.next_state
                                        if s is not None])

        stateBatch = torch.cat(batch.state)
        actionBatch = torch.cat(batch.action)
        rewardBatch = torch.cat(batch.reward)

        stateActionValues = self.policyNet(stateBatch).gather(1, actionBatch)

        nextStateValues = torch.zeros(self.batch_size, device=device)
        nextStateValues[nonFinalMask] = self.targetNet(nonFinalNextStates).max(1)[0].detach()

        expectedStateActionValues = (nextStateValues * self.gamma) + rewardBatch

        loss = F.smooth_l1_loss(stateActionValues, expectedStateActionValues.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policyNet.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    
