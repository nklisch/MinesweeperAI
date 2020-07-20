import gym
from gym import spaces
import numpy as np
import random

# channels to represent adjacent bomb locations and one for all spaces that have been uncovered so far
CHANNELS = 9


def isBomb(value):
    return value != 0


def countAdjacentBombs(x, y, bombBoard):
    bombs = 0
    for xMod in (range(-1, 2)):
        for yMod in range(-1, 2):
            currentX, currentY = (x + xMod), (y + yMod)
            if 0 <= currentX < bombBoard.shape[0] and 0 <= currentY < bombBoard.shape[1]:
                if not (currentX == x and currentY == y):
                    if isBomb(bombBoard[currentX][currentY]):
                        bombs += 1
    return bombs


def createBoardWithBombs(shape, ratioOfBombs, seed=None):
    size = shape[0] * shape[1]
    numOfBombs = int(size * ratioOfBombs)
    random.seed(seed)
    bombIndex = random.sample(range(0, size), numOfBombs)
    bombBoard = np.zeros(shape=size)
    bombBoard[bombIndex] = 1
    return bombBoard.reshape(shape), numOfBombs


def firstMoveNeverLose(x, y, bombBoard):
    if isBomb(bombBoard[x][y]):
        bombBoard[x][y] = 0
        for i in range(0, bombBoard.shape[0]):
            for j in range(0, bombBoard.shape[1]):
                if bombBoard[i][j] == 0 and i != x and j != y:
                    bombBoard[i][j] = 1


def createBombCountBoard(bombBoard):
    bombCountBoard = np.zeros(bombBoard.shape)
    for i in range(0, bombBoard.shape[0]):
        for j in range(0, bombBoard.shape[1]):
            bombCountBoard[i][j] = countAdjacentBombs(i, j, bombBoard)
    return bombCountBoard


def decodeAction(action, boardShape):
    x = action % boardShape[0]
    y = action / boardShape[0]
    return x, y


class MinesweeperEnv(gym.Env):

    def __init__(self, shape, ratioOfBombs=0.20):
        self.action_space = spaces.Discrete(shape[0] * shape[1])
        self.ratioOfBombs = ratioOfBombs
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(CHANNELS, shape[0], shape[1]))
        self.shape = shape
        self.board, self.bombBoard, self.bombCountBoard, self.numOfBombs, self.seed, self.firstMove = None
        self.reset()

    def autoUncover(self, x, y):
        for xMod in range(-1, 2):
            for yMod in range(-1, 2):
                adjacentBombs = self.bombCountBoard[currentX][currentY]
                currentX, currentY = (x + xMod), (y + yMod)
                if 0 <= currentX < self.bombBoard.shape[0] and 0 <= currentY < self.bombBoard.shape[1]:
                    if self.board[0][currentX][currentY] == 0:
                        if adjacentBombs == 0:
                            self.board[0][currentX][currentY] = 1
                            self.autoUncover(currentX, currentY)
                        elif not isBomb(self.bombBoard[currentX][currentY]):
                            self.board[adjacentBombs][currentX][currentY] = 1

    def areWeDone(self, x, y):
        if isBomb(self.bombBoard[x][y]):
            return True, -1
        elif self.board[self.board == 1].sum() <= self.numOfBombs:
            return True, 1
        return False, 0

    def step(self, action):
        x, y = decodeAction(action, self.board.shape)
        if self.firstMove:
            firstMoveNeverLose(x, y, self.bombBoard)
            self.bombCountBoard = createBombCountBoard(self.bombBoard)
            self.firstMove = False

        adjacentBombs = self.bombCountBoard[x][y]
        self.board[0][x][y] = 1
        self.board[adjacentBombs][x][y] = 1

        done, reward = self.areWeDone()
        if not done and adjacentBombs == 0:
            self.autoUncover(x, y)

        return self.board, reward, done, {}

    def reset(self):
        self.firstMove = True
        self.board = np.zeros((CHANNELS, self.shape[0], self.shape[1]))
        self.bombBoard, self.numOfBombs = createBoardWithBombs(self.shape, self.ratioOfBombs, self.seed)

    def seed(self, seed=None):
        self.seed = seed

    def close(self):
        return None

    def render(self, mode='human'):
        for x, row in enumerate(self.board[0]):
            for y, cell in enumerate(row):
                if cell == 0:
                    cell = '-'
                elif isBomb(self.bombBoard[x][y]):
                    cell = 'X'
                elif self.bombCountBoard[x][y] == 0:
                    cell = ' '
                elif self.bombCountBoard[x][y] > 0:
                    cell = self.bombCountBoard[x][y]

                print('''|{}'''.format(cell), end='')
            print('|')
