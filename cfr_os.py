################################################## MULTI-GRID ##################################################
import tkinter as tk
import numpy as np
import time
from random import randint
from datetime import datetime
import random
import math

# 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
expl = 0.12 # epsilon
N_ROW = 10 # number of rows
N_COL = 10 # number of columns
# GRID SETTINGS
EDGE = 30
SQUARE_SIZE = 60
DIRECTIONS = [[0, -SQUARE_SIZE], [SQUARE_SIZE, 0], [-SQUARE_SIZE, 0], [0, SQUARE_SIZE]] # UP, RIGHT, LEFT, DOWN

START = [1,1] # start position
TARGET = [4,7] # target position
OBSTACLES = [[1,2],[2,2],[3,2],[4,2],[7,3],[8,8],[3,5],[3,6],[3,7],[3,8]]

NUM_ACTIONS = 4 # number of actions
TIME_SLEEP = 0.10 # speed of agent
MAX_HISTORY = 200 # maximum depth of sequence of actions
ACTION_LIST = [0,1,2,3]
FORBID_ACTION = [1,0,3,2] # forbid agent to move backward
IS_RANDOM_START = False
nodeMap = {}
reward = {}
class Gridworld:
    def __init__(self):
        self.root = tk
        self.frame = self.root.Canvas(bg='black', height=N_ROW*200+EDGE+100, width=N_COL*200+EDGE+100)
        self.frame.pack()

    def grid(self):
        '''world'''
        self.frame.create_rectangle(EDGE, EDGE, EDGE+N_COL*SQUARE_SIZE, EDGE+N_ROW*SQUARE_SIZE, fill='black')
        """specials"""
        """target"""
        self.frame.create_rectangle(EDGE+TARGET[1]*SQUARE_SIZE, EDGE+TARGET[0]*SQUARE_SIZE, EDGE+TARGET[1]*SQUARE_SIZE+SQUARE_SIZE,  EDGE+TARGET[0]*SQUARE_SIZE+SQUARE_SIZE, fill='lime green') # target
        """obstacle"""
        for OBSTACLE in OBSTACLES:
            self.frame.create_rectangle(EDGE+OBSTACLE[1]*SQUARE_SIZE, EDGE+OBSTACLE[0]*SQUARE_SIZE, EDGE+OBSTACLE[1]*SQUARE_SIZE+SQUARE_SIZE,  EDGE+OBSTACLE[0]*SQUARE_SIZE+SQUARE_SIZE, fill='grey') # obstacle
        """frame line"""
        self.frame.create_line(EDGE, EDGE, EDGE+N_COL*SQUARE_SIZE, EDGE, EDGE+N_COL*SQUARE_SIZE, EDGE+N_ROW*SQUARE_SIZE, EDGE, EDGE+N_ROW*SQUARE_SIZE, EDGE, EDGE, fill='white', width=5)
        """horizontal lines"""
        for i in range(N_ROW-1):
            self.frame.create_line(EDGE, EDGE+SQUARE_SIZE*(i+1), EDGE+N_COL*SQUARE_SIZE, EDGE+SQUARE_SIZE*(i+1), fill='white', width=2)
        """virtical lines"""
        for i in range(N_COL-1):
            self.frame.create_line(EDGE+SQUARE_SIZE*(i+1), EDGE, EDGE+SQUARE_SIZE*(i+1), EDGE+N_ROW*SQUARE_SIZE, fill='white', width=2)

        self.frame.update()
        #self.root.mainloop()

    def get_action(self):
            self.move = np.random.choice(['up','down', 'left', 'right'])

    def reset(self):
        self.frame.delete(self.agent)
        if IS_RANDOM_START:
            while START in OBSTACLES or START == TARGET:
              START[0] = randint(0, N_ROW-1)
              START[1] = randint(0, N_COL-1)

        points = self.get_agent_points(START[0], START[1], 3)
        self.agent = self.frame.create_polygon(points, fill="cyan", outline="white")
        self.frame.update()
        time.sleep(TIME_SLEEP)

    def initialize_agent(self):
        self.grid()
        points = self.get_agent_points(START[0], START[1], 3)
        self.agent = self.frame.create_polygon(points, fill="cyan", outline="white")
        self.frame.update()
        time.sleep(TIME_SLEEP)

    def move_agent(self, row, col, action):
        self.frame.delete(self.agent)
        points = self.get_agent_points(row, col, action)
        self.agent = self.frame.create_polygon(points, fill="cyan", outline="white")
        self.frame.update()

    def get_agent_points(self, row, col, action):
        agent_t = SQUARE_SIZE//4
        if action == 0:
            points = [EDGE+SQUARE_SIZE//2+col*SQUARE_SIZE, EDGE+agent_t+row*SQUARE_SIZE, EDGE+agent_t+col*SQUARE_SIZE, EDGE+SQUARE_SIZE//2+agent_t+row*SQUARE_SIZE, EDGE+SQUARE_SIZE//2+agent_t+col*SQUARE_SIZE, EDGE+SQUARE_SIZE//2+agent_t+row*SQUARE_SIZE]
        elif action == 1:
            points = [EDGE+SQUARE_SIZE//2+col*SQUARE_SIZE, EDGE+SQUARE_SIZE//2+agent_t+row*SQUARE_SIZE, EDGE+agent_t+col*SQUARE_SIZE, EDGE+agent_t+row*SQUARE_SIZE, EDGE+SQUARE_SIZE//2+agent_t+col*SQUARE_SIZE, EDGE+agent_t+row*SQUARE_SIZE]
        elif action == 2:
            points = [EDGE+SQUARE_SIZE//2+agent_t+col*SQUARE_SIZE, EDGE+agent_t+row*SQUARE_SIZE, EDGE+SQUARE_SIZE//2+agent_t+col*SQUARE_SIZE, EDGE+SQUARE_SIZE//2+agent_t+row*SQUARE_SIZE, EDGE+agent_t+col*SQUARE_SIZE, EDGE+SQUARE_SIZE//2+row*SQUARE_SIZE]
        elif action == 3:
            points = [EDGE+agent_t+col*SQUARE_SIZE, EDGE+agent_t+row*SQUARE_SIZE, EDGE+agent_t+col*SQUARE_SIZE, EDGE+SQUARE_SIZE//2+agent_t+row*SQUARE_SIZE, EDGE+SQUARE_SIZE//2+agent_t+col*SQUARE_SIZE, EDGE+SQUARE_SIZE//2+row*SQUARE_SIZE]
        return points

################################################## CFR ##################################################


class Node:
    def __init__(self):
        self.history = ""
        self.regretSum = np.zeros(NUM_ACTIONS)
        self.strategy = np.zeros(NUM_ACTIONS)
        self.strategySum = np.zeros(NUM_ACTIONS)

    def get_strategy(self):
        normalizingSum = 0
        curX = int(self.history[0])
        curY = int(self.history[1])
        arr = [math.dist([curX-1, curY], TARGET), math.dist([curX+1, curY], TARGET), math.dist([curX, curY-1], TARGET), math.dist([curX, curY+1], TARGET)]
        sumarr = sum(arr)
        newarr = []
        for num in arr:
          newarr.append(num/sumarr)
        dp = {}
        for i in range(4):
            if newarr[i] in dp:
                dp[newarr[i]].append(i)
            else:
                dp[newarr[i]] = [i]
        order_arr = []
        for key, val in dp.items():
          order_arr.append([key,val])
        order_arr = sorted(order_arr, key=lambda x:x[0])

        for a in range(NUM_ACTIONS):
            self.strategy[a] = self.regretSum[a] if self.regretSum[a] > 0 else 0
            normalizingSum += self.strategy[a]

        if normalizingSum > 0:
          for a in range(NUM_ACTIONS):
            self.strategy[a] /= normalizingSum
        else:
            if len(order_arr) == 4:
              ranking = [0.8, 0.125, 0.05, 0.025]
              for i in range(len(order_arr)):
                self.strategy[order_arr[i][1]] = ranking[i]
            elif len(order_arr) == 3:
              ranking = [0.8, 0.15, 0.05]
              for i in range(len(order_arr)):
                for j in range(len(order_arr[i][1])):
                  self.strategy[order_arr[i][1][j]] = ranking[i]/(len(order_arr[i][1]))
            elif len(order_arr) == 2:
              ranking = [0.8,0.2]
              for i in range(len(order_arr)):
                for j in range(len(order_arr[i][1])):
                  self.strategy[order_arr[i][1][j]] = ranking[i]/(len(order_arr[i][1]))
            else:
              for a in range(NUM_ACTIONS):
                self.strategy[a] = 1.0 / NUM_ACTIONS
        for a in range(NUM_ACTIONS):
            self.strategySum[a] +=  self.strategy[a]
        return self.strategy

    def get_average_strategy(self):
        avgStrategy = np.zeros(NUM_ACTIONS)
        normalizingSum = 0
        for a in range(NUM_ACTIONS):
            normalizingSum += self.strategySum[a]
        for a in range(NUM_ACTIONS):
            if (normalizingSum > 0):
                avgStrategy[a] = round(self.strategySum[a] / normalizingSum, 2)
            else:
                avgStrategy[a] = round(1.0 / NUM_ACTIONS)
        return avgStrategy

    def get_action(self, strategy):
        return np.random.choice(np.arange(NUM_ACTIONS), 1, p=strategy)
    
    def get_distance(self, curCoord, target):
      d = math.sqrt((math.pow((curCoord[0]-target[0]),2))+(math.pow((curCoord[1]-target[1]),2)))
      return d

    def __str__(self):
        # + "; regret = " + str(self.regretSum)
        return self.history + ": " + str(self.get_average_strategy())


def train(iterations):
    util = 0
    grid = Gridworld()
    grid.initialize_agent()
    startCoord = str(START[0])+str(START[1])
    node = Node()
    node.history = startCoord
    nodeMap[startCoord] = node
    cur_expl = expl
    for i in range(iterations):
        print('iteration ', i+1)
        print('epsilon: ', cur_expl)
        util += cfr(startCoord, START, grid, 0, -1, cur_expl)
        cur_expl =max(0,cur_expl-expl/(iterations-1))


def getCoord(curCoord, action, grid):
    s = grid.frame.coords(grid.agent)
    newCoord = curCoord
    if action == 0: # up
        if curCoord[0] > 0:
          for OBSTACLE in OBSTACLES:
            if curCoord[0] == OBSTACLE[0]+1 and curCoord[1] == OBSTACLE[1]:
                return newCoord
          newCoord = [curCoord[0]-1, curCoord[1]]
          grid.move_agent(newCoord[0], newCoord[1], action)
          time.sleep(TIME_SLEEP)
          
    elif action == 1: # down
      if curCoord[0] < N_ROW-1:
        for OBSTACLE in OBSTACLES:
            if curCoord[0] == OBSTACLE[0]-1 and curCoord[1] == OBSTACLE[1]:
                return newCoord
        newCoord = [curCoord[0]+1, curCoord[1]]
        grid.move_agent(newCoord[0], newCoord[1], action)
        time.sleep(TIME_SLEEP)
    elif action == 2: # left
      if curCoord[1] > 0:
        for OBSTACLE in OBSTACLES:
            if curCoord[1] == OBSTACLE[1]+1 and curCoord[0] == OBSTACLE[0]:
                return newCoord
        newCoord = [curCoord[0], curCoord[1]-1]
        grid.move_agent(newCoord[0], newCoord[1], action)
        time.sleep(TIME_SLEEP)
    elif action == 3: # right
      if curCoord[1] < N_COL-1:
        for OBSTACLE in OBSTACLES:
            if curCoord[1] == OBSTACLE[1]-1 and curCoord[0] == OBSTACLE[0]:
                return newCoord
        newCoord = [curCoord[0], curCoord[1]+1]
        grid.move_agent(newCoord[0], newCoord[1], action)
        time.sleep(TIME_SLEEP)
    # grid.root.mainloop()
    return newCoord


def cfr(history, curCoord, grid, step, prevAction, cur_expl):
    if curCoord == TARGET:
        grid.reset()
        return 1000000-step
    elif step == MAX_HISTORY:
        grid.reset()
        return -step
    node = None
    if history in nodeMap:
        node = nodeMap[history]
    else:
        node = Node()
        node.history = history
        nodeMap[history] = node
    strategy = node.get_strategy()
    util = np.zeros(NUM_ACTIONS)

    nodeUtil = 0
    epsilon = random.uniform(0,1)
    if epsilon > (1-cur_expl):
        for a in range(NUM_ACTIONS):
            strategy[a] = 1.0 / NUM_ACTIONS
    a = random.choices(ACTION_LIST, weights=(strategy), k=1)[0] # get the action based on the strategy
    if prevAction != -1:
        i = 0
        while a == FORBID_ACTION[prevAction] and i < 10:
            a = random.choices(ACTION_LIST, weights=(strategy), k=1)[0] # get the action based on the strategy
            i+=1
        if a == FORBID_ACTION[prevAction]:
            strategy[FORBID_ACTION[prevAction]] = 0
            for i in range(NUM_ACTIONS):
                if i != FORBID_ACTION[prevAction]:
                    strategy[i] = 1.0 / (NUM_ACTIONS-1)
            a = random.choices(ACTION_LIST, weights=(strategy), k=1)[0]
    newCoord = getCoord(curCoord, a, grid)
    nextHistory = str(newCoord[0])+str(newCoord[1])
    if newCoord != curCoord:
        util[a] = cfr(nextHistory, newCoord, grid, step+1, a, cur_expl)
    else:
        util[prevAction] = cfr(nextHistory, newCoord, grid, step+1, prevAction, cur_expl)
    nodeUtil += strategy[a] * util[a]
    
    # calculate regret
    regret = util[a] - nodeUtil
    node.regretSum[a] += regret
    
    return nodeUtil


now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Train Time =", current_time)

iterations = 10

# number of iterations (10x10)
# 10 iterations approx 1min
# 100 iterations approx 4min
# 1000 iterations approx 30min
print("iterations =", iterations)
train(iterations)

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End Train Time =", current_time)

for key, value in reward.items():
    print(key, ": ", value)
for infoSet in nodeMap:
    print(nodeMap[infoSet].__str__())


