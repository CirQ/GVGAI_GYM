from collections import namedtuple
from enum import IntEnum
from itertools import product
import pickle
import random

import numpy as np

KeyType = str

class Sprite(IntEnum):
    Background = 59
    Base = 162
    Avatar = 61
    Sam = 134
    Bomb = 133 # 112 75
    Alien = 24
    # below is combined pictures
    Bomb_a = 112
    Bomb_b = 75
    Alien_Bomb = 19

class Action(IntEnum):  # assign names to action id
    NIL = 0
    USE = 1
    LEFT = 2
    RIGHT = 3

    @classmethod
    def random_action(cls): # randomly choose an action
        all_choices = [
            cls.NIL,
            cls.USE,
            cls.LEFT,
            cls.RIGHT ]
        return random.choice(all_choices)

    @classmethod
    def has_action(cls, v):
        return (v in cls) or any((v==e.value) for e in cls)


class QTableError(Exception):   # base error about QTable
    pass

class QTable(object):       # a Q table
    def __init__(self):
        self.table = {}

    def __check_key(self, key):         # check if is int or is the form (hash, action)
                                        # return True if is int, False if is tuple
        if isinstance(key, KeyType):
            if key not in self.table:   # add a new row if no such key
                self.table[key] = np.array([0.0, 0.0, 0.0, 0.0])
            return True
        else:
            if len(key) != 2:
                raise QTableError('invalid key: '+str(key))
            if not isinstance(key[0], KeyType):
                raise QTableError('invalid state hash: '+str(key[0]))
            if not Action.has_action(key[1]):
                raise QTableError('invalid action: '+str(key[1]))
            if key[0] not in self.table:    # add a new row if no such key
                self.table[key[0]] = np.array([0.0, 0.0, 0.0, 0.0])
            return False

    def __getitem__(self, key):
        if self.__check_key(key):           # get the whole row, e.g., table[233]
            return self.table[key]
        return self.table[key[0]][key[1]]   # get specific item, e.g., table[233, USE]

    def __setitem__(self, key, value):
        if self.__check_key(key):           # set the whole row, e.g., table[233] = 4
            self.table[key] = value * np.ones(4)
            return
        self.table[key[0]][key[1]] = value  # set specific item, e.g., table[233, USE] = 1

    def __delitem__(self, key):
        if not isinstance(key, int):        # delete a state, to reduce size, e.g., del table[233]
            raise QTableError('invalid state hash to delete: '+str(key))
        if key not in self.table:
            raise QTableError('no such state hash: '+str(key))
        del self.table[key]




class Agent(object):
    def __init__(self, model_name, epsilon=None, alpha=0.9, gamma=0.9, reuse=True):
        self.name = 'Agent11849180'
        self.model_name = model_name
        self.Qtable = self._load_Qtable(reuse)
        # for game map
        self.gamegrid = (9, 10)
        self.factor = 10
        # for state hash
        self.rolling_a = 69576527601013534054379470050410071764306367636630478633119489403438659847711
        self.rolling_N = 102755400274853561541650296157828008998572747603972956759050025850260627630159
        # for bellman equation
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.total_score = 0
        self.ZERO = np.zeros((self.gamegrid[0],1), dtype=np.uint8)

    def _load_Qtable(self, reuse):      # load an existing model
        if reuse:
            with open(self.model_name+'.pickle', 'rb') as r:
                return pickle.load(r)
        else:
            return QTable()

    def _dump_Qtable(self):             # dump the current model
        with open(self.model_name+'.pickle', 'wb') as w:
            pickle.dump(self.Qtable, w)

    Position = namedtuple('Position', ['r', 'c'])

    def _constrain_map(self, gamemap):
        conv = self._convolution(gamemap)
        conv = np.concatenate([self.ZERO, conv, self.ZERO], axis=1)
        avatar_position = np.where(conv==Sprite.Avatar)
        try:
            ar = avatar_position[0][0]
            ac = avatar_position[1][0]
            self.last_pos = Agent.Position(ar, ac)
        except IndexError:
            pass
        row_select = slice(self.last_pos.r-4, None, 1)
        column_select = slice(self.last_pos.c-1, self.last_pos.c+2)
        return conv[row_select, column_select]

    def _convolution(self, gamemap):    # perform a convolution on the game map
        # kernal = np.ones(shape=(self.factor, self.factor, 3), dtype=np.uint16)
        conv = np.zeros(shape=self.gamegrid, dtype=np.uint8)
        height, width = self.gamegrid
        gamemap = gamemap[:,:,0]
        for i, j in product(range(height), range(width)):
            area = gamemap[i*self.factor:(i+1)*self.factor,j*self.factor:(j+1)*self.factor]
            conv[i,j] = np.sum(area)    # equivalent to all kernal elements are 1s
        return conv

    def _rolling_hash(self, gamemap):   # rolling hash algorithm
        H = 0
        for c in gamemap.reshape(-1):
            H += (self.rolling_a*H + c) % self.rolling_N
        return H

    Target = [
        Sprite.Base,
        Sprite.Alien,
        Sprite.Bomb,
        Sprite.Bomb_a,
        Sprite.Bomb_b,
        Sprite.Alien_Bomb,
    ]

    def hashing(self, gamemap):         # for state compress
        reduce_map = self._constrain_map(gamemap)
        binary_map = np.isin(reduce_map, Agent.Target).astype(np.uint8)
        if KeyType == int:
            return self._rolling_hash(reduce_map)
        if KeyType == str:
            return str(tuple(map(tuple, binary_map)))
        raise QTableError('unknown key type: '+str(KeyType))

    def get_next_action(self, state):       # epsilon-greedy strategy
        if random.random() > self.epsilon:  # do exploitation
            return self.Qtable[state].argmax()
        else:                               # do exploration
            return Action.random_action()

    reward_inc = lambda s: (s+1)*10 if s>0 else s*100

    def update_Qtable(self, state0, act, state1, reward):   # Q learning update strategy
        reward = Agent.reward_inc(reward)
        currentQ = self.Qtable[state0, act]
        deltaQ = reward + self.gamma * self.Qtable[state1].max() - currentQ
        self.Qtable[state0, act] = currentQ + self.alpha * deltaQ


    def train_act(self, env, actions):
        this_state = self.hashing(env.env.img)                          # record this state
        action = self.get_next_action(this_state)                       # get new action
        stateObs, increScore, done, debug = env.step(action)            # one-step forward
        new_state = self.hashing(stateObs)                              # record new state
        env.render()
        self.update_Qtable(this_state, action, new_state, increScore)   # update q table

        self.total_score += increScore              # debuging message
        if done:                                    # game is over, output score
            print('Score is:', self.total_score)
            self._dump_Qtable()
        return done, debug

    def act(self, stateObs, actions):
        state = self.hashing(stateObs)
        return self.Qtable[state].argmax()
