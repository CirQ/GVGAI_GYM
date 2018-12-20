from collections import namedtuple
from enum import IntEnum
from itertools import product
import pickle
import random

import numpy as np


class Sprite(IntEnum):
    Background = 154
    Base = 8114
    Avatar = 15895
    Sam = 1012
    Bomb = 5356
    Alien = 11526

class Action(IntEnum):  # assign names to action id
    NIL = 0
    USE = 1
    LEFT = 2
    RIGHT = 3

    @classmethod
    def random_choice(cls): # randomly choose an action
        all_choices = [
            cls.USE,
            cls.LEFT,
            cls.RIGHT ]
        return random.choice(all_choices)



class QTableError(Exception):   # base error about QTable
    pass


class QTable(object):       # a Q table
    def __init__(self):
        self.table = {}

    def __check_key(self, key):         # check if is int or is the form (hash, action)
                                        # return True if is int, False if is tuple
        if isinstance(key, int):
            if key not in self.table:   # add a new row if no such key
                self.table[key] = np.array([0.0, 0.0, 0.0, 0.0])
            return True
        else:
            if len(key) != 2:
                raise QTableError('invalid key: '+str(key))
            if not isinstance(key[0], int):
                raise QTableError('invalid state hash: '+str(key[0]))
            if key[1] not in Action:
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
        if not isinstance(key, int):    # delete a state, to reduce size, e.g., del table[233]
            raise QTableError('invalid state hash to delete: '+str(key))
        if key not in self.table:
            raise QTableError('no such state hash: '+str(key))
        del self.table[key]




class Agent(object):
    def __init__(self, alpha=0.1, gamma=0.9, Qtable=None):
        self.name = 'Agent11849180'
        self.Qtable = self._load_Qtable(Qtable)
        self.gamegrid = (9, 10)
        self.factor = 10
        self.rolling_a = 69576527601013534054379470050410071764306367636630478633119489403438659847711
        self.rolling_N = 102755400274853561541650296157828008998572747603972956759050025850260627630159
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = 1.0

    def _load_Qtable(self, qtable):
        if qtable:
            self.model_name = qtable
            with open(qtable, 'rb') as r:
                return pickle.load(r)
        else:
            self.model_name = 'q-table.pickle'
            return QTable()

    def _dump_Qtable(self):
        with open(self.model_name, 'wb') as w:
            pickle.dump(self.Qtable, w)

    def _convolution(self, gamemap):
        # kernal = np.ones(shape=(self.factor, self.factor, 3), dtype=np.uint16)
        conv = np.zeros(shape=self.gamegrid, dtype=np.uint16)
        height, width = self.gamegrid
        for i, j in product(range(height), range(width)):
            area = gamemap[i*self.factor:(i+1)*self.factor,j*self.factor:(j+1)*self.factor,:3]
            conv[i,j] = np.sum(area)    # equivalent to all kernal elements are 1s
        return conv

    def _rolling_hash(self, gamemap):
        H = 0
        for c in gamemap.reshape(-1):
            H += (self.rolling_a*H + c) % self.rolling_N
        return H

    def hashing(self, gamemap):
        return self._rolling_hash(self._convolution(gamemap))

    def get_next_action(self, state):
        if random.random() > self.epsilon:  # do exploitation
            return self.Qtable[state].max()
        else:                               # do exploration
            return Action.random_choice()

    def update_Qtable(self, state0, act, state1, reward):
        currentQ = self.Qtable[state0, act]
        deltaQ = reward + self.gamma * self.Qtable[state1].max() - currentQ
        self.Qtable[state0, act] = currentQ + self.alpha * deltaQ


    def train_act(self, env, actions):
        this_state = self.hashing(env.env.img)
        action = self.get_next_action(this_state)
        stateObs, increScore, done, debug = env.step(action)
        new_state = self.hashing(stateObs)
        env.render()
        self.update_Qtable(this_state, action, new_state, increScore)
        if done:
            self._dump_Qtable()
        return done, debug

    def act(self, stateObs, actions):
        action_id = random.randint(0,len(actions)-1)
        return action_id
