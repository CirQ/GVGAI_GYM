#!/usr/bin/env python
import gym
import gym_gvgai
import Agent as Agent

env = gym_gvgai.make('gvgai-aai-lvl0-v0')

epsilon = 1.0
MAX_EPISODE = 10000
MAX_STEP = 1000

for r in range(MAX_EPISODE):
    epsilon = 0.000001
    agent = Agent.Agent('str_10_100_reward', epsilon, reuse=True)
    env.reset()
    actions = env.env.GVGAI.actions()
    for t in range(MAX_STEP):
        done, debug = agent.train_act(env, actions)
        if done:
            break
    print("episode", r, "epsilon", epsilon, "gameTick", t, "winState", debug["winner"])
