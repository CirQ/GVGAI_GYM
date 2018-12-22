#!/usr/bin/env python
import gym
import gym_gvgai
import Agent as Agent

env = gym_gvgai.make('gvgai-aai-lvl0-v0')

epsilon = 1.0
MAX_EPISODE = 10000

for r in range(MAX_EPISODE):
    epsilon *= 0.9995   # minmum is 0.006729527022146667
    agent = Agent.Agent(epsilon, reuse=True)
    stateObs = env.reset()
    actions = env.env.GVGAI.actions()
    for t in range(1000):
        done, debug = agent.train_act(env, actions)
        if done:
            break
    print("episode", r, "epsilon", epsilon, "gameTick", t, "winState", debug["winner"])
