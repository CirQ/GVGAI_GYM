#!/usr/bin/env python
import gym
import gym_gvgai
import Agent as Agent

env = gym_gvgai.make('gvgai-aai-lvl0-v0')

MAX_EPISODE = 1000
MAX_STEP = 1000

for r in range(MAX_EPISODE):
    if r < 100:
        epsilon = 0.5
    elif r < 200:
        epsilon = 0.1
    elif r < 300:
        epsilon = 0.01
    elif r < 400:
        epsilon = 0.001
    else:
        epsilon = 0.0001
    agent = Agent.Agent('str_inc_eps', epsilon, reuse=True)
    env.reset()
    actions = env.env.GVGAI.actions()
    for t in range(MAX_STEP):
        done, debug, score = agent.train_act(env, actions)
        if done:
            break
    print("episode", r, "epsilon", epsilon, "score", score, "gameTick", t, "winState", debug["winner"])
