#!/usr/bin/env python
import gym
import gym_gvgai
import Agent as Agent

env = gym_gvgai.make('gvgai-aai-lvl0-v0')
agent = Agent.Agent()
print('Starting ' + env.env.game + " with Level " + str(env.env.lvl))
stateObs = env.reset()
actions = env.env.GVGAI.actions()

for t in range(2000):
    done, debug = agent.train_act(env, actions)

    print("gameTick", t, "winState", debug["winner"])
    if done:
        break
