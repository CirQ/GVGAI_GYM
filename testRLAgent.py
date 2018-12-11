#!/usr/bin/env python
import gym
import gym_gvgai
import Agent as Agent

env = gym_gvgai.make('gvgai-aai-lvl0-v0')
agent = Agent.Agent()
print('Starting ' + env.env.game + " with Level " + str(env.env.lvl))
# reset environment
stateObs = env.reset()
actions = env.env.GVGAI.actions()
for t in range(2000):
    # choose action based on trained policy
    action_id = agent.act(stateObs, actions)
    # do action and get new state and its reward
    stateObs, increScore, done, debug = env.step(action_id)
    print("Action " + str(action_id) + " tick " + str(t+1) + " reward " + str(increScore) + " win " + debug["winner"])
    # break loop when terminal state is reached
    if done:
        break
