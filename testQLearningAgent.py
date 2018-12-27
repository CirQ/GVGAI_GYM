#!/usr/bin/env python
import gym
import gym_gvgai
import Agent

env = gym_gvgai.make('gvgai-aai-lvl0-v0')
agent = Agent.Agent('awesome/str_e-6level_reward_awesome')
actions = env.env.GVGAI.actions()
print('Starting ' + env.env.game + " with Level " + str(env.env.lvl))

for e in range(100):
    stateObs = env.reset()
    total_score = 0
    for t in range(1000):
        action_id = agent.act(stateObs, actions)
        stateObs, increScore, done, debug = env.step(action_id)
        total_score += increScore
        if done:
            break
    # episode, score, tick, win
    print(e, total_score, t+1, debug['winner'], sep=',')
