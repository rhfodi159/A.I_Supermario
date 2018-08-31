from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
import numpy as np
#from gym_super.gym_super_mario_bros_wra.actions import SIMPLE_MOVEMENT

"""Static action sets for binary to discrete action space wrappers."""


# actions for the simple run right environment
RIGHT_ONLY = [
    ['NOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
]


# actions for very simple movement
SIMPLE_MOVEMENT = [
    ['NOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['left'],
]


# actions for more complex movement
COMPLEX_MOVEMENT = [
    ['NOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['left'],
    ['left', 'A'],
    ['left', 'B'],
    ['left', 'A', 'B'],
    ['down'],
]
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, COMPLEX_MOVEMENT)
done = True

for i in range(1000):
    if done:
        state = env.reset()

    action = env.action_space.sample()

    observation , reward, done, info = env.step(action)

    print(state)

    print(reward)

    #env.render()

env.close()
