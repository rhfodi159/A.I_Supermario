import tensorflow as tf
import numpy as np

from tensorflow.python.framework import ops


from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
import numpy as np
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

env.reset()
next_state, reward, done, info = env.step(3)

print(info)
