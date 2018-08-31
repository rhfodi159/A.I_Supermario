from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
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


"""
Template

SuperMArioBros- world - level - V<version>

world : {1,2,3,4,5,6,7,8}

level : {1,2,3,4}

version : {0,1,2,3}

version 0 : standard , 1: downsampling , 2: pixel, 3:rectangle

NoFrameskip : added before the first hyphen in order to disable frame skip

example

to play world 3 and level 4 using downsample

SuperMarioBros-3-4-v1
"""
env = gym_super_mario_bros.make('SuperMarioBros-8-4-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, COMPLEX_MOVEMENT)
done = True
for step in range(5000):
    if done:
        state = env.reset()
    state, reward, done, indfo = env.step(env.action_space.sample())
    print(reward)
    env.render()

env.close()
