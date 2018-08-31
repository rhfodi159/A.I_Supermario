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
RENDER_ENV=True


input_height = env.observation_space.shape[0]
input_width = env.observation_space.shape[1]
input_channel = env.observation_space.shape[2]

conv_n_maps = [32,64,64]
conv_kernel_sizes = [(8,8),(4,4),(3,3)]
conv_strides = [4,2,1]
conv_paddings= ['SAME']*3
conv_activation = [tf.nn.relu] * 3
n_hidden_in =64 * 11 * 10
n_hidden = 512
hidden_activatin = tf.nn.relu
n_outputs = env.action_space
initializer = tf.contrib.layers.variance_scaling_initializer()  
