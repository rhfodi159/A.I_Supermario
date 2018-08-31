

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


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import timeit
import math
import os
import sys

#DRQN 클래스 정의


# CNN 레이어를 통과한 뒤 이미지의 최종 shape를 계산한다.
def get_input_shape(Image,Filter,Stride):
    layer1=math.ceil((Image - Filter + 1) / Stride)
    o1 = math.ceil((layer1 / Stride))
    layer2 = math.ceil(((o1 - Filter + 1)/ Stride))
    o2 = math.ceil((layer2 / Stride))
    layer3 = math.ceil(((o2 - Filter + 1) / Stride))
    o3 = math.ceil((layer3 / Stride ))
    return int(o3)

# DRQN 클래스를 정의하겠습니다.

class DRQN():

    def __init__(self, input_shape, num_actions, initial_learning_rate):
        #하이퍼파라미터를 초기화 시킨다.
        # input : 세로, 가로, 채널
        self.input_shape = input_shape
        #action 갯수
        self.num_actions = num_actions
        #neural network의 learnig rate
        self.learning_rate = initial_learning_rate
        #CNN의 하이퍼파라미터를 정의

        #filter size
        self.filter_size= 5
        #filter의 갯수
        self.num_filters = [16,32,64]
        #stride 갯수
        self.stride = 2
        #pool size
        self.poolsize = 2
        #convolutional layer의 shape
        self.convolution_shape = get_input_shape(input_shape[0], self.filter_size, self.stride) * get_input_shape(input_shape[1],
        self.filter_size, self.stride) * self.num_filters[2]

        #RNN 의 하이퍼파라미터와 마지막 feed forward layer를 정의한다.

        #뉴런의 갯수
        self.cell_size = 100
        #Hidden layers의 수
        self.hidden_layer = 50
        #drop out 확률
        self.dropout_probability = [0.3, 0.2]

        #optimization을 위한 하이퍼파라미터

        self.loss_decay_rate = 0.96
        self.loss_decay_steps = 180

        #CNN 의 모든 변수 초기화

        #input의 placeholder을 초기화 합니다. ( 세로,가로, 채널)




        self.input = tf.placeholder(tf.float32, [self.input_shape[0], self.input_shape[1],self.input_shape[2]], name="input")
        #action의 수와 같은 모양인 target value를 초기화한다 ( action value의 갯수 )
        self.target_vector = tf.placeholder(tf.float32,[self.num_actions , 1], name = 'target')

        #feature maps을 초기화한다.

        self.features1 = tf.Variable(initial_value = np.random.rand(self.filter_size, self.filter_size, input_shape[2], self.num_filters[0]), dtype=tf.float32)

        self.features2 = tf.Variable(initial_value = np.random.rand(self.filter_size, self.filter_size, self.num_filters[0], self.num_filters[1]), dtype= tf.float32)

        self.features3 = tf.Variable(initial_value = np.random.rand(self.filter_size, self.filter_size, self.num_filters[1], self.num_filters[2]), dtype = tf.float32)

        # RNN의 변수를 초기화한다.

        self.h= tf.Variable(initial_value = np.zeros((1, self.cell_size)), dtype = tf.float32)

        # 히든 -> 히든 의 weight matrix

        self.rW = tf.Variable(initial_value = np.random.uniform( low = -np.sqrt(6. /(self.convolution_shape + self.cell_size)), high = np.sqrt(6./(self.convolution_shape + self.cell_size)), size = (self.convolution_shape, self.cell_size)), dtype = tf.float32)

        # input -> hidden 의 weight matrix

        self.rU = tf.Variable(initial_value = np.random.uniform( low = -np.sqrt(6. /(2* self.cell_size)), high = np.sqrt(6./ (2* self.cell_size)), size= (self.cell_size, self.cell_size)), dtype= tf.float32)

        # hidden -> output weight matrix

        self.rV =tf.Variable(initial_value = np.random.uniform( low = -np.sqrt(6. / (2* self.cell_size)), high = np.sqrt(6./ (2* self.cell_size)), size = (self.cell_size, self.cell_size)), dtype=tf.float32)

        #bias

        self.rb = tf.Variable(initial_value = np.zeros(self.cell_size), dtype = tf.float32)
        self.rc = tf.Variable(initial_value = np.zeros(self.cell_size), dtype = tf.float32)

        #feed forward network의 웨이트와 바이어스를 초기화

        #Weight!

        self.fW = tf.Variable(initial_value = np.random.uniform( low = -np.sqrt(6./(self.cell_size + self.num_actions)), high = np.sqrt(6. / (self.cell_size + self. num_actions)), size = (self.cell_size, self.num_actions)), dtype=tf.float32)


        #Bias
        self.fb = tf.Variable(initial_value = np.zeros(self.num_actions), dtype = tf.float32)
        # learning rate

        self.step_count = tf.Variable(initial_value = 0, dtype = tf.float32)

        self.learning_rate = tf.train.exponential_decay(self.learning_rate, self.step_count, self.loss_decay_steps, self.loss_decay_steps, staircase = False)


        # 네트워크를 만들어보자

        ## 1. convolutional Network!

        self.conv1 = tf.nn.conv2d(input = tf.reshape(self.input, shape = (1, self.input_shape[0], self.input_shape[1], self.input_shape[2])), filter= self.features1, strides = [1,self.stride, self.stride,1], padding = "VALID")

        self.relu1=tf.nn.relu(self.conv1)

        self.pool1 = tf.nn.max_pool(self.relu1, ksize = [1,self.poolsize, self.poolsize, 1], strides = [1, self.stride, self.stride, 1], padding = "SAME")

        #second convolutional layer

        self.conv2 = tf.nn.conv2d(input = self.pool1, filter = self.features2, strides = [ 1, self.stride,self.stride, 1], padding = "SAME")
        self.relu2=tf.nn.relu(self.conv2)

        self.pool2 = tf.nn.max_pool(self.relu2, ksize = [1,self.poolsize, self.poolsize, 1], strides = [1, self.stride, self.stride, 1], padding = "SAME")
        #third convolutional layer

        self.conv3 = tf.nn.conv2d(input = self.pool2, filter = self.features3, strides = [1, self.stride, self.stride, 1], padding = "VALID")

        self.relu3= tf.nn.relu(self.conv3)

        self.pool3 = tf.nn.max_pool(self.relu3, ksize = [1, self.poolsize, self.poolsize, 1], strides = [1, self.stride, self.stride, 1], padding = "SAME")


        #드랍아웃을 추가하고 input을 reshape 한다.

        self.drop1 = tf.nn.dropout(self.pool3, self.dropout_probability[0])
        self.reshaped_input = tf.reshape(self.drop1, shape=[1,-1])


        #RNN 네트워크 구성

        self.h = tf.tanh(tf.matmul(self.reshaped_input, self.rW) + tf.matmul(self.h, self.rU)+self.rb)

        self.o = tf.nn.softmax(tf.matmul(self.h, self.rV) + self.rc)

        #RNN에 드랍아웃 추가
        self.drop2 = tf.nn.dropout(self.o, self.dropout_probability[1])

        #feed forward layer에 RNN의 결과를 feed  한다.
        self.output = tf.reshape(tf.matmul(self.drop2, self.fW) + self.fb, shape = [-1, 1])

        self.prediction = tf.argmax(self.output)

        #loss를 계산한다.

        self.loss = tf.reduce_mean(tf.square(self.target_vector - self.output))

        #Adam Optimizer을 사용하여 error을 minimize한다.

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

        #loss의 gradient를 계산한뒤 gradient를 업데이트한다.
        self.gradients = self.optimizer.compute_gradients(self.loss)
        self.update = self.optimizer.apply_gradients(self.gradients)

        self.parameters = (self.features1, self.features2 ,self.features3, self.rW, self.rU, self.rV, self.rb, self.rc, self.fW, self.fb)



#replaymemory

class ExperienceReplay():
    def __init__(self, buffer_size):
        # buffer를 만든다.
        self.buffer = []

        #buffer 의 사이즈

        self.buffer_size = buffer_size

        #버퍼의 사이즈가 Limit까지 쌓이면 새로운 transition은 넣고 오래된 transition을 제거한다.

    def appendToBuffer(self,memory_tuplet):
        if len(self.buffer) > self.buffer_size :
            for i in range(len(self.buffer) - self.buffer_size):
                self.buffer.remove(self.buffer[0])
        self.buffer.aapend(memory_tuplet)

    # n개의 transition을 sampling 하는 함수 정의
    def sample(self, n ):
        memories = []

        for i in range(n):
            memory_index = np.random.randint(0, len(self.buffer))
            memories.append(self.bufer[memory_index])
        return memories


    #train 함수 정의
def train(num_episodes, episode_length, learning_rate , scenario = "deatmatch.cfg", map_path = 'map02', render= True):
        #discount factor
        discount_factor = 0.99
        # 버퍼에 익스피리언스를 업데이트 하는 주기
        learning_rate = 0.01
        update_frequency = 5
        store_frequency  = 50

        #아웃풋을 프린팅하는 주기
        print_frequency = 1000

        #total reward와 total loss를 저장할 변수를 초기화

        total_reward = 0
        total_loss = 0
        old_q_value = 0

        # episodic reward와 loss를 저장할 리스트를 초기화
        rewards = []
        losses = []


        env = gym_super_mario_bros.make('SuperMarioBros-v0')
        env = BinarySpaceToDiscreteSpaceEnv(env, COMPLEX_MOVEMENT)

        env.reset()
        actionDRQN = DRQN((240, 256, 3), 11,learning_rate)
        targetDRQN = DRQN((240, 256, 3), 11,learning_rate)


        #experience buffer cell_size
        experiences = ExperienceReplay(1000000)
        # 모델 저장

        saver = tf.train.Saver({v.name: v for v in actionDRQN.parameters}, max_to_keep = 1)


        #학습을 시작해보자
        #샘플링을 위해 모든 변수를 초기화 시킨다. 그리고 버퍼에서 트렌지션을 storing한다.
        sample = 10
        store = 100

        with tf.Session() as sess:

            #모든 텐서플로우 변수를 초기화 한다.
            sess.run(tf.global_variables_initializer())
            for episode in range(num_episodes):
                #새로운 에피소드를 시작한다.
                env.reset()

                for frame in range(episode_length):
                    env.render()
                    state = env.observation_space.shape
                    print(state)
                    action = actionDRQN.prediction.eval(feed_dict = {actionDRQN.input: state})
                    #env.step (action을 통하)
                    next_state, reward, done, info = env.step(action)
                    #reward를 업데이트
                    total_reward += reward

                    state= next_state
                    #game이 끝나면 break한다.
                    if done:
                        break
                    #transition을 버퍼에 넣는다.
                    if (frame%store)==0:
                        experience.appendToBuffer((s,action,reward))

                    #buffer에서 샘플을 뽑는다.
                    if (frame%sample) == 0:
                        memory = experiences.sample(1)
                        mem_frame=memory[0][0]
                        mem_reward = memory[0][2]

                        #train

                        Q1 = actionDRQN.output.eval(feed_dict = {actionDRQN.input : state})
                        Q2 = targetDRQN.output.eval(feed_dict = {targetDRQN.input : mem_frame})

                        #learning rate

                        learning_rate = actionDRQN.learning_rate.eval()

                        #Q value를 계산한다.
                        Qtarget = old_q_value + learning+_rate * (mem_reward + discount_factor * Q2 - old_q_value)

                        #update

                        old_q_value = Qtarget

                        # loss 계산
                        loss =actionDRQN.loss.eval(feed_dict = {actionDRQN.target_vector : Qtarget, actionDRQN.input : mem_frame})

                        p
                        #update loss
                        total_loss += loss

                        # 두 네트워크를 업데이트한다.

                        actionDRQN.update.run(feed_dict = {actionDRQN.target+vector : Qtarget, actionDRQN.input : mem_frame})
                        targetDRQN.update.run(feed_dict = {targetDRQN.target+vector : Qtarget, targetDRQN.input : mem_frame})
                        rewards.append((episode, total_reward))
                        losses.append((episode, total_loss))

                        total_reward = 0
                        total_loss = 0

train=train(num_episodes=10000, episode_length = 300, learning_rate = 0.01, render = True)
