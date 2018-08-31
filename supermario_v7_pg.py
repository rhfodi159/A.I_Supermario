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
class PolicyGradient:
#모든 variable을 초기화 시킨다.

    def __init__(self, n_x, n_y, learning_rate = 0.01, reward_decay=0.95):
        # 스테이트의 수
        self.n_x = n_x
        # action의 수
        self.n_y = n_y

        #network의 learning rate

        self.lr = learning_rate

        #discount factor

        self.gamma =reward_decay

        # 다음 스테이트 (옵저베이션) , 행동, 리워드를 저장하기 위해 리스트를 초기화시킨다.

        self.episode_observations,self.episode_actions, self.episode_rewards = [],[],[]

        # neural network를 build하기 위해 build_network 이름으로 함수를 만든다.

        self.build_network()

        #로스를 저장한다.
        self.cost_history = []

        #tensorflow를 초기화한다.

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())


    #트렌지션 정보를 저장할 공간을 만드는 함수를 만든다.


    def store_transition(self, s,a,r):

        self.episode_observations.append(s)
        self.episode_rewards.append(r)


        #엑션을 저장
        action = np.zeros(self.n_y)

        action[a] = 1

        self.episode_actions.append(action)


    # 스테이트에서 엑션을 선택하는 함수를 정의

    def choose_action(self, observation):
        #reshape observation to (num_feature, 1)

    #    obersvation = env.observation_space.shape

    #    observation = observation[:,np.newaxis]

        #forward propagation으로 소프트맥스 값을 얻는다.

        pro_weights = self.sess.run(self.outputs_softmax, feed_dict = {self.X: np.array(observation).reshape(1,240,256,3)})

        action = np.random.choice(range(len(pro_weights.ravel())), p=pro_weights.ravel())
        return action

    #뉴럴네트워크 함수를 만든다.

    def build_network(self):

        #x와 y의 플레이스홀더

        self.X=tf.placeholder(tf.float32, shape =[None,240,256,3], name ="x")
        self.Y =tf.placeholder(tf.float32, shape= [None,1], name = "Y")

        #reward placesholder
        self.discounted_episode_rewards_norm =tf.placeholder(tf.float32, [None,], name="action_value")

        # 2개의 히든레이어 1개의 아웃풋 레이어를 만든다.

        units_layer_1 = 10
        units_layer_2 = 10

        units_output_layer = self.n_y

        with tf.name_scope("layer1"):

            layer1 = tf.layers.conv2d(self.X, filters= int(64), kernel_size=[3,3],padding='VALID',use_bias=False, name="layer1")

            layer1_1 = tf.nn.relu(layer1)

        with tf.name_scope("layer2"):
            layer2 = tf.layers.conv2d(layer1_1, filters = int(128), kernel_size=[3,3], padding='VALID', use_bias = False, name='layer2')
            layer2_1 = tf.nn.relu(layer2)


        with tf.name_scope("outlayer"):
            Z1= tf.contrib.layers.flatten(layer2_1)
            Z2=tf.layers.dense(Z1, units=10, use_bias=False)
            A2 = tf.nn.relu(Z2)
            Z3=tf.layers.dense(A2, units=10, use_bias=False)
            A3 = tf.nn.softmax(Z3)



        #softmax를 아웃풋 레이어에 적용

        logits =A3

        labels = tf.transpose(self.Y)

        self.outputs_softmax= tf.nn.softmax(logits, name="A3")

        #로스함수를 크로스엔트로피로 정해준다.

        neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels)
        print(neg_log_prob)

        # loss
        loss = tf.reduce_mean(neg_log_prob * self.discounted_episode_rewards_norm)

        # 아담을 사용하여 로스를 최소화 한다.

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)



    #discount and norm 되어진 리워드 함수를 정의

    def discount_and_norm_rewards(self):
        discounted_episode_rewards = np.zeros_like(self.episode_rewards)

        cumulative = 0

        for t in reversed(range(len(self.episode_rewards))):
            cumulative = cumulative * self.gamma + self.episode_rewards[t]
            discounted_episode_rewards[t]= cumulative

        discounted_episode_rewards=discounted_episode_rewards.astype('float32')
        discounted_episode_rewards -= np.mean(discounted_episode_rewards)
        discounted_episode_rewards /= np.std(discounted_episode_rewards)


        discounted_episode_rewards = np.array(discounted_episode_rewards).reshape(None,240,256,3)
        return discounted_episode_rewards

    # learning network

    def learn(self):

        #discount and normalize episodic reward

        discounted_episode_rewards_norm = self.discount_and_norm_rewards()

        #train the network를
        self.sess.run(self.train_op, feed_dict= {
            self.X: np.vstack(self.episode_observations).T,
            self.Y: np.vstack(np.array(self.episode_actions)).T,
            self.discounted_episode_rewards_norm: discounted_episode_rewards_norm,

        })
        #reset the network
        self.episode_observations, self.episode_actions, self.episode_rewards = [],[],[]
        return discounted_episode_rewards_norm


env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, COMPLEX_MOVEMENT)
done = True


RENDER_ENV=False

EPISODES = 5000
rewards = []
PG = PolicyGradient(n_x = env.observation_space.shape[0], n_y= env.action_space.n, learning_rate = 0.01,reward_decay=0.99)


for episodes in range(EPISODES):

    observation = env.reset()

    observation=np.array(observation).reshape(1,240,256,3)

    episode_reward = 0
    print("episode",episodes)

    while True:

        if RENDER_ENV: env.render()


        action = PG.choose_action(observation)


        next_state, reward, done, info = env.step(action)


        PG.store_transition(next_state, action, reward)

        episode_rewards_sum = sum(PG.episode_rewards)


        if done:
            episode_rewards_sum = sum(PG.episode_rewards)
            rewards.append(episode_rewards_sum)
            print(episode_rewards_sum)
            max_reward_so_far = np.amax(rewards)




            print("Reward sum ", episode_rewards_sum)
            print("Max reward", max_reward_so_far)

            #train the network
            discounted_episode_rewards_norm = PG.learn()

            break
        observation = next_state
