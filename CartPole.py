# Author :  Ma Mengyuan
# Time    : 2019/5/5 15:17
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gym

env = gym.make('CartPole-v0')
env.seed(1)
env = env.unwrapped
# env.reset()
# random_episodes = 0
# reward_sum = 0
# iter_max = 5
#
# rwd_record = []
# while random_episodes < iter_max:
#     env.render()
#     obs, rwd, done, _ = env.step(np.random.randint(0, 2))
#     reward_sum += rwd
#
#     if done:
#         random_episodes += 1
#         rwd_record.append(reward_sum)
#         print('Reward for this episode was:', reward_sum)
#         reward_sum = 0
#         env.reset()
# plt.plot(rwd_record)
# plt.xlabel('episode')
# plt.ylabel('Accumulated reward')
# plt.title('Training process')
# plt.show()
path = 'Model/'  # 文件夹名称：模型保存在该文件夹下
name = 'my_model'  # 模型名称
file_path = path + name  # 完整保存路径

H = 10
batch_size = 1
learning_rate = 12
D = 4
n_action = 2
gamma = 0.95


observations = tf.placeholder(tf.float32, [None, D], name='input_state')
action_label = tf.placeholder(tf.int32, [None, ], name='input_action_label')
Advantages = tf.placeholder(tf.float32, [None, ], name='Advantages')

W1 = tf.get_variable('W1', shape=[D, H], initializer=tf.contrib.layers.xavier_initializer())
b1 =tf.get_variable('b1', initializer=tf.constant(0.1))
layer1 = tf.nn.relu(tf.matmul(observations, W1)+b1)
W2 = tf.get_variable('W2', shape=[H, n_action], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.get_variable('b2',initializer=tf.constant(0.1))
score = tf.matmul(layer1, W2)+b2
prob = tf.nn.softmax(score,name='prob')

loglik_out = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=score, labels=action_label)
loss = tf.reduce_mean(loglik_out*Advantages)

train_op = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss)
saver = tf.train.Saver()

obs, acts, act_rs = [], [], []
reward_sum = 0
episode_number = 0
total_epsiodes = 1000
Average_reward =[]
num_ep = 10


def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(r.size)):
        running_add = running_add * gamma +r[t]
        discounted_r[t] = running_add
    discounted_r -= np.mean(discounted_r)
    discounted_r /= np.std(discounted_r)
    return discounted_r


with tf.Session() as sess:
    rendering = False
    init = tf.global_variables_initializer()
    sess.run(init)
    observation = env.reset()

    ckpt = tf.train.get_checkpoint_state('./'+path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    while episode_number <= total_epsiodes:
        if reward_sum  >300 or rendering ==True:
            env.render()
            rendering = True

        x = np.reshape(observation, [1, D])
        tfprob = sess.run(prob,feed_dict={observations: x})
        # print(tfprob)

        action = np.random.choice(n_action, p=tfprob.ravel())
        # print(action)
        observation_, reward, done, _ = env.step(action)
        # print(done)

        obs.append(x)
        acts.append(action)
        act_rs.append(reward)
        reward_sum += reward

        if done:
            episode_number += 1
            epx = np.vstack(obs)
            epy = np.array(acts)
            # print(epy.shape)
            epr = np.array(act_rs)
            # print(epr.shape)
            Average_reward.append(reward_sum)
            if episode_number % num_ep == 0:
                print('Sum reward for episode %d : %f. ' % (episode_number, reward_sum))
                Average_reward=[]

            discounted_epr = discount_rewards(epr)
            # print(discounted_epr.shape)
            cross_entroy, out, loss_value = sess.run([loglik_out, score, loss],
                feed_dict={observations: epx, action_label: epy, Advantages: discounted_epr})
            # print(cross_entroy.shape, out.shape, loss_value)

            sess.run(train_op, feed_dict={observations: epx, action_label: epy, Advantages: discounted_epr})

            if reward_sum > 500:
                print('Task solved in', episode_number, 'episodes!')
                break
            reward_sum = 0
            obs, acts, act_rs = [], [], []
            observation = env.reset()  # 每次尝试结束重置环境
        observation = observation_
    saver.save(sess, './' + file_path)  # 保存在当前文件夹下
# 测试部分
    if __name__ == '__main__':
        print('Test')
        episode_max = 10
        counter=0
        reward_sum = 0
        observation = env.reset()
        while counter<episode_max:
            env.render()
            x = np.reshape(observation, [1, D])
            tfprob = sess.run(prob,feed_dict={observations: x})
            # print(tfprob)
            action = tf.argmax(tfprob.ravel()).eval()  # greedy choosing
            # print(action)
            observation_, reward, done, _ = env.step(action)
            reward_sum += reward

            if done:
                counter += 1
                observation = env.reset()
                print('episode:', counter,', reward:',reward_sum)
                reward_sum = 0
            observation = observation_

