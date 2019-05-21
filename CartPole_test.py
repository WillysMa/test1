# Author :  MMY
# Time    : 2019/5/9 22:26
import tensorflow as tf
import numpy as np
import gym
env = gym.make('CartPole-v0')
env.seed(1)
env = env.unwrapped

path = 'Model/'  # 文件夹名称：模型保存在该文件夹下
name = 'my_model'  # 模型名称
file_path = path + name  # 完整保存路径
D=4

sess=tf.InteractiveSession()
saver = tf.train.import_meta_graph(file_path+'.meta')  # import graph model
graph = tf.get_default_graph()
observations = graph.get_tensor_by_name('input_state:0')
prob = graph.get_tensor_by_name('prob:0')

saver.restore(sess, tf.train.latest_checkpoint('./'+path))  # restore the value of network parameters by checkpoint

print('Test')
episode_max = 10
counter = 0
reward_sum = 0
observation = env.reset()
while counter < episode_max:
    env.render()
    x = np.reshape(observation, [1, D])
    tfprob = sess.run(prob, feed_dict={observations: x})
    # print(tfprob)
    action = tf.argmax(tfprob.ravel()).eval()
    # print(action)
    observation_, reward, done, _ = env.step(action)
    reward_sum += reward

    if done:
        counter += 1
        observation = env.reset()
        print('episode:', counter, ', reward:', reward_sum)
        reward_sum = 0
    observation = observation_
