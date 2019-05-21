# Author :  MMY
# Time    : 2019/5/7 20:13
import numpy as np
import tensorflow as tf

v = "I'm global variable"


class AA:
    v = "I'm  variable in class"
    print(v)

    def __init__(self):
        self.v = "I'm attribute of object"

    def ja1(self, x=2):
        self.aa=10+x

    def ja2(self):
        self.ja3()

    def ja3(self, y='Wow'):
        v = "I'm local variable " + y
        print(self.v)
        print(v)
        print(self.aa)

"""
1. class中定义的变量是局部变量，其生存区域在类内部，类内部def函数的外部
2. class内，def函数中定义的变量是局部变量，其生存区域在def函数内部
3. class内，对象属性值即self.xxx可以在其他对象方法内调用，对象属性值可在_init_构造函数中定义或对象方法内创建

4. class内，对象方法可以相互调用, 但需注意：
                当某个对象属性值self.xxx在对象方法def func_A中创建，
                其他对象方法def func_B内部若要引用self.xxx的话，必须
                先运行对象方法func_A，才能运行对象方法func_B
                
        注：（1）类是可运行的代码块，属性及方法仅在运行时被创建
               （2）方法定义位置无关紧要，调用顺序需要注意
                
"""

# ob = AA()
# ob.ja1()
# ob.ja2()
# ob.ja3('God')

# x=np.arange(1,13)
# y=x.reshape([3, -1])
# print(x)
# print(y)
# z=y[:, 1][np.newaxis, :]
# print(z)

# prob_weights = [[0.2], [0.3], [0.5]]
# p = np.ravel(prob_weights)
# print(p)
# D = np.shape(p)
# print(D)
#
# action = np.random.choice(range(D[0]), p=p)
# print(action)




# sess = tf.InteractiveSession()
# with tf.Session() as sess:
#     out =[[2.3, 3.6, 4.3, 1.5],[1.4, 2.1, 2.5, 3.2]]
#     prob = tf.nn.softmax(out)
#     print(prob.eval())
#     # action = np.random.choice(4, p=prob.eval())
#     action = [1,2]
#     print('original action=', action)
#     coded_act = tf.one_hot(action, 4)  # 将类别标签进行one_hot编码，以计算交叉熵
#     print('action=', coded_act.eval())
#     y1 = tf.reduce_sum(-tf.log(prob)*coded_act)
#     print('y1=', y1.eval())
#
#     y2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out, labels=action)
#     print('y2=', y2.eval())
#
#     y3 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=out, labels=coded_act)
#     print('y3=', y2.eval())
"""
out表示unscaled的网络输出（未经过其他处理的）
action表示直接以类别作标签，
        eg: action=0表示第1类， action=1表示第2类
coded_act表示用one_hot编码后的类别标签，
        eg: action=0时，标签值为[1,0,0]表示第1类
             action=1时，标签值为[0,1,0]表示第2类
1. tf.nn.softmax_cross_entropy_with_logits_v2(logits=out, labels=coded_act)
等价于
prob = tf.nn.softmax(out)
tf.reduce_sum(-tf.log(prob)*coded_act)
等价于
tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out, labels=action)
"""
N=2

sess=tf.InteractiveSession()
a =np.arange(12).reshape([3,-1])
print(a)
b = np.arange(6).reshape([2,-1])
print(b)
tG =np.array([a, b])
Buffer=np.zeros_like(tG)
for i in range(N):
    for ind,ele in enumerate(tG):
        Buffer[ind] += ele
    print('Buffer=',Buffer,'shape=',np.shape(Buffer))

grad=tf.divide(Buffer,N)
print(grad)