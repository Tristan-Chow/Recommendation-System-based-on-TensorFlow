import tensorflow as tf
from DNN import DNN
from datahelper import genreNumpy
import pickle
from DataInput import *

with open('E:/毕业论文/dataset.pkl', 'rb') as f:
    training_data = pickle.load(f)


def train(learning_rate=0.001):
    with tf.Graph().as_default():
        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True
        )
        sess = tf.Session(config=session_config)
        with sess.as_default():
            dnn = DNN(
                32, genreNumpy
            )
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(dnn.loss)
        sess.run(tf.global_variables_initializer())

    for i in range(50):
        for batch_num, train_data in DataInput(training_data, 64):
            feel_dict = {
                dnn.g: train_data[0],
                dnn.o: train_data[1],
                dnn.a: train_data[2],
                dnn.geo: train_data[3],
                dnn.wh: train_data[4],
                dnn.time: train_data[6]
            }

            step, _, loss = sess.run([global_step, train_op, dnn.loss], feel_dict)
