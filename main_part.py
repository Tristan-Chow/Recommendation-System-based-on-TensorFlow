import tensorflow as tf
from head import *
from test import *


def get_batches(Xs, ys, batch_size):
    for start in range(0, len(Xs), batch_size):
        end = min(start + batch_size, len(Xs))
        yield Xs[start:end], ys[start:end]

def get_inputs():
     feature = tf.placeholder(tf.float32, [None, 40], name='feature')
     label=tf.placeholder(tf.float32,[None,1],name='label')
     return feature,label

def layers1(input_tensor1):
     weight1=tf.Variable(tf.random_normal([40,32],stddev=1.0,seed=1))
     biase1=tf.Variable(tf.zeros([32]))
     pred=tf.nn.sigmoid(tf.matmul(input_tensor1,weight1)+biase1)
     return pred,weight1,biase1

def layer2(input_tensor2):
    weight2 = tf.Variable(tf.random_normal([32,16], stddev=1.0, seed=1))
    biase2 = tf.Variable(tf.zeros([16]))
    pred2 = tf.nn.tanh(tf.matmul(input_tensor2, weight2) + biase2)
    return pred2,weight2,biase2

def layer3(input_tensor3):
    weight3 = tf.Variable(tf.random_normal([16, 2], stddev=1.0, seed=1))
    biase3 = tf.Variable(tf.zeros([2]))
    pred3 = tf.nn.tanh(tf.matmul(input_tensor3, weight3) + biase3)
    return pred3,weight3,biase3

def layer4(input_tensor4):
    weight4 = tf.Variable(tf.random_normal([2, 1], stddev=1.0, seed=1))
    biase4= tf.Variable(tf.zeros([1]))
    pred4= tf.nn.sigmoid(tf.matmul(input_tensor4, weight4) + biase4)
    return pred4,weight4,biase4

tf.reset_default_graph()
train_graph = tf.Graph()
with train_graph.as_default():
    feature, label= get_inputs()
    pred,weight1,biase1=layers1(feature)
    pred2,weight2,biase2=layer2(pred)
    pred3,weight3,biase3=layer3(pred2)
    pred4,weight4,biase4=layer4(pred3)
    cross_entropy = -tf.reduce_mean(label * tf.log(pred4))
    train_step=tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
    corect_prediction=tf.equal(tf.round(pred4), label)
    accurancy=tf.reduce_mean(tf.cast(corect_prediction,tf.float32))
    tf.summary.scalar("loss",cross_entropy)
    tf.summary.scalar("accu",accurancy)
    merge_summary=tf.summary.merge_all()
with tf.Session(graph=train_graph) as sess:
    writer=tf.summary.FileWriter("D:/tensorboard/",graph=train_graph)
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(7500):
        start = i * 200
        end = start+200
        x=features[start:end]
        y=targets[start:end]
        test_x=test_features
        test_y=test_targets
        feed={
            feature:x,
            label:y
        }
        test_feed={
            feature:test_x,
            label:test_y
        }
        sess.run(train_step,feed_dict=feed)
        summary=sess.run(merge_summary,feed_dict=feed)
        writer.add_summary(summary,i)
        if i%500 ==0:
            total_cross_entrophy=sess.run(cross_entropy,feed_dict={feature:features,label:targets})
            result=sess.run(pred4, feed_dict=test_feed)
            validated_acc = sess.run(accurancy, feed_dict=test_feed)
            print(validated_acc)
            print(i,total_cross_entrophy,result)








