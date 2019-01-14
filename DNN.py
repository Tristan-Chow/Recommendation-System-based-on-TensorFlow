import tensorflow as tf
import pickle
from datahelper import *
import math

with open('.../remap.pkl', 'rb') as f:
    genreNumpy = pickle.load(f)


class DNN:
    def __init__(self, embedding_size, genre_size=genre_count, occ_size=occ_count, geo_size=geo_count,
                 l2_reg_lamda=0.0, genre_matrix=genreNumpy):
        self.g = tf.placeholder(tf.int64, [None, ])
        self.a = tf.placeholder(tf.float64, [None, ])
        self.o = tf.placeholder(tf.int64, [None, ])
        self.geo = tf.placeholder(tf.int64, [None, ])
        self.wh = tf.placeholder(tf.int64, [None, ])
        self.time = tf.placeholder(tf.int64, [None, ])
        self.y = tf.placeholder(tf.int64, [None])

        self.genre_Query = genre_matrix

        self.genre_emb_w = tf.Variable(tf.random_uniform([genre_size, embedding_size], -1.0, 1.0))
        self.occupation_emb_w = tf.Variable(tf.random_uniform([occ_size, embedding_size], -1.0, 1.0))
        self.geographic_emb_w = tf.Variable(tf.random_uniform([geo_size, embedding_size], -1.0, 1.0))
        # embedding process
        with tf.name_scope("embedding"):
            watch_genre = tf.gather(self.genre_Query, self.wh)
            watch_history_genre_emb = tf.nn.embedding_lookup(self.genre_emb_w, watch_genre)
            watch_history_final = tf.reduce_mean(tf.reduce_mean(watch_history_genre_emb, axis=0), axis=0)

            geo_emb = tf.nn.embedding_lookup(self.geographic_emb_w, self.geo)
            occ_emb = tf.nn.embedding_lookup(self.occupation_emb_w, self.o)

        user_vector = tf.concat([geo_emb, occ_emb, watch_history_final, self.g, self.a, self.time], axis=0)

        # Deep Neural Network
        with tf.name_scope("DNN"):
            d_layer_1 = tf.layers.Dense(user_vector, units=1024, activation=tf.nn.relu, use_bias=True, name='f1',
                                        trainable=True)
            d_layer_2 = tf.layers.Dense(d_layer_1, units=512, activation=tf.nn.relu, use_bias=True, name='f2',
                                        trainable=True)
            d_layer_3 = tf.layers.Dense(d_layer_2, units=256, activation=tf.nn.relu, use_bias=True, name='f3',
                                        trainable=True)
        movie_embedding = tf.Variable(
            tf.truncated_normal([movie_count, embedding_size], steddev=1.0 / math.sqrt(embedding_size)), trainable=True)
        biase = tf.Variable(tf.zeros([movie_count]))

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(
                tf.nn.sampled_softmax_loss(inputs=d_layer_3, weights=movie_embedding, biases=biase,
                                           num_classes=movie_count,
                                           num_true=1))

