import tensorflow as tf
import numpy as np
import tf_geometric as tfg

class VAE_no_crowds(tf.keras.Model):
    def __init__(self, task_num, feature_size, worker_num, class_num, answer_num, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_num = task_num
        self.feature_size = feature_size
        self.worker_num = worker_num
        self.class_num = class_num
        self.answer_num = answer_num
        self.hidden_size = 128

        self.Efc = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        self.Efc_mean = tf.keras.layers.Dense(self.class_num, activation=None)
        self.Efc_log_std = tf.keras.layers.Dense(self.class_num, activation=None)
        self.left_bn_1 = tf.keras.layers.BatchNormalization(center=False, scale=False)
        self.left_bn_2 = tf.keras.layers.BatchNormalization(center=False, scale=False)
        self.Dropout = tf.keras.layers.Dropout(0.5)

        self.Dfc_1 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        self.Dfc_2 = tf.keras.layers.Dense(self.feature_size, activation=None)

    def encoder(self, x, training=False):
        # task_feature = self.left_bn_1(task_feature)
        h = self.Efc(x)
        # h = self.left_bn_2(h)
        mean = self.Efc_mean(h)
        log_std = self.Efc_log_std(h)
        # cls_out = self.Efc_mean(h)
        return mean, log_std
        # return tf.nn.softmax(cls_out)

    def sample_z(self, mean, log_std):
        std = tf.math.exp(log_std)
        eps = tf.random.normal(shape=(self.task_num, self.class_num), mean=0, stddev=0.01)
        return mean + eps * std

    def decoder(self, z, training=False):
        h = self.Dfc_1(z)
        _x = self.Dfc_2(h)
        return _x

    def call(self, x, training=False):
        mean, log_std = self.encoder(x, training=training)
        z = self.sample_z(mean, log_std)
        _x = self.decoder(z, training=training)
        cls_out = self.encoder(x, training=training)
        # return _x, mean, log_std
        return cls_out