from data_loader import load_SP_dataset, load_Music_dataset, load_LabelMe_dataset, load_BCD_dataset, load_Reuters_dataset, \
                        load_Bill_dataset, load_Head_dataset, load_Shape_dataset, load_Forehead_dataset, load_Throat_dataset, load_Underpart_dataset,\
                        load_Breast_dataset, shuffle_data
import tensorflow as tf
import numpy as np
import os
import random
from sklearn.metrics import accuracy_score, f1_score
# from sklearn.mixture import GaussianMixture as GMM
# import matplotlib.pyplot as plt
import tf_geometric as tfg
from model import Dual_Tower

def seed_tensorflow(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '0' # `pip install tensorflow-determinism` first,使用与tf>2.1

# seed_tensorflow(42)

class BNN():

    def __init__(self, feature_size, hidden_size, prior_sigma_1=1.5, prior_sigma_2=0.1, prior_pi=0.5):

        self.feature_size = feature_size
        self.hidden_size = hidden_size

        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi_1 = prior_pi
        self.prior_pi_2 = 1.0 - prior_pi
        self.kernel_mu_1, self.bias_mu_1, self.kernel_rho_1, self.bias_rho_1 = self.generate_mu_rho([1, 20])
        self.kernel_mu_2, self.bias_mu_2, self.kernel_rho_2, self.bias_rho_2 = self.generate_mu_rho([20, 20])
        self.kernel_mu_3, self.bias_mu_3, self.kernel_rho_3, self.bias_rho_3 = self.generate_mu_rho([20, 1])
        self.trainables = [self.kernel_mu_1, self.bias_mu_1, self.kernel_rho_1, self.bias_rho_1, \
                           self.kernel_mu_2, self.bias_mu_2, self.kernel_rho_2, self.bias_rho_2, \
                           self.kernel_mu_3, self.bias_mu_3, self.kernel_rho_3, self.bias_rho_3]
        self.optimizer = tf.keras.optimizers.Adam(0.08)

    def generate_mu_rho(self, shape):
        kernel_mu = tf.Variable(tf.random.truncated_normal(shape, mean=0., stddev=1.))
        bias_mu = tf.Variable(tf.random.truncated_normal(shape[1:], mean=0., stddev=1.))
        kernel_rho = tf.Variable(tf.zeros(shape))
        bias_rho = tf.Variable(tf.zeros(shape[1:]))
        return kernel_mu, bias_mu, kernel_rho, bias_rho

    def __call__(self, inputs):
        self.loss = []
        features_1 = self.dense(inputs, self.kernel_mu_1, self.bias_mu_1, self.kernel_rho_1, self.bias_rho_1)
        features_2 = self.dense(tf.nn.relu(features_1), self.kernel_mu_2, self.bias_mu_2, self.kernel_rho_2, self.bias_rho_2)
        features_3 = self.dense(tf.nn.relu(features_2), self.kernel_mu_3, self.bias_mu_3, self.kernel_rho_3, self.bias_rho_3)
        return features_3

    def dense(self, features, kernel_mu, bias_mu, kernel_rho, bias_rho):
        kernel_sigma = tf.math.softplus(kernel_rho)
        kernel = kernel_mu + kernel_sigma * tf.random.normal(kernel_mu.shape)
        self.kl_Qwtheta_Pw(kernel, kernel_mu, kernel_sigma)
        bias_sigma = tf.math.softplus(bias_rho)
        bias = bias_mu + bias_sigma * tf.random.normal(bias_mu.shape)
        self.kl_Qwtheta_Pw(bias, bias_mu, bias_sigma)
        return tf.einsum('ij,jk->ik', features, kernel) + bias

    def kl_Qwtheta_Pw(self, w, mu, sigma):
        Qwtheta = tf.math.log(self.gaussian_distribution_density(w, mu, sigma)) #poster
        Pw = tf.math.log(self.prior_pi_1 * self.gaussian_distribution_density(w, 0.0, self.prior_sigma_1)
                         +
                         self.prior_pi_2 * self.gaussian_distribution_density(w, 0.0, self.prior_sigma_2)) # prior
        self.loss.append(tf.math.reduce_sum(Qwtheta - Pw))

    def gaussian_distribution_density(self, x, mu, sigma):
        return 1.0 / ((2 * np.pi) ** 0.5 * sigma) * tf.math.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + 1e-30

class LableMe_model(tf.keras.Model):
    # def __init__(self, task_num, feature_size, worker_num, class_num, answer_num=None, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.task_num = task_num
    #     self.feature_size = feature_size
    #     self.worker_num = worker_num
    #     self.class_num = class_num
    #     self.answer_num = answer_num
    #     self.hidden_size = 128
    #
    #     # self.worker_feature = tf.Variable(tf.random.normal((self.worker_num, self.hidden_size), mean=0, stddev=0.01))
    #     self.worker_mu = tf.Variable(tf.random.normal((self.worker_num, self.class_num), mean=0, stddev=0.01))
    #     self.worker_rho = tf.Variable(tf.random.normal((self.worker_num, self.class_num), mean=0, stddev=0.01))
    #     # self.worker_feature = tf.one_hot(indices=range(self.worker_num), depth=self.worker_num)
    #
    #     self.flatten = tf.keras.layers.Flatten()
    #     self.left_fc1 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
    #     self.left_fc2 = tf.keras.layers.Dense(self.class_num)
    #     self.Dropout = tf.keras.layers.Dropout(0.5)
    #     self.left_bn = tf.keras.layers.BatchNormalization(center=False, scale=False)
    #     self.left_bn1 = tf.keras.layers.BatchNormalization(center=False, scale=False)
    #
    #     self.right_fc1 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
    #     # self.right_fc2 = tf.keras.layers.Dense(self.class_num, activation='relu')
    #     self.right_fc3 = tf.keras.layers.Dense(self.class_num, activation=None)
    #     self.right_bn = tf.keras.layers.BatchNormalization(center=False, scale=False)
    #     self.right_bn1 = tf.keras.layers.BatchNormalization(center=False, scale=False)
    #     self.fc_task_mu = tf.keras.layers.Dense(self.class_num, activation=None)
    #     self.fc_task_log_sigma = tf.keras.layers.Dense(self.class_num, activation=None)
    #
    #     self.fc_worker_mu = tf.keras.layers.Dense(self.class_num, activation='relu')
    #     self.fc_worker_rho = tf.keras.layers.Dense(self.class_num, activation='relu')
    #
    #     self.p_pure = np.ones((self.class_num, ), dtype=np.float32)/self.class_num
    #     # self.weight = tf.Variable(tf.ones(self.answer_num,))
    #
    #     # self.gnn1 = tfg.layers.GAT(self.hidden_size, activation=None, num_heads=8, attention_units=8, edge_drop_rate=0.1)
    #     # self.gnn1 = tfg.layers.GCN(self.hidden_size, activation=tf.nn.relu)
    #     # self.gnn2 = tfg.layers.GCN(self.class_num)
    #
    # def sample(self, mean, log_std):
    #     std = tf.math.exp(log_std)
    #     eps = tf.random.normal(shape=(self.task_num, self.class_num), mean=0, stddev=0.01)
    #     return mean + eps * std
    #
    # def left_NN(self, task_feature, training=None):
    #     task_feature = self.flatten(task_feature)
    #
    #     # task_feature = self.left_bn(task_feature)
    #     task_feature = self.left_fc1(task_feature)
    #     task_feature = self.Dropout(task_feature, training)
    #     # task_feature = self.left_bn1(task_feature)
    #     cls_out = self.left_fc2(task_feature)
    #
    #     return cls_out
    #
    # def right_NN(self, task_feature, answers, training=None):
    #
    #     task_ids = answers[:, 0]
    #     worker_ids = answers[:, 1]
    #     label_ids = answers[:, 2]
    #     row = task_ids
    #     col = worker_ids + self.task_num
    #     Row = np.concatenate([row, col], axis=-1)
    #     Col = np.concatenate([col, row], axis=-1)
    #     edge_index = np.concatenate([[Row], [Col]], axis=0)
    #
    #     task_feature = self.flatten(task_feature)
    #     # task_feature = self.right_bn(task_feature)
    #     task_feature = self.right_fc1(task_feature)
    #     task_feature = self.Dropout(task_feature, training)
    #     # task_feature = self.right_bn1(task_feature)
    #     # task_feature = self.right_fc3(task_feature)
    #
    #     self.task_mu = self.fc_task_mu(task_feature)
    #     self.task_log_sigma = self.fc_task_log_sigma(task_feature)
    #     task_feature = self.sample(self.task_mu, self.task_log_sigma)
    #
    #
    #
    #     # worker_mu = self.fc_worker_mu(self.worker_feature)
    #     # worker_rho = self.fc_worker_rho(self.worker_feature)
    #
    #     # node_feature = self.right_fc3(tf.concat([task_feature, worker_sigma], axis=0))
    #     # gnn_out = self.gnn1([node_feature, np.array(edge_index, dtype=np.int32)], training=training)
    #     # gnn_out = self.gnn2([gnn_out, np.array(edge_index, dtype=np.int32)], training=training)
    #
    #     # a = tf.gather(node_feature, range(self.task_num))
    #     # b = tf.gather(node_feature, range(self.task_num, self.task_num+self.worker_num))
    #
    #     # worker_bias = tf.gather(node_feature, range(self.task_num))[:, :, None] * tf.gather(node_feature, range(self.worker_num, ))[:, None, :]
    #     # print(worker_bias)
    #
    #     masked_task_feature = tf.gather(task_feature, row)
    #     masked_worker_mu = tf.gather(self.worker_mu, col - self.task_num)
    #     masked_worker_rho = tf.gather(self.worker_rho, col - self.task_num)
    #
    #     # agg_task_feature = tf.math.unsorted_segment_mean(data=masked_task_feature, segment_ids=task_ids, num_segments=self.task_num) \
    #     #                    + \
    #     #                    tf.math.unsorted_segment_sum(data=masked_worker_feature, segment_ids=task_ids, num_segments=self.task_num)
    #     # agg_worker_feature = tf.math.unsorted_segment_mean(data=masked_worker_feature, segment_ids=worker_ids, num_segments=self.worker_num) \
    #     #                    + \
    #     #                    tf.math.unsorted_segment_sum(data=masked_task_feature, segment_ids=worker_ids, num_segments=self.worker_num)
    #
    #     # masked_task_feature = tf.gather(task_feature, row) # p(t)
    #
    #     # worker_bias = tf.sparse.from_dense(tf.matmul(task_feature, worker_feature, transpose_b=True))
    #
    #     crowd_bias = masked_task_feature * tf.math.log(1+tf.math.exp(masked_worker_rho)) + masked_worker_mu # tf.nn.sigmoid(masked_worker_rho) tf.math.log(1+tf.math.exp(masked_worker_rho))
    #     # agg_in = masked_task_feature * tf.nn.sigmoid(masked_worker_feature) + tf.nn.sigmoid(self.worker_bias)
    #
    #     # agg_out = tf.math.unsorted_segment_sum(data=crowd_bias, segment_ids=task_ids, num_segments=self.task_num)
    #     # agg_out = agg_task_feature * tf.nn.sigmoid(agg_worker_feature)
    #     # agg_out = tf.nn.softmax(agg_out, axis=-1)
    #
    #     return crowd_bias, task_feature
    #
    # def call(self, input, training=False):
    #     task_feature, answers = input
    #     cls_out = self.left_NN(task_feature, training)
    #     crowd_bias, agg_out = self.right_NN(task_feature, answers, training)
    #     return cls_out, crowd_bias, agg_out
    #
    # def loss_fuction(self, cls_out, agg_out, crowd_bias, answers, gamma=0.1):
    #
    #     # MIG loss
    #     cls_out = tf.nn.softmax(cls_out, axis=-1)
    #     agg_out = tf.nn.softmax(agg_out, axis=-1)
    #     batch_num = cls_out.shape[0]
    #     I = tf.cast(np.eye(batch_num), dtype=tf.float32)
    #     E = tf.cast(np.ones((batch_num, batch_num)), dtype=tf.float32)
    #     normalize_1 = batch_num
    #     normalize_2 = batch_num * (batch_num - 1)
    #
    #     new_output = cls_out #/ self.p_pure
    #     m = tf.matmul(new_output, agg_out, transpose_b=True)
    #     noise = np.random.rand(1) * 0.0001
    #     m1 = tf.math.log(m * I + I * noise + E - I) # i<->i + i<->j. 此处E - I是为了让log为0，以便最大化P(i,i)互信息
    #     # m1 = tf.math.log(m * I + E - I)  # i<->i + i<->j. 此处E - I是为了让log为0，以便最大化P(i,i)互信息
    #     m2 = m * (E - I) # i<->j，最小化P(i,j)互信息
    #     mig_loss = -(tf.reduce_sum(tf.reduce_sum(m1)) + batch_num) / normalize_1 + tf.reduce_sum(tf.reduce_sum(m2)) / normalize_2
    #
    #     # answer loss
    #     EC_loss = tf.nn.softmax_cross_entropy_with_logits(logits=crowd_bias, labels=tf.one_hot(indices=answers, depth=self.class_num), axis=1)
    #     # EC_loss = tf.reduce_sum(EC_loss)
    #
    #     kl_loss = -tf.reduce_sum(1 + 2 * self.task_log_sigma - tf.square(self.task_mu) - tf.square(tf.exp(self.task_log_sigma))) * 0.5
    #
    #     total_loss = mig_loss + tf.reduce_sum(EC_loss) + kl_loss
    #
    #     #loss 来自 KL，与MIG相反数
    #     return total_loss
    def __init__(self, task_num, feature_size, worker_num, class_num, answer_num=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_num = task_num
        self.feature_size = feature_size
        self.worker_num = worker_num
        self.class_num = class_num
        self.answer_num = answer_num
        self.hidden_size = 128

        # self.worker_feature = tf.Variable(tf.random.normal((self.worker_num, self.hidden_size), mean=0, stddev=0.01))
        self.worker_mu = tf.Variable(tf.random.normal((self.worker_num, self.class_num), mean=0, stddev=0.01), name='worker_mu')
        self.worker_rho = tf.Variable(tf.random.normal((self.worker_num, self.class_num), mean=0, stddev=0.01), name='worker_rho')
        # self.worker_feature = tf.one_hot(indices=range(self.worker_num), depth=self.worker_num)

        # self.flatten = tf.keras.layers.Flatten()
        # self.left_fc1 = tf.keras.layers.Dense(self.hidden_size, activation='relu', name='task_lfc_1')
        # self.left_fc2 = tf.keras.layers.Dense(self.class_num, name='task_lfc_2')
        self.Dropout = tf.keras.layers.Dropout(0.5)
        # self.left_bn = tf.keras.layers.BatchNormalization(center=False, scale=False)
        # self.left_bn1 = tf.keras.layers.BatchNormalization(center=False, scale=False)

        self.right_fc1 = tf.keras.layers.Dense(self.hidden_size, activation='relu', name='task_rfc_1')
        # self.right_fc2 = tf.keras.layers.Dense(self.class_num, activation='relu')
        # self.right_fc3 = tf.keras.layers.Dense(self.class_num, activation=None)
        self.right_bn = tf.keras.layers.BatchNormalization(center=False, scale=False)
        self.right_bn1 = tf.keras.layers.BatchNormalization(center=False, scale=False)
        self.fc_task_mu = tf.keras.layers.Dense(self.class_num, activation=None, name='task_rfc_mu')
        self.fc_task_log_sigma = tf.keras.layers.Dense(self.class_num, activation=None, name='task_rfc_log_sigma')

        self.mu_bn = tf.keras.layers.BatchNormalization(center=False, scale=False)
        # self.sigma_bn = tf.keras.layers.BatchNormalization(center=False, scale=False)

        self.right_fc2 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        self.right_fc3 = tf.keras.layers.Dense(self.feature_size, activation=None)

        # self.fc_worker_mu = tf.keras.layers.Dense(self.class_num, activation='relu')
        # self.fc_worker_rho = tf.keras.layers.Dense(self.class_num, activation='relu')

        self.p_pure = np.ones((self.class_num, ), dtype=np.float32)/self.class_num

        # self.weight = tf.Variable(tf.ones(self.answer_num,))

        # self.gnn1 = tfg.layers.GAT(self.hidden_size, activation=None, num_heads=8, attention_units=8, edge_drop_rate=0.1)
        # self.gnn1 = tfg.layers.GCN(self.hidden_size, activation=tf.nn.relu)
        # self.gnn2 = tfg.layers.GCN(self.class_num)

    def sample(self, mean, log_std):
        std = tf.math.exp(log_std)
        eps = tf.random.normal(shape=(self.task_num, self.class_num), mean=0, stddev=0.01)
        return mean + eps * std

    # def left_NN(self, task_feature, training=None):
    #
    #     task_feature = self.left_bn(task_feature)
    #     task_feature = self.left_fc1(task_feature)
    #     task_feature = self.Dropout(task_feature, training)
    #     task_feature = self.left_bn1(task_feature)
    #     cls_out = self.left_fc2(task_feature)
    #
    #     return cls_out

    def right_NN(self, task_feature, training=None):
        # task_feature = self.right_bn(task_feature)
        task_feature = self.right_fc1(task_feature)
        task_feature = self.Dropout(task_feature, training)
        # task_feature = self.right_bn1(task_feature)
        # task_feature = self.right_fc3(task_feature)

        self.task_mu = self.fc_task_mu(task_feature)
        self.task_log_sigma = self.fc_task_log_sigma(task_feature)
        # task_mu = self.mu_bn(task_mu)
        # task_log_sigma = self.sigma_bn(task_log_sigma)
        z = self.sample(self.task_mu, self.task_log_sigma)

        x = self.right_fc2(z)
        x = self.right_bn1(x)
        x = self.right_fc3(x)

        return z, x, self.task_mu, self.task_log_sigma

    def worker_NN(self, task_feature, answers):

        task_ids = answers[:, 0]
        worker_ids = answers[:, 1]
        label_ids = answers[:, 2]
        row = task_ids
        col = worker_ids + self.task_num
        Row = np.concatenate([row, col], axis=-1)
        Col = np.concatenate([col, row], axis=-1)
        edge_index = np.concatenate([[Row], [Col]], axis=0)

        masked_task_feature = tf.gather(task_feature, row)
        self.masked_task_mu = tf.gather(self.task_mu, row)
        self.masked_task_sigma = tf.math.exp(tf.gather(self.task_log_sigma, row))
        self.masked_worker_mu = tf.gather(self.worker_mu, col - self.task_num)
        self.masked_worker_rho = tf.gather(self.worker_rho, col - self.task_num)


        crowd_bias = masked_task_feature * tf.nn.softplus(self.masked_worker_rho) + self.masked_worker_mu # tf.nn.sigmoid(masked_worker_rho) tf.math.log(1+tf.math.exp(masked_worker_rho))

        # self.poster_prior = self.kl_Qwtheta_Pw(worker_mu_kernel, self.workerMu_mu, self.workerMu_rho) \
        #                      + \
        #                      self.kl_Qwtheta_Pw(worker_sigma_kernel, self.workerRho_mu, self.workerRho_rho)

        return crowd_bias

    def gaussian_distribution_density(self, x, mu, sigma):
        return 1.0 / ((2 * np.pi) ** 0.5 * sigma) * tf.math.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + 1e-30

    def call(self, input, training=False):
        task_feature, answers = input
        # cls_out = self.left_NN(task_feature, training)
        sample_task_feature, recons_task_feature, task_mu, task_log_sigma = self.right_NN(task_feature, training)
        crowd_bias  = self.worker_NN(sample_task_feature, answers)
        return crowd_bias, sample_task_feature, recons_task_feature, task_mu, task_log_sigma

    def kl_Qwtheta_Pw(self, w, mu, sigma):
        Qwtheta = tf.math.log(self.gaussian_distribution_density(w, mu, sigma))  # poster
        Pw = tf.math.log(0.5 * self.gaussian_distribution_density(w, 0.0, 1.5)
                         +
                         0.5 * self.gaussian_distribution_density(w, 0.0, 0.1))  # prior
        return tf.math.reduce_mean(Qwtheta - Pw)

    def MIG_loss(self, cls_out, agg_out):
        # MIG loss
        cls_out = tf.nn.softmax(cls_out, axis=-1)
        agg_out = tf.nn.softmax(agg_out, axis=-1)
        batch_num = cls_out.shape[0]
        I = tf.cast(np.eye(batch_num), dtype=tf.float32)
        E = tf.cast(np.ones((batch_num, batch_num)), dtype=tf.float32)
        normalize_1 = batch_num
        normalize_2 = batch_num * (batch_num - 1)

        new_output = cls_out  # / self.p_pure
        m = tf.matmul(new_output, agg_out, transpose_b=True)
        noise = np.random.rand(1) * 0.0001
        m1 = tf.math.log(m * I + I * noise + E - I)  # i<->i + i<->j. 此处E - I是为了让log为0，以便最大化P(i,i)互信息
        # m1 = tf.math.log(m * I + E - I)  # i<->i + i<->j. 此处E - I是为了让log为0，以便最大化P(i,i)互信息
        m2 = m * (E - I)  # i<->j，最小化P(i,j)互信息
        mig_loss = -(tf.reduce_sum(tf.reduce_sum(m1)) + batch_num) / normalize_1 + tf.reduce_sum(tf.reduce_sum(m2)) / normalize_2
        return mig_loss

    def CE_loss(self, crowd_bias, answers):
        return tf.nn.softmax_cross_entropy_with_logits(logits=crowd_bias,
                                                        labels=tf.one_hot(indices=answers, depth=self.class_num),
                                                        axis=1)

    def KL_loss(self, task_mu, task_log_sigma):
        return -tf.reduce_sum(1 + 2 * task_log_sigma - tf.square(task_mu) - tf.square(tf.exp(task_log_sigma))) * 0.5

    def loss_fuction(self, task_feature, recons_task_feature, crowd_bias, answers, task_mu, task_log_sigma, gamma=0.1):

        # poster

        poster = self.kl_Qwtheta_Pw(self.masked_worker_mu, self.masked_task_mu, self.masked_task_sigma) \
                 + \
                 self.kl_Qwtheta_Pw(self.masked_worker_rho, self.masked_task_mu, self.masked_task_sigma)

        #recons_loss
        mig_loss = self.MIG_loss(task_feature, recons_task_feature)

        # recons_loss
        # recons_loss = tf.reduce_mean(tf.sqrt(tf.square(task_feature - recons_task_feature))) * 0
        # print(recons_loss)

        # KL_loss
        kl_loss = self.KL_loss(task_mu, task_log_sigma) #+ self.KL_loss(self.worker_mu, self.worker_rho)

        # CE_loss
        EC_loss = self.CE_loss(crowd_bias, answers)

        total_loss = mig_loss + kl_loss + tf.reduce_sum(EC_loss) - poster

        #loss 来自 KL，与MIG相反数
        return total_loss
        # return tf.reduce_sum(EC_loss), mig_loss, kl_loss

class Music_model(tf.keras.Model):
    def __init__(self, task_num, feature_size, worker_num, class_num, answer_num=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_num = task_num
        self.feature_size = feature_size
        self.worker_num = worker_num
        self.class_num = class_num
        self.answer_num = answer_num
        self.hidden_size = 128

        # self.worker_feature = tf.Variable(tf.random.normal((self.worker_num, self.hidden_size), mean=0, stddev=0.01))
        self.worker_mu = tf.Variable(tf.random.normal((self.worker_num, self.class_num), mean=0, stddev=0.01), name='worker_mu')
        self.worker_rho = tf.Variable(tf.random.normal((self.worker_num, self.class_num), mean=0, stddev=0.01), name='worker_rho')
        # self.worker_feature = tf.one_hot(indices=range(self.worker_num), depth=self.worker_num)

        # self.flatten = tf.keras.layers.Flatten()
        # self.left_fc1 = tf.keras.layers.Dense(self.hidden_size, activation='relu', name='task_lfc_1')
        # self.left_fc2 = tf.keras.layers.Dense(self.class_num, name='task_lfc_2')
        self.Dropout = tf.keras.layers.Dropout(0.5)
        # self.left_bn = tf.keras.layers.BatchNormalization(center=False, scale=False)
        # self.left_bn1 = tf.keras.layers.BatchNormalization(center=False, scale=False)

        self.right_fc1 = tf.keras.layers.Dense(self.hidden_size, activation='relu', name='task_rfc_1')
        # self.right_fc2 = tf.keras.layers.Dense(self.class_num, activation='relu')
        # self.right_fc3 = tf.keras.layers.Dense(self.class_num, activation=None)
        self.right_bn = tf.keras.layers.BatchNormalization(center=False, scale=False)
        self.right_bn1 = tf.keras.layers.BatchNormalization(center=False, scale=False)
        self.fc_task_mu = tf.keras.layers.Dense(self.class_num, activation=None, name='task_rfc_mu')
        self.fc_task_log_sigma = tf.keras.layers.Dense(self.class_num, activation=None, name='task_rfc_log_sigma')

        self.mu_bn = tf.keras.layers.BatchNormalization(center=False, scale=False)
        # self.sigma_bn = tf.keras.layers.BatchNormalization(center=False, scale=False)

        self.right_fc2 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        self.right_fc3 = tf.keras.layers.Dense(self.feature_size, activation=None)

        # self.fc_worker_mu = tf.keras.layers.Dense(self.class_num, activation='relu')
        # self.fc_worker_rho = tf.keras.layers.Dense(self.class_num, activation='relu')

        self.p_pure = np.ones((self.class_num, ), dtype=np.float32)/self.class_num

        # self.weight = tf.Variable(tf.ones(self.answer_num,))

        # self.gnn1 = tfg.layers.GAT(self.hidden_size, activation=None, num_heads=8, attention_units=8, edge_drop_rate=0.1)
        # self.gnn1 = tfg.layers.GCN(self.hidden_size, activation=tf.nn.relu)
        # self.gnn2 = tfg.layers.GCN(self.class_num)

    def sample(self, mean, log_std):
        std = tf.math.exp(log_std)
        eps = tf.random.normal(shape=(self.task_num, self.class_num), mean=0, stddev=0.01)
        return mean + eps * std

    # def left_NN(self, task_feature, training=None):
    #
    #     task_feature = self.left_bn(task_feature)
    #     task_feature = self.left_fc1(task_feature)
    #     task_feature = self.Dropout(task_feature, training)
    #     task_feature = self.left_bn1(task_feature)
    #     cls_out = self.left_fc2(task_feature)
    #
    #     return cls_out

    def right_NN(self, task_feature, training=None):
        task_feature = self.right_bn(task_feature)
        task_feature = self.right_fc1(task_feature)
        task_feature = self.Dropout(task_feature, training)
        # task_feature = self.right_bn1(task_feature)
        # task_feature = self.right_fc3(task_feature)

        self.task_mu = self.fc_task_mu(task_feature)
        self.task_log_sigma = self.fc_task_log_sigma(task_feature)
        # task_mu = self.mu_bn(task_mu)
        # task_log_sigma = self.sigma_bn(task_log_sigma)
        z = self.sample(self.task_mu, self.task_log_sigma)

        x = self.right_fc2(z)
        x = self.right_bn1(x)
        x = self.right_fc3(x)

        return z, x, self.task_mu, self.task_log_sigma

    def worker_NN(self, task_feature, answers):

        task_ids = answers[:, 0]
        worker_ids = answers[:, 1]
        label_ids = answers[:, 2]
        row = task_ids
        col = worker_ids + self.task_num
        Row = np.concatenate([row, col], axis=-1)
        Col = np.concatenate([col, row], axis=-1)
        edge_index = np.concatenate([[Row], [Col]], axis=0)

        masked_task_feature = tf.gather(task_feature, row)
        self.masked_task_mu = tf.gather(self.task_mu, row)
        self.masked_task_sigma = tf.math.exp(tf.gather(self.task_log_sigma, row))
        self.masked_worker_mu = tf.gather(self.worker_mu, col - self.task_num)
        self.masked_worker_rho = tf.gather(self.worker_rho, col - self.task_num)


        crowd_bias = masked_task_feature * tf.nn.softplus(self.masked_worker_rho) + self.masked_worker_mu # tf.nn.sigmoid(masked_worker_rho) tf.math.log(1+tf.math.exp(masked_worker_rho))

        # self.poster_prior = self.kl_Qwtheta_Pw(worker_mu_kernel, self.workerMu_mu, self.workerMu_rho) \
        #                      + \
        #                      self.kl_Qwtheta_Pw(worker_sigma_kernel, self.workerRho_mu, self.workerRho_rho)

        return crowd_bias

    def gaussian_distribution_density(self, x, mu, sigma):
        return 1.0 / ((2 * np.pi * sigma) ** 0.5) * tf.math.exp(-(x - mu) ** 2 / (2 * sigma)) + 1e-30

    def call(self, input, training=False):
        task_feature, answers = input
        # cls_out = self.left_NN(task_feature, training)
        sample_task_feature, recons_task_feature, task_mu, task_log_sigma = self.right_NN(task_feature, training)
        crowd_bias  = self.worker_NN(sample_task_feature, answers)
        return crowd_bias, sample_task_feature, recons_task_feature, task_mu, task_log_sigma

    def kl_Qwtheta_Pw(self, w, mu, sigma):
        Qwtheta = tf.math.log(self.gaussian_distribution_density(w, mu, sigma))  # poster
        Pw = tf.math.log(self.gaussian_distribution_density(w, 0.0, 0.01))
                         # +
                         # 0.5 * self.gaussian_distribution_density(w, 0.0, 0.1))  # prior
        return tf.math.reduce_mean(Qwtheta - Pw)

    def MSE_loss(self, cls_out, agg_out):
        # # MIG loss
        # cls_out = tf.nn.softmax(cls_out, axis=-1)
        # agg_out = tf.nn.softmax(agg_out, axis=-1)
        # batch_num = cls_out.shape[0]
        # I = tf.cast(np.eye(batch_num), dtype=tf.float32)
        # E = tf.cast(np.ones((batch_num, batch_num)), dtype=tf.float32)
        # normalize_1 = batch_num
        # normalize_2 = batch_num * (batch_num - 1)
        #
        # new_output = cls_out  # / self.p_pure
        # m = tf.matmul(new_output, agg_out, transpose_b=True)
        # noise = np.random.rand(1) * 0.0001
        # m1 = tf.math.log(m * I + I * noise + E - I)  # i<->i + i<->j. 此处E - I是为了让log为0，以便最大化P(i,i)互信息
        # # m1 = tf.math.log(m * I + E - I)  # i<->i + i<->j. 此处E - I是为了让log为0，以便最大化P(i,i)互信息
        # m2 = m * (E - I)  # i<->j，最小化P(i,j)互信息
        # mig_loss = -(tf.reduce_sum(tf.reduce_sum(m1)) + batch_num) / normalize_1 + tf.reduce_sum(tf.reduce_sum(m2)) / normalize_2
        #
        # return mig_loss
        return tf.keras.losses.MeanSquaredError()(cls_out, agg_out)

    def CE_loss(self, crowd_bias, answers):
        return tf.nn.softmax_cross_entropy_with_logits(logits=crowd_bias,
                                                        labels=tf.one_hot(indices=answers, depth=self.class_num),
                                                        axis=1)

    def KL_loss(self, task_mu, task_log_sigma):
        return -tf.reduce_sum(1 + 2 * task_log_sigma - tf.square(task_mu) - tf.square(tf.exp(task_log_sigma))) * 0.5
        # return -tf.reduce_sum(1 + 2 * tf.ones_like(task_log_sigma) - tf.square(task_mu) - tf.ones_like(task_log_sigma)) * 0.5

    def loss_fuction(self, task_feature, recons_task_feature, crowd_bias, answers, task_mu, task_log_sigma, gamma=0.1):
        # CE_loss
        EC_loss = self.CE_loss(crowd_bias, answers)

        # poster # tf.math.reduce_sum(tf.math.exp(2 * self.task_log_sigma), axis=0)
        # poster = self.kl_Qwtheta_Pw(self.worker_mu, tf.math.reduce_sum(self.task_mu / tf.math.exp(2 * self.task_log_sigma), axis=0)/tf.math.reduce_sum(1 / tf.math.exp(2 * self.task_log_sigma), axis=0),
        #                             tf.math.reduce_sum(tf.math.exp(2 * self.task_log_sigma), axis=0)) \
        #          + \
        #          self.kl_Qwtheta_Pw(self.worker_rho, tf.math.reduce_sum(self.task_mu / tf.math.exp(2 * self.task_log_sigma), axis=0)/tf.math.reduce_sum(1 / tf.math.exp(2 * self.task_log_sigma), axis=0),
        #                             tf.math.reduce_sum(tf.math.exp(2 * self.task_log_sigma), axis=0))
        poster = self.kl_Qwtheta_Pw(self.masked_worker_mu,
                                    self.masked_task_mu,
                                    tf.math.pow(self.masked_task_sigma, 2)) \
                 + \
                 self.kl_Qwtheta_Pw(self.masked_worker_rho,
                                    self.masked_task_mu,
                                    tf.math.pow(self.masked_task_sigma, 2))

        #recons_loss
        mse_loss = self.MSE_loss(task_feature, recons_task_feature)

        # KL_loss
        kl_loss = self.KL_loss(task_mu, task_log_sigma) #+ self.KL_loss(self.worker_mu, self.worker_rho)

        total_loss = mse_loss + kl_loss + tf.reduce_sum(EC_loss) + poster

        #loss 来自 KL，与MIG相反数
        return total_loss
        # return tf.reduce_sum(EC_loss), mig_loss, kl_loss

class SP_model(tf.keras.Model):
    def __init__(self, task_num, feature_size, worker_num, class_num, answer_num=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_num = task_num
        self.feature_size = feature_size
        self.worker_num = worker_num
        self.class_num = class_num
        self.answer_num = answer_num
        self.hidden_size = 128

        # self.worker_feature = tf.Variable(tf.random.normal((self.worker_num, self.hidden_size), mean=0, stddev=0.01))
        self.worker_mu = tf.Variable(tf.random.normal((self.worker_num, self.class_num), mean=0, stddev=0.01), name='worker_mu')
        self.worker_rho = tf.Variable(tf.random.normal((self.worker_num, self.class_num), mean=0, stddev=0.01), name='worker_rho')
        # self.worker_feature = tf.one_hot(indices=range(self.worker_num), depth=self.worker_num)

        # self.flatten = tf.keras.layers.Flatten()
        # self.left_fc1 = tf.keras.layers.Dense(self.hidden_size, activation='relu', name='task_lfc_1')
        # self.left_fc2 = tf.keras.layers.Dense(self.class_num, name='task_lfc_2')
        self.Dropout = tf.keras.layers.Dropout(0.5)
        # self.left_bn = tf.keras.layers.BatchNormalization(center=False, scale=False)
        # self.left_bn1 = tf.keras.layers.BatchNormalization(center=False, scale=False)

        self.right_fc1 = tf.keras.layers.Dense(self.hidden_size, activation='relu', name='task_rfc_1')
        # self.right_fc2 = tf.keras.layers.Dense(self.class_num, activation='relu')
        # self.right_fc3 = tf.keras.layers.Dense(self.class_num, activation=None)
        # self.right_bn = tf.keras.layers.BatchNormalization(center=False, scale=False)
        # self.right_bn1 = tf.keras.layers.BatchNormalization(center=False, scale=False)
        self.fc_task_mu = tf.keras.layers.Dense(self.class_num, activation=None, name='task_rfc_mu')
        self.fc_task_log_sigma = tf.keras.layers.Dense(self.class_num, activation=None, name='task_rfc_log_sigma')

        self.mu_bn = tf.keras.layers.BatchNormalization(center=False, scale=False)
        # self.sigma_bn = tf.keras.layers.BatchNormalization(center=False, scale=False)

        self.right_fc2 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        self.right_fc3 = tf.keras.layers.Dense(self.feature_size, activation=None)

        # self.fc_worker_mu = tf.keras.layers.Dense(self.class_num, activation='relu')
        # self.fc_worker_rho = tf.keras.layers.Dense(self.class_num, activation='relu')

        self.p_pure = np.ones((self.class_num, ), dtype=np.float32)/self.class_num

        # self.weight = tf.Variable(tf.ones(self.answer_num,))

        # self.gnn1 = tfg.layers.GAT(self.hidden_size, activation=None, num_heads=8, attention_units=8, edge_drop_rate=0.1)
        # self.gnn1 = tfg.layers.GCN(self.hidden_size, activation=tf.nn.relu)
        # self.gnn2 = tfg.layers.GCN(self.class_num)

    def sample(self, mean, log_std):
        std = tf.math.exp(log_std)
        eps = tf.random.normal(shape=(self.task_num, self.class_num), mean=0, stddev=0.01)
        return mean + eps * std

    def right_NN(self, task_feature, training=None):
        # task_feature = self.right_bn(task_feature)
        task_feature = self.right_fc1(task_feature)
        task_feature = self.Dropout(task_feature, training)
        # task_feature = self.right_bn1(task_feature)
        # task_feature = self.right_fc3(task_feature)

        self.task_mu = self.fc_task_mu(task_feature)
        self.task_log_sigma = self.fc_task_log_sigma(task_feature)
        # task_mu = self.mu_bn(task_mu)
        # task_log_sigma = self.sigma_bn(task_log_sigma)
        z = self.sample(self.task_mu, self.task_log_sigma)

        x = self.right_fc2(z)
        # x = self.right_bn1(x)
        x = self.right_fc3(x)

        return z, x, self.task_mu, self.task_log_sigma

    def worker_NN(self, task_feature, answers):

        task_ids = answers[:, 0]
        worker_ids = answers[:, 1]
        label_ids = answers[:, 2]
        row = task_ids
        col = worker_ids + self.task_num
        Row = np.concatenate([row, col], axis=-1)
        Col = np.concatenate([col, row], axis=-1)
        edge_index = np.concatenate([[Row], [Col]], axis=0)

        masked_task_feature = tf.gather(task_feature, row)
        self.masked_task_mu = tf.gather(self.task_mu, row)
        self.masked_task_sigma = tf.math.exp(tf.gather(self.task_log_sigma, row))
        self.masked_worker_mu = tf.gather(self.worker_mu, col - self.task_num)
        self.masked_worker_rho = tf.gather(self.worker_rho, col - self.task_num)


        crowd_bias = masked_task_feature * tf.nn.softplus(self.masked_worker_rho) + self.masked_worker_mu # tf.nn.sigmoid(masked_worker_rho) tf.math.log(1+tf.math.exp(masked_worker_rho))

        # self.poster_prior = self.kl_Qwtheta_Pw(worker_mu_kernel, self.workerMu_mu, self.workerMu_rho) \
        #                      + \
        #                      self.kl_Qwtheta_Pw(worker_sigma_kernel, self.workerRho_mu, self.workerRho_rho)

        return crowd_bias

    def gaussian_distribution_density(self, x, mu, sigma):
        return 1.0 / ((2 * np.pi) ** 0.5 * sigma) * tf.math.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + 1e-30

    def call(self, input, training=False):
        task_feature, answers = input
        # cls_out = self.left_NN(task_feature, training)
        sample_task_feature, recons_task_feature, task_mu, task_log_sigma = self.right_NN(task_feature, training)
        crowd_bias  = self.worker_NN(sample_task_feature, answers)
        return crowd_bias, sample_task_feature, recons_task_feature, task_mu, task_log_sigma

    def kl_Qwtheta_Pw(self, w, mu, sigma):
        Qwtheta = tf.math.log(self.gaussian_distribution_density(w, mu, sigma))  # poster
        Pw = tf.math.log(self.gaussian_distribution_density(w, 0.0, 0.01))
                         # +
                         # 0.5 * self.gaussian_distribution_density(w, 0.0, 0.1))  # prior
        return tf.math.reduce_mean(Qwtheta - Pw)

    def MIG_loss(self, cls_out, agg_out):
        # # MIG loss
        # cls_out = tf.nn.softmax(cls_out, axis=-1)
        # agg_out = tf.nn.softmax(agg_out, axis=-1)
        # batch_num = cls_out.shape[0]
        # I = tf.cast(np.eye(batch_num), dtype=tf.float32)
        # E = tf.cast(np.ones((batch_num, batch_num)), dtype=tf.float32)
        # normalize_1 = batch_num
        # normalize_2 = batch_num * (batch_num - 1)
        #
        # new_output = cls_out  # / self.p_pure
        # m = tf.matmul(new_output, agg_out, transpose_b=True)
        # noise = np.random.rand(1) * 0.0001
        # m1 = tf.math.log(m * I + I * noise + E - I)  # i<->i + i<->j. 此处E - I是为了让log为0，以便最大化P(i,i)互信息
        # # m1 = tf.math.log(m * I + E - I)  # i<->i + i<->j. 此处E - I是为了让log为0，以便最大化P(i,i)互信息
        # m2 = m * (E - I)  # i<->j，最小化P(i,j)互信息
        # mig_loss = -(tf.reduce_sum(tf.reduce_sum(m1)) + batch_num) / normalize_1 + tf.reduce_sum(tf.reduce_sum(m2)) / normalize_2
        # return mig_loss
        return tf.keras.losses.MeanSquaredError()(cls_out, agg_out)

    def CE_loss(self, crowd_bias, answers):
        return tf.nn.softmax_cross_entropy_with_logits(logits=crowd_bias,
                                                        labels=tf.one_hot(indices=answers, depth=self.class_num),
                                                        axis=1)

    def KL_loss(self, task_mu, task_log_sigma):
        return -tf.reduce_sum(1 + 2 * task_log_sigma - tf.square(task_mu) - tf.square(tf.exp(task_log_sigma))) * 0.5

    def loss_fuction(self, task_feature, recons_task_feature, crowd_bias, answers, task_mu, task_log_sigma, gamma=0.1):

        # poster

        # poster = self.kl_Qwtheta_Pw(self.worker_mu,
        #                             tf.math.reduce_sum(self.task_mu / tf.math.exp(2 * self.task_log_sigma),
        #                                                axis=0) / tf.math.reduce_sum(
        #                                 1 / tf.math.exp(2 * self.task_log_sigma), axis=0),
        #                             tf.math.reduce_sum(tf.math.exp(2 * self.task_log_sigma), axis=0)) \
        #          + \
        #          self.kl_Qwtheta_Pw(self.worker_rho,
        #                             tf.math.reduce_sum(self.task_mu / tf.math.exp(2 * self.task_log_sigma),
        #                                                axis=0) / tf.math.reduce_sum(
        #                                 1 / tf.math.exp(2 * self.task_log_sigma), axis=0),
        #                             tf.math.reduce_sum(tf.math.exp(2 * self.task_log_sigma), axis=0))

        poster = self.kl_Qwtheta_Pw(self.masked_worker_mu,
                                    self.masked_task_mu,
                                    tf.math.pow(self.masked_task_sigma, 2)) \
                 + \
                 self.kl_Qwtheta_Pw(self.masked_worker_rho,
                                    self.masked_task_mu,
                                    tf.math.pow(self.masked_task_sigma, 2))

        #recons_loss
        mig_loss = self.MIG_loss(task_feature, recons_task_feature)

        # recons_loss
        # recons_loss = tf.reduce_mean(tf.sqrt(tf.square(task_feature - recons_task_feature))) * 0
        # print(recons_loss)

        # KL_loss
        kl_loss = self.KL_loss(task_mu, task_log_sigma) #+ self.KL_loss(self.worker_mu, self.worker_rho)

        # CE_loss
        EC_loss = self.CE_loss(crowd_bias, answers)

        total_loss = mig_loss + kl_loss + tf.reduce_sum(EC_loss) + poster

        #loss 来自 KL，与MIG相反数
        return total_loss
        # return tf.reduce_sum(EC_loss), mig_loss, kl_loss

class BCD_model(tf.keras.Model):
    def __init__(self, task_num, feature_size, worker_num, class_num, answer_num=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_num = task_num
        self.feature_size = feature_size
        self.worker_num = worker_num
        self.class_num = class_num
        self.answer_num = answer_num
        self.hidden_size = 128

        # self.worker_feature = tf.Variable(tf.random.normal((self.worker_num, self.hidden_size), mean=0, stddev=0.01))
        self.worker_mu = tf.Variable(tf.random.normal((self.worker_num, self.class_num), mean=0, stddev=0.01),
                                     name='worker_mu')
        self.worker_rho = tf.Variable(tf.random.normal((self.worker_num, self.class_num), mean=0, stddev=0.01),
                                      name='worker_rho')
        # self.worker_feature = tf.one_hot(indices=range(self.worker_num), depth=self.worker_num)

        # self.flatten = tf.keras.layers.Flatten()
        # self.left_fc1 = tf.keras.layers.Dense(self.hidden_size, activation='relu', name='task_lfc_1')
        # self.left_fc2 = tf.keras.layers.Dense(self.class_num, name='task_lfc_2')
        self.Dropout = tf.keras.layers.Dropout(0.5)
        # self.left_bn = tf.keras.layers.BatchNormalization(center=False, scale=False)
        # self.left_bn1 = tf.keras.layers.BatchNormalization(center=False, scale=False)

        self.right_fc1 = tf.keras.layers.Dense(self.hidden_size, activation='relu', name='task_rfc_1')
        # self.right_fc2 = tf.keras.layers.Dense(self.class_num, activation='relu')
        # self.right_fc3 = tf.keras.layers.Dense(self.class_num, activation=None)
        self.right_bn = tf.keras.layers.BatchNormalization(center=False, scale=False)
        # self.right_bn1 = tf.keras.layers.BatchNormalization(center=False, scale=False)
        self.fc_task_mu = tf.keras.layers.Dense(self.class_num, activation=None, name='task_rfc_mu')
        self.fc_task_log_sigma = tf.keras.layers.Dense(self.class_num, activation=None, name='task_rfc_log_sigma')

        self.mu_bn = tf.keras.layers.BatchNormalization(center=False, scale=False)
        # self.sigma_bn = tf.keras.layers.BatchNormalization(center=False, scale=False)

        self.right_fc2 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        self.right_fc3 = tf.keras.layers.Dense(self.feature_size, activation=None)

        # self.fc_worker_mu = tf.keras.layers.Dense(self.class_num, activation='relu')
        # self.fc_worker_rho = tf.keras.layers.Dense(self.class_num, activation='relu')

        self.p_pure = np.ones((self.class_num,), dtype=np.float32) / self.class_num

        # self.weight = tf.Variable(tf.ones(self.answer_num,))

        # self.gnn1 = tfg.layers.GAT(self.hidden_size, activation=None, num_heads=8, attention_units=8, edge_drop_rate=0.1)
        # self.gnn1 = tfg.layers.GCN(self.hidden_size, activation=tf.nn.relu)
        # self.gnn2 = tfg.layers.GCN(self.class_num)

    def sample(self, mean, log_std):
        std = tf.math.exp(log_std)
        eps = tf.random.normal(shape=(self.task_num, self.class_num), mean=0, stddev=0.01)
        return mean + eps * std

    def right_NN(self, task_feature, training=None):
        # task_feature = self.right_bn(task_feature)
        task_feature = self.right_fc1(task_feature)
        task_feature = self.Dropout(task_feature, training)
        # task_feature = self.right_bn1(task_feature)
        # task_feature = self.right_fc3(task_feature)

        self.task_mu = self.fc_task_mu(task_feature)
        self.task_log_sigma = self.fc_task_log_sigma(task_feature)
        # task_mu = self.mu_bn(task_mu)
        # task_log_sigma = self.sigma_bn(task_log_sigma)
        z = self.sample(self.task_mu, self.task_log_sigma)

        x = self.right_fc2(z)
        # x = self.right_bn1(x)
        x = self.right_fc3(x)

        return z, x, self.task_mu, self.task_log_sigma

    def worker_NN(self, task_feature, answers):
        task_ids = answers[:, 0]
        worker_ids = answers[:, 1]
        label_ids = answers[:, 2]
        row = task_ids
        col = worker_ids + self.task_num
        Row = np.concatenate([row, col], axis=-1)
        Col = np.concatenate([col, row], axis=-1)
        edge_index = np.concatenate([[Row], [Col]], axis=0)

        masked_task_feature = tf.gather(task_feature, row)
        self.masked_task_mu = tf.gather(self.task_mu, row)
        self.masked_task_sigma = tf.math.exp(tf.gather(self.task_log_sigma, row))
        self.masked_worker_mu = tf.gather(self.worker_mu, col - self.task_num)
        self.masked_worker_rho = tf.gather(self.worker_rho, col - self.task_num)

        crowd_bias = masked_task_feature * tf.nn.softplus(
            self.masked_worker_rho) + self.masked_worker_mu  # tf.nn.sigmoid(masked_worker_rho) tf.math.log(1+tf.math.exp(masked_worker_rho))

        # self.poster_prior = self.kl_Qwtheta_Pw(worker_mu_kernel, self.workerMu_mu, self.workerMu_rho) \
        #                      + \
        #                      self.kl_Qwtheta_Pw(worker_sigma_kernel, self.workerRho_mu, self.workerRho_rho)

        return crowd_bias

    def gaussian_distribution_density(self, x, mu, sigma):
        return 1.0 / ((2 * np.pi) ** 0.5 * sigma) * tf.math.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + 1e-30

    def call(self, input, training=False):
        task_feature, answers = input
        # cls_out = self.left_NN(task_feature, training)
        sample_task_feature, recons_task_feature, task_mu, task_log_sigma = self.right_NN(task_feature, training)
        crowd_bias = self.worker_NN(sample_task_feature, answers)
        return crowd_bias, sample_task_feature, recons_task_feature, task_mu, task_log_sigma

    def kl_Qwtheta_Pw(self, w, mu, sigma):
        Qwtheta = tf.math.log(self.gaussian_distribution_density(w, mu, sigma))  # poster
        Pw = tf.math.log(self.gaussian_distribution_density(w, 0.0, 0.01))
                         # +
                         # 0.5 * self.gaussian_distribution_density(w, 0.0, 0.1))  # prior
        return tf.math.reduce_mean(Qwtheta - Pw)

    def MIG_loss(self, cls_out, agg_out):
        # # MIG loss
        # cls_out = tf.nn.softmax(cls_out, axis=-1)
        # agg_out = tf.nn.softmax(agg_out, axis=-1)
        # batch_num = cls_out.shape[0]
        # I = tf.cast(np.eye(batch_num), dtype=tf.float32)
        # E = tf.cast(np.ones((batch_num, batch_num)), dtype=tf.float32)
        # normalize_1 = batch_num
        # normalize_2 = batch_num * (batch_num - 1)
        #
        # new_output = cls_out  # / self.p_pure
        # m = tf.matmul(new_output, agg_out, transpose_b=True)
        # noise = np.random.rand(1) * 0.0001
        # m1 = tf.math.log(m * I + I * noise + E - I)  # i<->i + i<->j. 此处E - I是为了让log为0，以便最大化P(i,i)互信息
        # # m1 = tf.math.log(m * I + E - I)  # i<->i + i<->j. 此处E - I是为了让log为0，以便最大化P(i,i)互信息
        # m2 = m * (E - I)  # i<->j，最小化P(i,j)互信息
        # mig_loss = -(tf.reduce_sum(tf.reduce_sum(m1)) + batch_num) / normalize_1 + tf.reduce_sum(
        #     tf.reduce_sum(m2)) / normalize_2
        # return mig_loss

        return tf.keras.losses.MeanSquaredError()(cls_out, agg_out)

    def CE_loss(self, crowd_bias, answers):
        return tf.nn.softmax_cross_entropy_with_logits(logits=crowd_bias,
                                                       labels=tf.one_hot(indices=answers, depth=self.class_num),
                                                       axis=1)

    def KL_loss(self, task_mu, task_log_sigma):
        return -tf.reduce_sum(1 + 2 * task_log_sigma - tf.square(task_mu) - tf.square(tf.exp(task_log_sigma))) * 0.5

    def loss_fuction(self, task_feature, recons_task_feature, crowd_bias, answers, task_mu, task_log_sigma, gamma=0.1):
        # poster

        # poster = self.kl_Qwtheta_Pw(self.worker_mu,
        #                             tf.math.reduce_sum(self.task_mu / tf.math.exp(2 * self.task_log_sigma),
        #                                                axis=0) / tf.math.reduce_sum(
        #                                 1 / tf.math.exp(2 * self.task_log_sigma), axis=0),
        #                             tf.math.reduce_sum(tf.math.exp(2 * self.task_log_sigma), axis=0)) \
        #          + \
        #          self.kl_Qwtheta_Pw(self.worker_rho,
        #                             tf.math.reduce_sum(self.task_mu / tf.math.exp(2 * self.task_log_sigma),
        #                                                axis=0) / tf.math.reduce_sum(
        #                                 1 / tf.math.exp(2 * self.task_log_sigma), axis=0),
        #                             tf.math.reduce_sum(tf.math.exp(2 * self.task_log_sigma), axis=0))

        poster = self.kl_Qwtheta_Pw(self.masked_worker_mu,
                                    self.masked_task_mu,
                                    tf.math.pow(self.masked_task_sigma, 2)) \
                 + \
                 self.kl_Qwtheta_Pw(self.masked_worker_rho,
                                    self.masked_task_mu,
                                    tf.math.pow(self.masked_task_sigma, 2))

        # recons_loss
        mig_loss = self.MIG_loss(task_feature, recons_task_feature)

        # recons_loss
        # recons_loss = tf.reduce_mean(tf.sqrt(tf.square(task_feature - recons_task_feature))) * 0
        # print(recons_loss)

        # KL_loss
        kl_loss = self.KL_loss(task_mu, task_log_sigma)  # + self.KL_loss(self.worker_mu, self.worker_rho)

        # CE_loss
        EC_loss = self.CE_loss(crowd_bias, answers)

        total_loss = mig_loss + kl_loss + tf.reduce_sum(EC_loss) + poster

        # loss 来自 KL，与MIG相反数
        return total_loss
        # return tf.reduce_sum(EC_loss), mig_loss, kl_loss

class Reuters_model(tf.keras.Model):
    def __init__(self, task_num, feature_size, worker_num, class_num, answer_num, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_num = task_num
        self.feature_size = feature_size
        self.worker_num = worker_num
        self.class_num = class_num
        self.answer_num = answer_num
        self.hidden_size = 128

        # self.worker_feature = tf.Variable(tf.random.normal((self.worker_num, self.hidden_size), mean=0, stddev=0.01))
        self.worker_mu = tf.Variable(tf.random.normal((self.worker_num, self.class_num), mean=0, stddev=0.01))
        self.worker_rho = tf.Variable(tf.random.normal((self.worker_num, self.class_num), mean=0, stddev=0.01))
        # self.worker_feature = tf.one_hot(indices=range(self.worker_num), depth=self.worker_num)

        self.flatten = tf.keras.layers.Flatten()
        self.left_fc1 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        self.left_fc2 = tf.keras.layers.Dense(self.class_num)
        self.Dropout = tf.keras.layers.Dropout(0.5)
        self.left_bn = tf.keras.layers.BatchNormalization(center=False, scale=False)
        self.left_bn1 = tf.keras.layers.BatchNormalization(center=False, scale=False)

        self.right_fc1 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        # self.right_fc2 = tf.keras.layers.Dense(self.class_num, activation='relu')
        self.right_fc3 = tf.keras.layers.Dense(self.class_num, activation=None)
        self.right_bn = tf.keras.layers.BatchNormalization(center=False, scale=False)
        self.right_bn1 = tf.keras.layers.BatchNormalization(center=False, scale=False)
        self.fc_task_mu = tf.keras.layers.Dense(self.class_num, activation=None)
        self.fc_task_log_sigma = tf.keras.layers.Dense(self.class_num, activation=None)

        self.fc_worker_mu = tf.keras.layers.Dense(self.class_num, activation='relu')
        self.fc_worker_rho = tf.keras.layers.Dense(self.class_num, activation='relu')

        self.p_pure = np.ones((self.class_num, ), dtype=np.float32)/self.class_num
        # self.weight = tf.Variable(tf.ones(self.answer_num,))

        # self.gnn1 = tfg.layers.GAT(self.hidden_size, activation=None, num_heads=8, attention_units=8, edge_drop_rate=0.1)
        # self.gnn1 = tfg.layers.GCN(self.hidden_size, activation=tf.nn.relu)
        # self.gnn2 = tfg.layers.GCN(self.class_num)

    def sample(self, mean, log_std):
        std = tf.math.exp(log_std)
        eps = tf.random.normal(shape=(self.task_num, self.class_num), mean=0, stddev=0.01)
        return mean + eps * std

    def left_NN(self, task_feature, training=None):

        # task_feature = self.left_bn(task_feature)
        task_feature = self.left_fc1(task_feature)
        task_feature = self.Dropout(task_feature, training)
        # task_feature = self.left_bn1(task_feature)
        cls_out = self.left_fc2(task_feature)

        return cls_out

    def right_NN(self, task_feature, answers, training=None):

        task_ids = answers[:, 0]
        worker_ids = answers[:, 1]
        label_ids = answers[:, 2]
        row = task_ids
        col = worker_ids + self.task_num
        Row = np.concatenate([row, col], axis=-1)
        Col = np.concatenate([col, row], axis=-1)
        edge_index = np.concatenate([[Row], [Col]], axis=0)

        # task_feature = self.right_bn(task_feature)
        task_feature = self.right_fc1(task_feature)
        task_feature = self.Dropout(task_feature, training)
        # task_feature = self.right_bn1(task_feature)
        # task_feature = self.right_fc3(task_feature)

        self.task_mu = self.fc_task_mu(task_feature)
        self.task_log_sigma = self.fc_task_log_sigma(task_feature)
        task_feature = self.sample(self.task_mu, self.task_log_sigma)



        # worker_mu = self.fc_worker_mu(self.worker_feature)
        # worker_rho = self.fc_worker_rho(self.worker_feature)

        # node_feature = self.right_fc3(tf.concat([task_feature, worker_sigma], axis=0))
        # gnn_out = self.gnn1([node_feature, np.array(edge_index, dtype=np.int32)], training=training)
        # gnn_out = self.gnn2([gnn_out, np.array(edge_index, dtype=np.int32)], training=training)

        # a = tf.gather(node_feature, range(self.task_num))
        # b = tf.gather(node_feature, range(self.task_num, self.task_num+self.worker_num))

        # worker_bias = tf.gather(node_feature, range(self.task_num))[:, :, None] * tf.gather(node_feature, range(self.worker_num, ))[:, None, :]
        # print(worker_bias)

        masked_task_feature = tf.gather(task_feature, row)
        masked_worker_mu = tf.gather(self.worker_mu, col - self.task_num)
        masked_worker_rho = tf.gather(self.worker_rho, col - self.task_num)

        # agg_task_feature = tf.math.unsorted_segment_mean(data=masked_task_feature, segment_ids=task_ids, num_segments=self.task_num) \
        #                    + \
        #                    tf.math.unsorted_segment_sum(data=masked_worker_feature, segment_ids=task_ids, num_segments=self.task_num)
        # agg_worker_feature = tf.math.unsorted_segment_mean(data=masked_worker_feature, segment_ids=worker_ids, num_segments=self.worker_num) \
        #                    + \
        #                    tf.math.unsorted_segment_sum(data=masked_task_feature, segment_ids=worker_ids, num_segments=self.worker_num)

        # masked_task_feature = tf.gather(task_feature, row) # p(t)

        # worker_bias = tf.sparse.from_dense(tf.matmul(task_feature, worker_feature, transpose_b=True))

        crowd_bias = masked_task_feature * tf.math.log(1+tf.math.exp(masked_worker_rho)) + masked_worker_mu # tf.nn.sigmoid(masked_worker_rho) tf.math.log(1+tf.math.exp(masked_worker_rho))
        # agg_in = masked_task_feature * tf.nn.sigmoid(masked_worker_feature) + tf.nn.sigmoid(self.worker_bias)

        # agg_out = tf.math.unsorted_segment_sum(data=crowd_bias, segment_ids=task_ids, num_segments=self.task_num)
        # agg_out = agg_task_feature * tf.nn.sigmoid(agg_worker_feature)
        # agg_out = tf.nn.softmax(agg_out, axis=-1)

        return crowd_bias, task_feature

    def call(self, input, training=False):
        task_feature, answers = input
        cls_out = self.left_NN(task_feature, training)
        crowd_bias, agg_out = self.right_NN(task_feature, answers, training)
        return cls_out, crowd_bias, agg_out

    def loss_fuction(self, cls_out, agg_out, crowd_bias, answers, gamma=0.1):

        # MIG loss
        cls_out = tf.nn.softmax(cls_out, axis=-1)
        agg_out = tf.nn.softmax(agg_out, axis=-1)
        batch_num = cls_out.shape[0]
        I = tf.cast(np.eye(batch_num), dtype=tf.float32)
        E = tf.cast(np.ones((batch_num, batch_num)), dtype=tf.float32)
        normalize_1 = batch_num
        normalize_2 = batch_num * (batch_num - 1)

        new_output = cls_out #/ self.p_pure
        m = tf.matmul(new_output, agg_out, transpose_b=True)
        noise = np.random.rand(1) * 0.0001
        m1 = tf.math.log(m * I + I * noise + E - I) # i<->i + i<->j. 此处E - I是为了让log为0，以便最大化P(i,i)互信息
        # m1 = tf.math.log(m * I + E - I)  # i<->i + i<->j. 此处E - I是为了让log为0，以便最大化P(i,i)互信息
        m2 = m * (E - I) # i<->j，最小化P(i,j)互信息
        mig_loss = -(tf.reduce_sum(tf.reduce_sum(m1)) + batch_num) / normalize_1 + tf.reduce_sum(tf.reduce_sum(m2)) / normalize_2

        # answer loss
        EC_loss = tf.nn.softmax_cross_entropy_with_logits(logits=crowd_bias, labels=tf.one_hot(indices=answers, depth=self.class_num), axis=1)
        # EC_loss = tf.reduce_sum(EC_loss)

        kl_loss = -tf.reduce_sum(1 + 2 * self.task_log_sigma - tf.square(self.task_mu) - tf.square(tf.exp(self.task_log_sigma))) * 0.5

        total_loss = mig_loss + tf.reduce_sum(EC_loss) + kl_loss

        #loss 来自 KL，与MIG相反数
        return total_loss

class CUB_model(tf.keras.Model):
    # def __init__(self, task_num, feature_size, worker_num, class_num, answer_num=None, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.task_num = task_num
    #     self.feature_size = feature_size
    #     self.worker_num = worker_num
    #     self.class_num = class_num
    #     self.answer_num = answer_num
    #     self.hidden_size = 128
    #
    #     # self.worker_feature = tf.Variable(tf.random.normal((self.worker_num, self.hidden_size), mean=0, stddev=0.01))
    #     self.worker_mu = tf.Variable(tf.random.normal((self.worker_num, self.class_num), mean=0, stddev=0.01), name='worker_mu')
    #     self.worker_rho = tf.Variable(tf.random.normal((self.worker_num, self.class_num), mean=0, stddev=0.01), name='worker_rho')
    #     # self.worker_feature = tf.one_hot(indices=range(self.worker_num), depth=self.worker_num)
    #
    #     # self.flatten = tf.keras.layers.Flatten()
    #     # self.left_fc1 = tf.keras.layers.Dense(self.hidden_size, activation='relu', name='task_lfc_1')
    #     # self.left_fc2 = tf.keras.layers.Dense(self.class_num, name='task_lfc_2')
    #     self.Dropout = tf.keras.layers.Dropout(0.5)
    #     # self.left_bn = tf.keras.layers.BatchNormalization(center=False, scale=False)
    #     # self.left_bn1 = tf.keras.layers.BatchNormalization(center=False, scale=False)
    #
    #     self.right_fc1 = tf.keras.layers.Dense(self.hidden_size, activation='relu', name='task_rfc_1')
    #     # self.right_fc2 = tf.keras.layers.Dense(self.class_num, activation='relu')
    #     # self.right_fc3 = tf.keras.layers.Dense(self.class_num, activation=None)
    #     self.right_bn = tf.keras.layers.BatchNormalization(center=False, scale=False)
    #     self.right_bn1 = tf.keras.layers.BatchNormalization(center=False, scale=False)
    #     self.fc_task_mu = tf.keras.layers.Dense(self.class_num, activation=None, name='task_rfc_mu')
    #     self.fc_task_log_sigma = tf.keras.layers.Dense(self.class_num, activation=None, name='task_rfc_log_sigma')
    #
    #     self.mu_bn = tf.keras.layers.BatchNormalization(center=False, scale=False)
    #     # self.sigma_bn = tf.keras.layers.BatchNormalization(center=False, scale=False)
    #
    #     self.right_fc2 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
    #     self.right_fc3 = tf.keras.layers.Dense(self.feature_size, activation=None)
    #
    #     # self.fc_worker_mu = tf.keras.layers.Dense(self.class_num, activation='relu')
    #     # self.fc_worker_rho = tf.keras.layers.Dense(self.class_num, activation='relu')
    #
    #     self.p_pure = np.ones((self.class_num, ), dtype=np.float32)/self.class_num
    #
    #     # self.weight = tf.Variable(tf.ones(self.answer_num,))
    #
    #     # self.gnn1 = tfg.layers.GAT(self.hidden_size, activation=None, num_heads=8, attention_units=8, edge_drop_rate=0.1)
    #     # self.gnn1 = tfg.layers.GCN(self.hidden_size, activation=tf.nn.relu)
    #     # self.gnn2 = tfg.layers.GCN(self.class_num)
    #
    # def sample(self, mean, log_std):
    #     std = tf.math.exp(log_std)
    #     eps = tf.random.normal(shape=(self.task_num, self.class_num), mean=0, stddev=0.01)
    #     return mean + eps * std
    #
    # # def left_NN(self, task_feature, training=None):
    # #
    # #     task_feature = self.left_bn(task_feature)
    # #     task_feature = self.left_fc1(task_feature)
    # #     task_feature = self.Dropout(task_feature, training)
    # #     task_feature = self.left_bn1(task_feature)
    # #     cls_out = self.left_fc2(task_feature)
    # #
    # #     return cls_out
    #
    # def right_NN(self, task_feature, training=None):
    #     # task_feature = self.right_bn(task_feature)
    #     task_feature = self.right_fc1(task_feature)
    #     task_feature = self.Dropout(task_feature, training)
    #     # task_feature = self.right_bn1(task_feature)
    #     # task_feature = self.right_fc3(task_feature)
    #
    #     task_mu = self.fc_task_mu(task_feature)
    #     task_log_sigma = self.fc_task_log_sigma(task_feature)
    #     # task_mu = self.mu_bn(task_mu)
    #     # task_log_sigma = self.sigma_bn(task_log_sigma)
    #
    #     z = self.sample(task_mu, task_log_sigma)
    #
    #     x = self.right_fc2(z)
    #     # x = self.right_bn1(x)
    #     x = self.right_fc3(x)
    #
    #     return z, x, task_mu, task_log_sigma
    #
    # def worker_NN(self, task_feature, answers):
    #
    #     task_ids = answers[:, 0]
    #     worker_ids = answers[:, 1]
    #     label_ids = answers[:, 2]
    #     row = task_ids
    #     col = worker_ids + self.task_num
    #     Row = np.concatenate([row, col], axis=-1)
    #     Col = np.concatenate([col, row], axis=-1)
    #     edge_index = np.concatenate([[Row], [Col]], axis=0)
    #
    #     masked_task_feature = tf.gather(task_feature, row)
    #     masked_worker_mu = tf.gather(self.worker_mu, col - self.task_num)
    #     masked_worker_rho = tf.gather(self.worker_rho, col - self.task_num)
    #
    #     crowd_bias = masked_task_feature * tf.math.softplus(masked_worker_rho) + masked_worker_mu # tf.nn.sigmoid(masked_worker_rho) tf.math.log(1+tf.math.exp(masked_worker_rho))
    #
    #     # self.poster = self.get_poster(crowd_bias, masked_worker_mu, masked_worker_rho)
    #
    #     return crowd_bias
    #
    # def gaussian_distribution_density(self, x, mu, sigma):
    #     return 1.0 / ((2 * np.pi) ** 0.5 * sigma) * tf.math.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + 1e-30
    #
    # def call(self, input, training=False):
    #     task_feature, answers = input
    #     # cls_out = self.left_NN(task_feature, training)
    #     sample_task_feature, recons_task_feature, task_mu, task_log_sigma = self.right_NN(task_feature, training)
    #     crowd_bias  = self.worker_NN(sample_task_feature, answers)
    #     return crowd_bias, sample_task_feature, recons_task_feature, task_mu, task_log_sigma
    #
    # def get_poster(self, crowd_bias, masked_worker_mu, masked_worker_rho):
    #     Qwtheta = tf.math.log(self.gaussian_distribution_density(crowd_bias, masked_worker_mu, masked_worker_rho))
    #     return tf.reduce_mean(Qwtheta)
    #
    # def MIG_loss(self, cls_out, agg_out):
    #     # MIG loss
    #     cls_out = tf.nn.softmax(cls_out, axis=-1)
    #     agg_out = tf.nn.softmax(agg_out, axis=-1)
    #     batch_num = cls_out.shape[0]
    #     I = tf.cast(np.eye(batch_num), dtype=tf.float32)
    #     E = tf.cast(np.ones((batch_num, batch_num)), dtype=tf.float32)
    #     normalize_1 = batch_num
    #     normalize_2 = batch_num * (batch_num - 1)
    #
    #     new_output = cls_out  # / self.p_pure
    #     m = tf.matmul(new_output, agg_out, transpose_b=True)
    #     noise = np.random.rand(1) * 0.0001
    #     m1 = tf.math.log(m * I + I * noise + E - I)  # i<->i + i<->j. 此处E - I是为了让log为0，以便最大化P(i,i)互信息
    #     # m1 = tf.math.log(m * I + E - I)  # i<->i + i<->j. 此处E - I是为了让log为0，以便最大化P(i,i)互信息
    #     m2 = m * (E - I)  # i<->j，最小化P(i,j)互信息
    #     mig_loss = -(tf.reduce_sum(tf.reduce_sum(m1)) + batch_num) / normalize_1 + tf.reduce_sum(tf.reduce_sum(m2)) / normalize_2
    #     return mig_loss
    #
    # def CE_loss(self, crowd_bias, answers):
    #     return tf.nn.softmax_cross_entropy_with_logits(logits=crowd_bias,
    #                                                     labels=tf.one_hot(indices=answers, depth=self.class_num),
    #                                                     axis=1)
    #
    # def KL_loss(self, task_mu, task_log_sigma):
    #     return -tf.reduce_sum(1 + 2 * task_log_sigma - tf.square(task_mu) - tf.square(tf.exp(task_log_sigma))) * 0.5
    #
    # def loss_fuction(self, task_feature, recons_task_feature, crowd_bias, answers, task_mu, task_log_sigma, gamma=0.1):
    #
    #     #recons_loss
    #     mig_loss = self.MIG_loss(task_feature, recons_task_feature)
    #
    #     # recons_loss
    #     # recons_loss = tf.reduce_mean(tf.sqrt(tf.square(task_feature - recons_task_feature))) * 0
    #     # print(recons_loss)
    #
    #     # KL_loss
    #     kl_loss = self.KL_loss(task_mu, task_log_sigma)
    #
    #     # CE_loss
    #     EC_loss = self.CE_loss(crowd_bias, answers)
    #
    #     total_loss = mig_loss + kl_loss + tf.reduce_sum(EC_loss) #- self.poster
    #
    #     #loss 来自 KL，与MIG相反数
    #     return total_loss
    #     # return tf.reduce_sum(EC_loss), mig_loss, kl_loss
    def __init__(self, task_num, feature_size, worker_num, class_num, answer_num=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_num = task_num
        self.feature_size = feature_size
        self.worker_num = worker_num
        self.class_num = class_num
        self.answer_num = answer_num
        self.hidden_size = 128

        # self.worker_feature = tf.Variable(tf.random.normal((self.worker_num, self.hidden_size), mean=0, stddev=0.01))
        self.worker_mu = tf.Variable(tf.random.normal((self.worker_num, self.class_num), mean=0, stddev=0.01), name='worker_mu')
        self.worker_rho = tf.Variable(tf.random.normal((self.worker_num, self.class_num), mean=0, stddev=0.01), name='worker_rho')
        # self.worker_feature = tf.one_hot(indices=range(self.worker_num), depth=self.worker_num)

        # self.flatten = tf.keras.layers.Flatten()
        # self.left_fc1 = tf.keras.layers.Dense(self.hidden_size, activation='relu', name='task_lfc_1')
        # self.left_fc2 = tf.keras.layers.Dense(self.class_num, name='task_lfc_2')
        self.Dropout = tf.keras.layers.Dropout(0.5)
        # self.left_bn = tf.keras.layers.BatchNormalization(center=False, scale=False)
        # self.left_bn1 = tf.keras.layers.BatchNormalization(center=False, scale=False)

        self.right_fc1 = tf.keras.layers.Dense(self.hidden_size, activation='relu', name='task_rfc_1')
        # self.right_fc2 = tf.keras.layers.Dense(self.class_num, activation='relu')
        # self.right_fc3 = tf.keras.layers.Dense(self.class_num, activation=None)
        # self.right_bn = tf.keras.layers.BatchNormalization(center=False, scale=False)
        # self.right_bn1 = tf.keras.layers.BatchNormalization(center=False, scale=False)
        self.fc_task_mu = tf.keras.layers.Dense(self.class_num, activation=None, name='task_rfc_mu')
        self.fc_task_log_sigma = tf.keras.layers.Dense(self.class_num, activation=None, name='task_rfc_log_sigma')

        self.mu_bn = tf.keras.layers.BatchNormalization(center=False, scale=False)
        # self.sigma_bn = tf.keras.layers.BatchNormalization(center=False, scale=False)

        self.right_fc2 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        self.right_fc3 = tf.keras.layers.Dense(self.feature_size, activation=None)

        # self.fc_worker_mu = tf.keras.layers.Dense(self.class_num, activation='relu')
        # self.fc_worker_rho = tf.keras.layers.Dense(self.class_num, activation='relu')

        self.p_pure = np.ones((self.class_num, ), dtype=np.float32)/self.class_num

        # self.weight = tf.Variable(tf.ones(self.answer_num,))

        # self.gnn1 = tfg.layers.GAT(self.hidden_size, activation=None, num_heads=8, attention_units=8, edge_drop_rate=0.1)
        # self.gnn1 = tfg.layers.GCN(self.hidden_size, activation=tf.nn.relu)
        # self.gnn2 = tfg.layers.GCN(self.class_num)

    def sample(self, mean, log_std):
        std = tf.math.exp(log_std)
        eps = tf.random.normal(shape=(self.task_num, self.class_num), mean=0, stddev=0.01)
        return mean + eps * std

    # def left_NN(self, task_feature, training=None):
    #
    #     task_feature = self.left_bn(task_feature)
    #     task_feature = self.left_fc1(task_feature)
    #     task_feature = self.Dropout(task_feature, training)
    #     task_feature = self.left_bn1(task_feature)
    #     cls_out = self.left_fc2(task_feature)
    #
    #     return cls_out

    def right_NN(self, task_feature, training=None):
        # task_feature = self.right_bn(task_feature)
        task_feature = self.right_fc1(task_feature)
        task_feature = self.Dropout(task_feature, training)
        # task_feature = self.right_bn1(task_feature)
        # task_feature = self.right_fc3(task_feature)

        self.task_mu = self.fc_task_mu(task_feature)
        self.task_log_sigma = self.fc_task_log_sigma(task_feature)
        # task_mu = self.mu_bn(task_mu)
        # task_log_sigma = self.sigma_bn(task_log_sigma)
        z = self.sample(self.task_mu, self.task_log_sigma)

        x = self.right_fc2(z)
        # x = self.right_bn1(x)
        x = self.right_fc3(x)

        return z, x, self.task_mu, self.task_log_sigma

    def worker_NN(self, task_feature, answers):

        task_ids = answers[:, 0]
        worker_ids = answers[:, 1]
        label_ids = answers[:, 2]
        row = task_ids
        col = worker_ids + self.task_num
        Row = np.concatenate([row, col], axis=-1)
        Col = np.concatenate([col, row], axis=-1)
        edge_index = np.concatenate([[Row], [Col]], axis=0)

        masked_task_feature = tf.gather(task_feature, row)
        self.masked_task_mu = tf.gather(self.task_mu, row)
        self.masked_task_sigma = tf.math.exp(tf.gather(self.task_log_sigma, row))
        self.masked_worker_mu = tf.gather(self.worker_mu, col - self.task_num)
        self.masked_worker_rho = tf.gather(self.worker_rho, col - self.task_num)


        crowd_bias = masked_task_feature * tf.nn.softplus(self.masked_worker_rho) + self.masked_worker_mu # tf.nn.sigmoid(masked_worker_rho) tf.math.log(1+tf.math.exp(masked_worker_rho))

        # self.poster_prior = self.kl_Qwtheta_Pw(worker_mu_kernel, self.workerMu_mu, self.workerMu_rho) \
        #                      + \
        #                      self.kl_Qwtheta_Pw(worker_sigma_kernel, self.workerRho_mu, self.workerRho_rho)

        return crowd_bias

    def gaussian_distribution_density(self, x, mu, sigma):
        return 1.0 / ((2 * np.pi) ** 0.5 * sigma) * tf.math.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + 1e-30

    def call(self, input, training=False):
        task_feature, answers = input
        # cls_out = self.left_NN(task_feature, training)
        sample_task_feature, recons_task_feature, task_mu, task_log_sigma = self.right_NN(task_feature, training)
        crowd_bias  = self.worker_NN(sample_task_feature, answers)
        return crowd_bias, sample_task_feature, recons_task_feature, task_mu, task_log_sigma

    def kl_Qwtheta_Pw(self, w, mu, sigma):
        Qwtheta = tf.math.log(self.gaussian_distribution_density(w, mu, sigma))  # poster
        Pw = tf.math.log(0.5 * self.gaussian_distribution_density(w, 0.0, 1.5)
                         +
                         0.5 * self.gaussian_distribution_density(w, 0.0, 0.1))  # prior
        return tf.math.reduce_mean(Qwtheta - Pw)

    def MIG_loss(self, cls_out, agg_out):
        # MIG loss
        cls_out = tf.nn.softmax(cls_out, axis=-1)
        agg_out = tf.nn.softmax(agg_out, axis=-1)
        batch_num = cls_out.shape[0]
        I = tf.cast(np.eye(batch_num), dtype=tf.float32)
        E = tf.cast(np.ones((batch_num, batch_num)), dtype=tf.float32)
        normalize_1 = batch_num
        normalize_2 = batch_num * (batch_num - 1)

        new_output = cls_out  # / self.p_pure
        m = tf.matmul(new_output, agg_out, transpose_b=True)
        noise = np.random.rand(1) * 0.0001
        m1 = tf.math.log(m * I + I * noise + E - I)  # i<->i + i<->j. 此处E - I是为了让log为0，以便最大化P(i,i)互信息
        # m1 = tf.math.log(m * I + E - I)  # i<->i + i<->j. 此处E - I是为了让log为0，以便最大化P(i,i)互信息
        m2 = m * (E - I)  # i<->j，最小化P(i,j)互信息
        mig_loss = -(tf.reduce_sum(tf.reduce_sum(m1)) + batch_num) / normalize_1 + tf.reduce_sum(tf.reduce_sum(m2)) / normalize_2
        return mig_loss

    def CE_loss(self, crowd_bias, answers):
        return tf.nn.softmax_cross_entropy_with_logits(logits=crowd_bias,
                                                        labels=tf.one_hot(indices=answers, depth=self.class_num),
                                                        axis=1)

    def KL_loss(self, task_mu, task_log_sigma):
        return -tf.reduce_sum(1 + 2 * task_log_sigma - tf.square(task_mu) - tf.square(tf.exp(task_log_sigma))) * 0.5

    def loss_fuction(self, task_feature, recons_task_feature, crowd_bias, answers, task_mu, task_log_sigma, gamma=0.1):

        # poster

        poster = self.kl_Qwtheta_Pw(self.masked_worker_mu, self.masked_task_mu, self.masked_task_sigma) \
                 + \
                 self.kl_Qwtheta_Pw(self.masked_worker_rho, self.masked_task_mu, self.masked_task_sigma)

        #recons_loss
        mig_loss = self.MIG_loss(task_feature, recons_task_feature)

        # recons_loss
        # recons_loss = tf.reduce_mean(tf.sqrt(tf.square(task_feature - recons_task_feature))) * 0
        # print(recons_loss)

        # KL_loss
        kl_loss = self.KL_loss(task_mu, task_log_sigma) #+ self.KL_loss(self.worker_mu, self.worker_rho)

        # CE_loss
        EC_loss = self.CE_loss(crowd_bias, answers)

        total_loss = mig_loss + kl_loss + tf.reduce_sum(EC_loss) - poster

        #loss 来自 KL，与MIG相反数
        return total_loss
        # return tf.reduce_sum(EC_loss), mig_loss, kl_loss

print()

def run_LableMe():
    # batch_size = 10000
    # best_acc = 0
    #
    # task_feature, answers, answer_matrix, answers_bin_missings, truths = load_LabelMe_dataset()
    # task_num, worker_num, class_num = answers_bin_missings.shape
    # feature_size = task_feature.shape[1]
    # # answer_num = answers.shape[0]
    # trainer = LableMe_model(task_num, feature_size, worker_num, class_num)
    #
    # # shuffle_train_data, shuffle_answers_bin_missings = shuffle_data(task_feature, answers_bin_missings, batch_size)
    # learning_rate = 1e-3
    # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    #
    # if task_feature.shape[0] % batch_size == 0:
    #     steps = int(task_feature.shape[0] / batch_size)
    # else:
    #     steps = int((task_feature.shape[0] / batch_size) + 1)
    # for epoch in range(1000):
    #     print('epoch:', epoch)
    #     for step in range(steps):
    #         # print('step:', step)
    #         # batch_train_data = shuffle_train_data[step * batch_size:(step + 1) * batch_size, :]
    #         # batch_answers_bin_missings = shuffle_answers_bin_missings[step * batch_size:(step + 1) * batch_size, :]
    #         with tf.GradientTape() as tape:
    #             cls_out, crowd_bias, agg_out = trainer([task_feature, answers], training=True)
    #
    #             loss = trainer.loss_fuction(cls_out, agg_out, crowd_bias, answers[:, -1])
    #             # print(loss)
    #
    #             vars = tape.watched_variables()
    #             grads = tape.gradient(loss, vars)
    #             optimizer.apply_gradients(zip(grads, vars))
    #
    #     cls_out, _, agg_out = trainer([task_feature, answers], training=False)
    #
    #     flag = tf.compat.v1.to_int32(tf.equal(tf.argmax(cls_out, axis=-1), truths))
    #     acc = tf.reduce_sum(flag) / truths.shape[0]
    #
    #     flag1 = tf.compat.v1.to_int32(tf.equal(tf.argmax(agg_out, axis=-1), truths))
    #     acc1 = tf.reduce_sum(flag1) / truths.shape[0]
    #
    #     if acc1 > best_acc:
    #         best_acc = acc1
    #     # print('Acc:', tf.reduce_sum(flag) / truths.shape[0])
    #     print("step = {}\tloss = {}\tbest_accuracy = {}\tacc = {}\tacc1 = {}".format(step, loss, best_acc, acc, acc1))
    #     # print('.................')
    batch_size = 10000
    best_acc = 0

    task_feature, answers, answer_matrix, answers_bin_missings, truths = load_LabelMe_dataset()
    task_num, worker_num, class_num = answers_bin_missings.shape
    feature_size = task_feature.shape[1]
    answer_num = answers.shape[0]
    trainer = LableMe_model(task_num, feature_size, worker_num, class_num, answer_num)

    # shuffle_train_data, shuffle_answers_bin_missings = shuffle_data(task_feature, answers_bin_missings, batch_size)
    learning_rate = 4e-3
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    if task_feature.shape[0] % batch_size == 0:
        steps = int(task_feature.shape[0] / batch_size)
    else:
        steps = int((task_feature.shape[0] / batch_size) + 1)
    for epoch in range(1000):
        print('epoch:', epoch)
        for step in range(steps):
            # print('step:', step)
            # batch_train_data = shuffle_train_data[step * batch_size:(step + 1) * batch_size, :]
            # batch_answers_bin_missings = shuffle_answers_bin_missings[step * batch_size:(step + 1) * batch_size, :]

            with tf.GradientTape() as tape:
                crowd_bias, sample_task_feature, recons_task_feature, task_mu, task_log_sigma = trainer(
                    [task_feature, answers], training=True)

                loss = trainer.loss_fuction(task_feature, recons_task_feature, crowd_bias, answers[:, -1], task_mu,
                                            task_log_sigma)

                vars = tape.watched_variables()
                grads = tape.gradient(loss, vars)
                optimizer.apply_gradients(zip(grads, vars))

        crowd_bias, sample_task_feature, recons_task_feature, task_mu, task_log_sigma = trainer([task_feature, answers],
                                                                                                training=False)

        # flag = tf.compat.v1.to_int32(tf.equal(tf.argmax(cls_out, axis=-1), truths))
        # acc = tf.reduce_sum(flag) / truths.shape[0]

        flag1 = tf.compat.v1.to_int32(tf.equal(tf.argmax(sample_task_feature, axis=-1), truths))
        acc1 = tf.reduce_sum(flag1) / truths.shape[0]

        if acc1 > best_acc:
            best_acc = acc1
        # print('Acc:', tf.reduce_sum(flag) / truths.shape[0])
        print("step = {}\tloss = {}\tbest_accuracy = {}\tacc1 = {}".format(step, loss, best_acc, acc1))
        # print('.................')
def run_Music():
    batch_size = 700
    best_acc = 0
    # best_auc = 0
    best_macro = 0
    # best_micro = 0
    best_acc_hard = 0
    best_macro_hard = 0

    task_feature, answers, answer_matrix, answers_bin_missings, truths, hard_example = load_Music_dataset()
    task_num, worker_num, class_num = answers_bin_missings.shape
    feature_size = task_feature.shape[1]
    answer_num = answers.shape[0]
    trainer = Music_model(task_num, feature_size, worker_num, class_num, answer_num)

    shuffle_train_data, shuffle_answers_bin_missings = shuffle_data(task_feature, answers_bin_missings, batch_size)
    learning_rate = 4e-3
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    if task_feature.shape[0] % batch_size == 0:
        steps = int(task_feature.shape[0] / batch_size)
    else:
        steps = int((task_feature.shape[0] / batch_size) + 1)
    for epoch in range(1000):
        print('epoch:', epoch)
        for step in range(steps):
            # print('step:', step)
            # batch_train_data = shuffle_train_data[step * batch_size:(step + 1) * batch_size, :]
            # batch_answers_bin_missings = shuffle_answers_bin_missings[step * batch_size:(step + 1) * batch_size, :]

            with tf.GradientTape() as tape:
                crowd_bias, sample_task_feature, recons_task_feature, task_mu, task_log_sigma = trainer([task_feature, answers], training=True)

                loss = trainer.loss_fuction(task_feature, recons_task_feature, crowd_bias, answers[:, -1], task_mu, task_log_sigma)

                vars = tape.watched_variables()
                grads = tape.gradient(loss, vars)
                optimizer.apply_gradients(zip(grads, vars))

        crowd_bias, sample_task_feature, recons_task_feature, task_mu, task_log_sigma = trainer([task_feature, answers], training=False)

        flag = tf.compat.v1.to_int32(tf.equal(tf.argmax(sample_task_feature, axis=-1), truths))
        acc = tf.reduce_sum(flag) / truths.shape[0]

        flag_hard = tf.compat.v1.to_int32(tf.equal(tf.argmax(tf.gather(sample_task_feature, hard_example[:, 0]), axis=-1), hard_example[:, -1]))
        acc_hard = tf.reduce_sum(flag_hard) / hard_example.shape[0]
        # print(hard_example[:, 0], hard_example[:, -1])

        # if acc1>0.83:
        #     np.save('{}_task_mu.npy'.format(acc1), task_mu)
        #     np.save('{}_task_log_sigma.npy'.format(acc1), task_log_sigma)
        #     np.save('{}_worker_mu.npy'.format(acc1), trainer.worker_mu)
        #     np.save('{}_worker_rho.npy'.format(acc1), trainer.worker_rho)

        # m = tf.keras.metrics.AUC().update_state([1,0,0,1], [0, 0.5, 0.5, 0.9])

        # m = tf.keras.metrics.AUC(num_thresholds=200)
        # m.update_state(tf.one_hot(tf.argmax(sample_task_feature, axis=-1), class_num), tf.one_hot(truths, class_num))

        macro = f1_score(y_true=truths, y_pred=tf.argmax(sample_task_feature, axis=-1), average='macro')
        # micro = f1_score(y_true=truths, y_pred=tf.argmax(sample_task_feature, axis=-1), average='micro')
        macro_hard = f1_score(y_true=hard_example[:, -1], y_pred=tf.argmax(tf.gather(sample_task_feature, hard_example[:, 0]), axis=-1), average='macro')

        if acc > best_acc:
            best_acc = acc
        if acc_hard > best_acc_hard:
            best_acc_hard = acc_hard
        # if m.result().numpy() > best_auc:
        #     best_auc = m.result().numpy()
        if macro > best_macro:
            best_macro = macro
        if macro_hard > best_macro_hard:
            best_macro_hard = macro_hard
        # if micro > best_micro:
        #     best_micro = micro
        # print('Acc:', tf.reduce_sum(flag) / truths.shape[0])
        print("step = {}\tloss = {}".format(epoch, loss))
        print("best_acc = {}\tacc1 = {}".format(best_acc, acc))
        # print("best_auc = {}\tauc = {}".format(best_auc, m.result().numpy()))
        print("best_macro = {}\tmacro = {}".format(best_macro, macro))
        # print("best_micro = {}\tmicro = {}".format(best_micro, micro))
        print("best_acc_hard = {}\tacc_hard = {}".format(best_acc_hard, acc_hard))
        print("best_macro_hard = {}\tmacro_hard = {}".format(best_macro_hard, macro_hard))
        # print('.................')
    print("hard example number:", hard_example.shape[0])
def run_SP():
    batch_size = 5000
    best_acc = 0
    best_auc = 0
    best_macro = 0

    task_feature, answers, answer_matrix, answers_bin_missings, truths = load_SP_dataset()
    task_num, worker_num, class_num = answers_bin_missings.shape
    feature_size = task_feature.shape[1]
    answer_num = answers.shape[0]
    trainer = SP_model(task_num, feature_size, worker_num, class_num, answer_num)

    shuffle_train_data, shuffle_answers_bin_missings = shuffle_data(task_feature, answers_bin_missings, batch_size)
    learning_rate = 5e-3
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    if task_feature.shape[0] % batch_size == 0:
        steps = int(task_feature.shape[0] / batch_size)
    else:
        steps = int((task_feature.shape[0] / batch_size) + 1)
    for epoch in range(1000):
        print('epoch:', epoch)
        for step in range(steps):
            # print('step:', step)
            # batch_train_data = shuffle_train_data[step * batch_size:(step + 1) * batch_size, :]
            # batch_answers_bin_missings = shuffle_answers_bin_missings[step * batch_size:(step + 1) * batch_size, :]

            with tf.GradientTape() as tape:
                crowd_bias, sample_task_feature, recons_task_feature, task_mu, task_log_sigma = trainer([task_feature, answers], training=True)

                loss = trainer.loss_fuction(task_feature, recons_task_feature, crowd_bias, answers[:, -1], task_mu, task_log_sigma)

                vars = tape.watched_variables()
                grads = tape.gradient(loss, vars)
                optimizer.apply_gradients(zip(grads, vars))

        crowd_bias, sample_task_feature, recons_task_feature, task_mu, task_log_sigma = trainer([task_feature, answers], training=False)

        # flag = tf.compat.v1.to_int32(tf.equal(tf.argmax(cls_out, axis=-1), truths))
        # acc = tf.reduce_sum(flag) / truths.shape[0]

        flag1 = tf.compat.v1.to_int32(tf.equal(tf.argmax(sample_task_feature, axis=-1), truths))
        acc1 = tf.reduce_sum(flag1) / truths.shape[0]

        # m = tf.keras.metrics.AUC().update_state([1,0,0,1], [0, 0.5, 0.5, 0.9])

        m = tf.keras.metrics.AUC(num_thresholds=200)
        m.update_state(tf.one_hot(tf.argmax(sample_task_feature, axis=-1), class_num), tf.one_hot(truths, class_num))

        macro = f1_score(y_true=truths, y_pred=tf.argmax(sample_task_feature, axis=-1), average='macro')
        # micro = f1_score(y_true=truths, y_pred=tf.argmax(sample_task_feature, axis=-1), average='micro')

        if acc1 > best_acc:
            best_acc = acc1
        if m.result().numpy() > best_auc:
            best_auc = m.result().numpy()
        if macro > best_macro:
            best_macro = macro
        # if micro > best_micro:
        #     best_micro = micro
        # print('Acc:', tf.reduce_sum(flag) / truths.shape[0])
        print("step = {}\tloss = {}".format(epoch, loss))
        print("best_acc = {}\tacc1 = {}".format(best_acc, acc1))
        print("best_auc = {}\tauc = {}".format(best_auc, m.result().numpy()))
        print("best_macro = {}\tmacro = {}".format(best_macro, macro))
        # print("best_micro = {}\tmicro = {}".format(best_micro, micro))
        # print('.................')
def run_BCD():
    batch_size = 1000
    best_acc = 0
    best_auc = 0
    best_macro = 0
    best_acc_hard = 0
    best_macro_hard = 0

    task_feature, answers, answer_matrix, answers_bin_missings, truths, hard_example = load_BCD_dataset()
    task_num, worker_num, class_num = answers_bin_missings.shape
    feature_size = task_feature.shape[1]
    answer_num = answers.shape[0]
    trainer = BCD_model(task_num, feature_size, worker_num, class_num, answer_num)

    shuffle_train_data, shuffle_answers_bin_missings = shuffle_data(task_feature, answers_bin_missings, batch_size)
    learning_rate = 5e-3
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    if task_feature.shape[0] % batch_size == 0:
        steps = int(task_feature.shape[0] / batch_size)
    else:
        steps = int((task_feature.shape[0] / batch_size) + 1)
    for epoch in range(100):
        print('epoch:', epoch)
        for step in range(steps):
            # print('step:', step)
            # batch_train_data = shuffle_train_data[step * batch_size:(step + 1) * batch_size, :]
            # batch_answers_bin_missings = shuffle_answers_bin_missings[step * batch_size:(step + 1) * batch_size, :]

            with tf.GradientTape() as tape:
                crowd_bias, sample_task_feature, recons_task_feature, task_mu, task_log_sigma = trainer([task_feature, answers], training=True)

                loss = trainer.loss_fuction(task_feature, recons_task_feature, crowd_bias, answers[:, -1], task_mu, task_log_sigma)

                vars = tape.watched_variables()
                grads = tape.gradient(loss, vars)
                optimizer.apply_gradients(zip(grads, vars))

        crowd_bias, sample_task_feature, recons_task_feature, task_mu, task_log_sigma = trainer([task_feature, answers], training=False)

        # flag = tf.compat.v1.to_int32(tf.equal(tf.argmax(cls_out, axis=-1), truths))
        # acc = tf.reduce_sum(flag) / truths.shape[0]

        flag = tf.compat.v1.to_int32(tf.equal(tf.argmax(sample_task_feature, axis=-1), truths))
        acc = tf.reduce_sum(flag) / truths.shape[0]

        flag_hard = tf.compat.v1.to_int32(
            tf.equal(tf.argmax(tf.gather(sample_task_feature, hard_example[:, 0]), axis=-1), hard_example[:, -1]))
        acc_hard = tf.reduce_sum(flag_hard) / hard_example.shape[0]
        # print(hard_example[:, 0], hard_example[:, -1])

        # if acc1>0.83:
        #     np.save('{}_task_mu.npy'.format(acc1), task_mu)
        #     np.save('{}_task_log_sigma.npy'.format(acc1), task_log_sigma)
        #     np.save('{}_worker_mu.npy'.format(acc1), trainer.worker_mu)
        #     np.save('{}_worker_rho.npy'.format(acc1), trainer.worker_rho)

        # m = tf.keras.metrics.AUC().update_state([1,0,0,1], [0, 0.5, 0.5, 0.9])

        # m = tf.keras.metrics.AUC(num_thresholds=200)
        # m.update_state(tf.one_hot(tf.argmax(sample_task_feature, axis=-1), class_num), tf.one_hot(truths, class_num))

        macro = f1_score(y_true=truths, y_pred=tf.argmax(sample_task_feature, axis=-1), average='macro')
        # micro = f1_score(y_true=truths, y_pred=tf.argmax(sample_task_feature, axis=-1), average='micro')
        macro_hard = f1_score(y_true=hard_example[:, -1], y_pred=tf.argmax(tf.gather(sample_task_feature, hard_example[:, 0]), axis=-1), average='macro')

        if acc > best_acc:
            best_acc = acc
        if acc_hard > best_acc_hard:
            best_acc_hard = acc_hard
        # if m.result().numpy() > best_auc:
        #     best_auc = m.result().numpy()
        if macro > best_macro:
            best_macro = macro
        if macro_hard > best_macro_hard:
            best_macro_hard = macro_hard
        # if micro > best_micro:
        #     best_micro = micro
        # print('Acc:', tf.reduce_sum(flag) / truths.shape[0])
        print("step = {}\tloss = {}".format(epoch, loss))
        print("best_acc = {}\tacc1 = {}".format(best_acc, acc))
        # print("best_auc = {}\tauc = {}".format(best_auc, m.result().numpy()))
        print("best_macro = {}\tmacro = {}".format(best_macro, macro))
        # print("best_micro = {}\tmicro = {}".format(best_micro, micro))
        print("best_acc_hard = {}\tacc_hard = {}".format(best_acc_hard, acc_hard))
        print("best_macro_hard = {}\tmacro_hard = {}".format(best_macro_hard, macro_hard))
        # print('.................')
    print("hard example number:", hard_example.shape[0])
def run_Reuters():
    batch_size = 1786
    best_acc = 0
    best_auc = 0
    best_macro = 0
    best_acc_hard = 0
    best_macro_hard = 0

    task_feature, answers, answer_matrix, answers_bin_missings, truths, hard_example = load_Reuters_dataset()
    task_num, worker_num, class_num = answers_bin_missings.shape
    feature_size = task_feature.shape[1]
    answer_num = answers.shape[0]
    trainer = SP_model(task_num, feature_size, worker_num, class_num, answer_num)

    shuffle_train_data, shuffle_answers_bin_missings = shuffle_data(task_feature, answers_bin_missings, batch_size)
    learning_rate = 5e-3
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    if task_feature.shape[0] % batch_size == 0:
        steps = int(task_feature.shape[0] / batch_size)
    else:
        steps = int((task_feature.shape[0] / batch_size) + 1)
    for epoch in range(100):
        print('epoch:', epoch)
        for step in range(steps):
            # print('step:', step)
            # batch_train_data = shuffle_train_data[step * batch_size:(step + 1) * batch_size, :]
            # batch_answers_bin_missings = shuffle_answers_bin_missings[step * batch_size:(step + 1) * batch_size, :]

            with tf.GradientTape() as tape:
                crowd_bias, sample_task_feature, recons_task_feature, task_mu, task_log_sigma = trainer([task_feature, answers], training=True)

                loss = trainer.loss_fuction(task_feature, recons_task_feature, crowd_bias, answers[:, -1], task_mu,
                                            task_log_sigma)

                vars = tape.watched_variables()
                grads = tape.gradient(loss, vars)
                optimizer.apply_gradients(zip(grads, vars))

        crowd_bias, sample_task_feature, recons_task_feature, task_mu, task_log_sigma = trainer([task_feature, answers], training=False)

        # flag = tf.compat.v1.to_int32(tf.equal(tf.argmax(cls_out, axis=-1), truths))
        # acc = tf.reduce_sum(flag) / truths.shape[0]

        flag = tf.compat.v1.to_int32(tf.equal(tf.argmax(sample_task_feature, axis=-1), truths))
        acc = tf.reduce_sum(flag) / truths.shape[0]

        flag_hard = tf.compat.v1.to_int32(
            tf.equal(tf.argmax(tf.gather(sample_task_feature, hard_example[:, 0]), axis=-1), hard_example[:, -1]))
        acc_hard = tf.reduce_sum(flag_hard) / hard_example.shape[0]
        # print(hard_example[:, 0], hard_example[:, -1])

        # if acc1>0.83:
        #     np.save('{}_task_mu.npy'.format(acc1), task_mu)
        #     np.save('{}_task_log_sigma.npy'.format(acc1), task_log_sigma)
        #     np.save('{}_worker_mu.npy'.format(acc1), trainer.worker_mu)
        #     np.save('{}_worker_rho.npy'.format(acc1), trainer.worker_rho)

        # m = tf.keras.metrics.AUC().update_state([1,0,0,1], [0, 0.5, 0.5, 0.9])

        # m = tf.keras.metrics.AUC(num_thresholds=200)
        # m.update_state(tf.one_hot(tf.argmax(sample_task_feature, axis=-1), class_num), tf.one_hot(truths, class_num))

        macro = f1_score(y_true=truths, y_pred=tf.argmax(sample_task_feature, axis=-1), average='macro')
        # micro = f1_score(y_true=truths, y_pred=tf.argmax(sample_task_feature, axis=-1), average='micro')
        macro_hard = f1_score(y_true=hard_example[:, -1], y_pred=tf.argmax(tf.gather(sample_task_feature, hard_example[:, 0]), axis=-1), average='macro')

        if acc > best_acc:
            best_acc = acc
        if acc_hard > best_acc_hard:
            best_acc_hard = acc_hard
        # if m.result().numpy() > best_auc:
        #     best_auc = m.result().numpy()
        if macro > best_macro:
            best_macro = macro
        if macro_hard > best_macro_hard:
            best_macro_hard = macro_hard
        # if micro > best_micro:
        #     best_micro = micro
        # print('Acc:', tf.reduce_sum(flag) / truths.shape[0])
        print("step = {}\tloss = {}".format(epoch, loss))
        print("best_acc = {}\tacc1 = {}".format(best_acc, acc))
        # print("best_auc = {}\tauc = {}".format(best_auc, m.result().numpy()))
        print("best_macro = {}\tmacro = {}".format(best_macro, macro))
        # print("best_micro = {}\tmicro = {}".format(best_micro, micro))
        print("best_acc_hard = {}\tacc_hard = {}".format(best_acc_hard, acc_hard))
        print("best_macro_hard = {}\tmacro_hard = {}".format(best_macro_hard, macro_hard))
        # print('.................')
    print("hard example number:", hard_example.shape[0])
def run_CUB():
    batch_size = 6033
    best_acc = 0
    best_auc = 0
    best_macro = 0
    best_acc_hard = 0
    best_macro_hard = 0

    # load_Bill_dataset, load_Head_dataset, load_Shape_dataset, load_Forehead_dataset, load_Throat_dataset, load_Underpart_dataset, load_Breast_dataset

    task_feature, answers, answer_matrix, answers_bin_missings, truths, hard_example = load_Underpart_dataset() # load_Underpart_dataset
    task_num, worker_num, class_num = answers_bin_missings.shape
    feature_size = task_feature.shape[1]
    answer_num = answers.shape[0]
    trainer = SP_model(task_num, feature_size, worker_num, class_num, answer_num)

    shuffle_train_data, shuffle_answers_bin_missings = shuffle_data(task_feature, answers_bin_missings, batch_size)
    learning_rate = 5e-3
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    if task_feature.shape[0] % batch_size == 0:
        steps = int(task_feature.shape[0] / batch_size)
    else:
        steps = int((task_feature.shape[0] / batch_size) + 1)
    for epoch in range(100):
        print('epoch:', epoch)
        for step in range(steps):
            # print('step:', step)
            # batch_train_data = shuffle_train_data[step * batch_size:(step + 1) * batch_size, :]
            # batch_answers_bin_missings = shuffle_answers_bin_missings[step * batch_size:(step + 1) * batch_size, :]

            with tf.GradientTape() as tape:
                crowd_bias, sample_task_feature, recons_task_feature, task_mu, task_log_sigma = trainer([task_feature, answers], training=True)

                loss = trainer.loss_fuction(task_feature, recons_task_feature, crowd_bias, answers[:, -1], task_mu, task_log_sigma)

                vars = tape.watched_variables()
                grads = tape.gradient(loss, vars)
                optimizer.apply_gradients(zip(grads, vars))

        crowd_bias, sample_task_feature, recons_task_feature, task_mu, task_log_sigma = trainer([task_feature, answers], training=False)

        flag = tf.compat.v1.to_int32(tf.equal(tf.argmax(sample_task_feature, axis=-1), truths))
        acc = tf.reduce_sum(flag) / truths.shape[0]

        flag_hard = tf.compat.v1.to_int32(
            tf.equal(tf.argmax(tf.gather(sample_task_feature, hard_example[:, 0]), axis=-1), hard_example[:, -1]))
        acc_hard = tf.reduce_sum(flag_hard) / hard_example.shape[0]
        # print(hard_example[:, 0], hard_example[:, -1])

        # if acc1>0.83:
        #     np.save('{}_task_mu.npy'.format(acc1), task_mu)
        #     np.save('{}_task_log_sigma.npy'.format(acc1), task_log_sigma)
        #     np.save('{}_worker_mu.npy'.format(acc1), trainer.worker_mu)
        #     np.save('{}_worker_rho.npy'.format(acc1), trainer.worker_rho)

        # m = tf.keras.metrics.AUC().update_state([1,0,0,1], [0, 0.5, 0.5, 0.9])

        # m = tf.keras.metrics.AUC(num_thresholds=200)
        # m.update_state(tf.one_hot(tf.argmax(sample_task_feature, axis=-1), class_num), tf.one_hot(truths, class_num))

        macro = f1_score(y_true=truths, y_pred=tf.argmax(sample_task_feature, axis=-1), average='macro')
        # micro = f1_score(y_true=truths, y_pred=tf.argmax(sample_task_feature, axis=-1), average='micro')
        macro_hard = f1_score(y_true=hard_example[:, -1], y_pred=tf.argmax(tf.gather(sample_task_feature, hard_example[:, 0]), axis=-1), average='macro')

        if acc > best_acc:
            best_acc = acc
        if acc_hard > best_acc_hard:
            best_acc_hard = acc_hard
        # if m.result().numpy() > best_auc:
        #     best_auc = m.result().numpy()
        if macro > best_macro:
            best_macro = macro
        if macro_hard > best_macro_hard:
            best_macro_hard = macro_hard
        # if micro > best_micro:
        #     best_micro = micro
        # print('Acc:', tf.reduce_sum(flag) / truths.shape[0])
        print("step = {}\tloss = {}".format(epoch, loss))
        print("best_acc = {}\tacc1 = {}".format(best_acc, acc))
        # print("best_auc = {}\tauc = {}".format(best_auc, m.result().numpy()))
        print("best_macro = {}\tmacro = {}".format(best_macro, macro))
        # print("best_micro = {}\tmicro = {}".format(best_micro, micro))
        print("best_acc_hard = {}\tacc_hard = {}".format(best_acc_hard, acc_hard))
        print("best_macro_hard = {}\tmacro_hard = {}".format(best_macro_hard, macro_hard))
        # print('.................')
    print("hard example number:", hard_example.shape[0])

# run_LableMe()
# run_Music() #OK
# run_SP()
# run_BCD() #OK
# run_Reuters() # OK
run_CUB() # underpart OK

print()
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# import matplotlib.cm as cm
# from matplotlib import pyplot as plt
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
# import tensorflow.keras as keras
# from tensorflow.keras import backend as K
# from tensorflow.keras.layers import Layer
# import random
# from TiReMGE.Model import TiReMGE
# from TiReMGE.utils import update_feature, get_adj, update_reliability, dis_loss, MIG_loss, eval
# from TiReMGE.process_demo import get_edge
#
# def one_hot(target, n_classes):
#     targets = np.array([target]).reshape(-1)
#     one_hot_targets = np.eye(n_classes)[targets]
#     return one_hot_targets
#
# def MIG_loss(cls_out, agg_out):
#     batch_num = cls_out.shape[0]
#
#     I = tf.cast(np.eye(batch_num), dtype=tf.float32)
#     E = tf.cast(np.ones((batch_num, batch_num)), dtype=tf.float32)
#     normalize_1 = batch_num
#     normalize_2 = batch_num * (batch_num - 1)
#
#     new_output = cls_out / cls_out.shape[1]
#     m = tf.matmul(new_output, agg_out, transpose_b=True)
#     noise = np.random.rand(1) * 0.0001
#     m1 = tf.math.log(m * I + I * noise + E - I)  # i<->i + i<->j. 此处E - I是为了让log为0，以便最大化P(i,i)互信息
#     # m1 = tf.math.log(m * I + E - I)
#     m2 = m * (E - I)  # i<->j，最小化P(i,j)互信息
#     # print(m)
#
#     # loss 来自 KL，与MIG相反数
#     return -(tf.reduce_sum(tf.reduce_sum(m1)) + batch_num) / normalize_1 + tf.reduce_sum(
#         tf.reduce_sum(m2)) / normalize_2
#
# def CEloss(y_true, y_pred):
#     # print(y_true)
#     # print(y_pred)
#     vec = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true, axis=1)
#     # print(vec)
#     mask = tf.equal(y_true[:, 0, :], -1)
#     zer = tf.zeros_like(vec)
#     loss = tf.where(mask, x=zer, y=vec)
#
#     # kernel_vals = [var for var in vars if "kernel" in var.name]
#     # l2_losses = [tf.nn.l2_loss(kernel_var) for kernel_var in kernel_vals]
#
#     return tf.reduce_sum(loss) #+ tf.add_n(l2_losses) * 0
#
# def shuffle_data(train_data, answers_bin_missings, batch_size):
#     data_num = train_data.shape[0]
#     data_index = list(range(data_num))
#     # random.shuffle(data_index)
#     # if data_num % batch_size == 0:
#     #     flag = int(data_num/batch_size)
#     # else:
#     #     flag = int(data_num / batch_size) + 1
#     shuffle_train_data = train_data[data_index]
#     shuffle_answers_bin_missings = answers_bin_missings[data_index]
#     # for i in range(flag):
#     return shuffle_train_data, shuffle_answers_bin_missings
#
# class BNN(tf.keras.Model):
#     def __init__(self, N_ANNOT, N_CLASSES, prior_sigma_1=1.5, prior_sigma_2=0.1, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.N_CLASSES = N_CLASSES
#         self.N_ANNOT = N_ANNOT
#         self.kernel_mu = tf.Variable(tf.random.truncated_normal((self.N_ANNOT, self.N_CLASSES, self.N_CLASSES), mean=0., stddev=1.))
#         self.kernel_rho = tf.Variable(tf.random.truncated_normal((self.N_ANNOT, self.N_CLASSES, self.N_CLASSES), mean=0., stddev=1.))
#         self.prior_sigma_1 = prior_sigma_1
#         self.prior_sigma_2 = prior_sigma_2
#         self.prior_pi_1 = 0.5
#         self.prior_pi_2 = 1 - self.prior_pi_1
#
#         self.kernel = 0
#         self.kernel_sigma = 0
#         self.loss = 0
#
#     def gaussian_distribution_density(self, x, mu, sigma):
#         return 1.0 / ((2 * np.pi) ** 0.5 * sigma) * tf.math.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + 1e-30
#
#     def get_prior(self):
#         prior = tf.math.log(
#             self.prior_pi_1 * self.gaussian_distribution_density(self.kernel, 0.0, self.prior_sigma_1)
#             +
#             self.prior_pi_2 * self.gaussian_distribution_density(self.kernel, 0.0, self.prior_sigma_2)
#         )  # prior
#         return prior
#
#     def get_poster(self):
#         poster = tf.math.log(self.gaussian_distribution_density(self.kernel, self.kernel_mu, self.kernel_sigma))
#         return poster
#
#     def call(self, cls_out):
#         self.kernel_sigma = tf.math.softplus(self.kernel_rho)
#         self.kernel = self.kernel_mu + self.kernel_sigma * tf.random.normal(self.kernel_mu.shape)
#
#         poster = self.get_poster()
#         prior = self.get_prior()
#         self.loss = tf.reduce_mean(poster - prior)
#
#         return tf.keras.backend.dot(cls_out, self.kernel)
#
# class LableMe_model(tf.keras.Model):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.NUM_RUNS = 30
#         self.DATA_PATH = "../dataset/LabelMe/prepared/"
#         self.N_CLASSES = 8
#         self.data_train_vgg16, self.answers, self.answers_bin_missings, self.labels_train = self.load_LabelMe_dataset()
#         self.N_ANNOT = self.answers.shape[1]
#         self.worker_feature = np.eye(self.N_ANNOT)
#
#         self.flatten = Flatten()
#         self.fc1 = Dense(128, activation='relu')
#         self.fc2 = Dense(self.N_CLASSES)
#         self.Dropout = Dropout(0.5)
#
#         # self.fc3 = Dense(128, activation=None)
#         # self.fc4 = Dense(20, activation=None)
#         # self.fc5 = Dense(20, activation=None)
#
#         self.kernel = tf.Variable(self.identity_init((self.N_ANNOT, self.N_CLASSES, self.N_CLASSES)))
#         self.bias = tf.Variable(tf.zeros((1, self.N_ANNOT, self.N_CLASSES)))
#         # self.common_kernel = tf.Variable(self.identity_init((self.N_CLASSES, self.N_CLASSES)))
#
#     def load_LabelMe_dataset(self):
#
#         def load_data(filename):
#             with open(filename, 'rb') as f:
#                 data = np.load(f)
#             return data
#
#         print("\nLoading train data...")
#
#         # images processed by VGG16
#         data_train_vgg16 = load_data(self.DATA_PATH + "data_train_vgg16.npy")
#         print(data_train_vgg16.shape)
#
#         # ground truth labels
#         labels_train = load_data(self.DATA_PATH + "labels_train.npy")
#         print(labels_train.shape)
#
#         # data from Amazon Mechanical Turk
#         print("\nLoading AMT data...")
#         answers = load_data(self.DATA_PATH + "answers.npy")
#         print(answers.shape)
#         N_ANNOT = answers.shape[1]
#         print("\nN_CLASSES:", self.N_CLASSES)
#         print("N_ANNOT:", N_ANNOT)
#
#         # load test data
#         print("\nLoading test data...")
#
#         # images processed by VGG16
#         self.data_test_vgg16 = load_data(self.DATA_PATH + "data_test_vgg16.npy")
#         print(self.data_test_vgg16.shape)
#
#         # test labels
#         self.labels_test = load_data(self.DATA_PATH + "labels_test.npy")
#         print(self.labels_test.shape)
#
#         print("\nConverting to one-hot encoding...")
#         labels_train_bin = one_hot(labels_train, self.N_CLASSES)
#         print(labels_train_bin.shape)
#         labels_test_bin = one_hot(self.labels_test, self.N_CLASSES)
#         print(labels_test_bin.shape)
#
#         answers_bin_missings = []
#         for i in range(len(answers)):
#             row = []
#             for r in range(N_ANNOT):
#                 if answers[i, r] == -1:
#                     row.append(-1 * np.ones(self.N_CLASSES))
#                 else:
#                     row.append(one_hot(answers[i, r], self.N_CLASSES)[0, :])
#             answers_bin_missings.append(row)
#         answers_bin_missings = np.array(answers_bin_missings).swapaxes(1, 2)  # task, class, worker
#         print(answers_bin_missings.shape)
#         return data_train_vgg16, answers, answers_bin_missings, labels_train
#
#     def identity_init(self, shape):
#         out = np.ones(shape, dtype=np.float32) * 0
#         if len(shape) == 3:
#             for r in range(shape[0]):
#                 for i in range(shape[1]):
#                     out[r, i, i] = 2
#         elif len(shape) == 2:
#             for i in range(shape[1]):
#                 out[i, i] = 2
#         return out
#
#     def classifier(self, input, training=None):
#         flatten_input = self.flatten(input)
#         x = self.Dropout(self.fc1(flatten_input), training)
#         cls_out = tf.nn.softmax(self.fc2(x), axis=-1)
#         return flatten_input, cls_out
#
#     # def common_module(self, input):
#     #     instance_difficulty = self.fc3(input)
#     #     instance_difficulty = tf.math.l2_normalize(self.fc4(instance_difficulty), axis=-1)
#     #
#     #     # instance_difficulty = F.normalize(instance_difficulty)
#     #     worker_feature = tf.math.l2_normalize(self.fc5(self.worker_feature), axis=-1)
#     #     # user_feature = F.normalize(user_feature)
#     #     # common_rate = torch.einsum('ij,kj->ik', (instance_difficulty, user_feature))
#     #     common_rate = tf.matmul(instance_difficulty, worker_feature, transpose_b=True)
#     #     common_rate = tf.nn.sigmoid(common_rate)
#     #     return common_rate
#
#     def call(self, input=None, training=None):
#         flatten_input, cls_out = self.classifier(input, training)
#         # common_rate = self.common_module(flatten_input)
#         # common_prob = tf.matmul(cls_out, self.common_kernel)
#         indivi_prob = tf.keras.backend.dot(cls_out, self.kernel) + self.bias
#         # crowds_out = common_rate[:, :, None] * common_prob[:, None, :] + (1 - common_rate[:, :, None]) * indivi_prob
#         return cls_out, tf.transpose(indivi_prob, [0, 2, 1])
#
# class Music_model(tf.keras.Model):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.NUM_RUNS = 30
#         self.DATA_PATH = "../dataset/music/"
#         self.N_CLASSES = 10
#         self.train_data, self.answers, self.answer_matrix, self.answers_bin_missings_0, self.answers_bin_missings_1, self.labels_train = self.load_Music_dataset()
#         self.N_ANNOT = self.answer_matrix.shape[1]
#         self.worker_feature = np.eye(self.N_ANNOT)
#
#         self.flatten = Flatten()
#         self.fc1 = Dense(128, activation='relu')
#         self.fc2 = Dense(self.N_CLASSES)
#         self.Dropout = Dropout(0.5)
#         self.bn = tf.keras.layers.BatchNormalization(center=False, scale=False)
#         self.bn1 = tf.keras.layers.BatchNormalization(center=False, scale=False)
#
#         self.fc3 = Dense(128, activation='relu')
#         self.fc4 = Dense(self.train_data.shape[-1])
#         # self.Dropout = Dropout(0.5)
#         # self.bn = tf.keras.layers.BatchNormalization(center=False, scale=False)
#         # self.bn1 = tf.keras.layers.BatchNormalization(center=False, scale=False)
#
#         # self.fc3 = Dense(128, activation=None)
#         # self.fc4 = Dense(80, activation=None)
#         # self.fc5 = Dense(80, activation=None)
#
#         self.kernel = tf.Variable(self.identity_init((self.N_ANNOT, self.N_CLASSES, self.N_CLASSES)))
#         # self.common_kernel = tf.Variable(self.identity_init((self.N_CLASSES, self.N_CLASSES)))
#         # self.BNN = BNN(self.N_ANNOT, self.N_CLASSES)
#         self.p_pure = np.ones((self.N_CLASSES), dtype=np.float32) / self.N_CLASSES
#
#     def load_Music_dataset(self):
#         truth_head = pd.read_csv('%s%s' % (self.DATA_PATH, 'truth.csv'), nrows=0)
#         truth = pd.read_csv('%s%s' % (self.DATA_PATH, 'truth.csv'), usecols=truth_head).values[:, -1]
#         # print(truth)
#
#         task_feature_head = pd.read_csv('%s%s' % (self.DATA_PATH, 'task_feature.csv'), nrows=0)
#         # print(task_feature_head)
#         task_feature = pd.read_csv('%s%s' % (self.DATA_PATH, 'task_feature.csv'), usecols=task_feature_head).values
#         # print(task_feature)
#
#         answers_head = pd.read_csv('%s%s' % (self.DATA_PATH, 'answer.csv'), nrows=0)
#         answers = pd.read_csv('%s%s' % (self.DATA_PATH, 'answer.csv'), usecols=answers_head).values
#         task_num = max(answers[:, 0]) + 1
#         worker_num = max(answers[:, 1]) + 1
#         answer_matrix = -1 * np.ones((task_num, worker_num), dtype=np.int32)
#
#         for i in range(answers.shape[0]):
#             answer_matrix[answers[i][0], answers[i][1]] = answers[i][2]
#         # print(answer_matrix[-1])
#         answers_bin_missings_0 = []
#         answers_bin_missings_1 = []
#         for i in range(len(answer_matrix)):
#             row_0 = []
#             row_1 = []
#             for r in range(worker_num):
#                 if answer_matrix[i, r] == -1:
#                     row_0.append(np.zeros(self.N_CLASSES))
#                     row_1.append(-1 * np.ones(self.N_CLASSES))
#                 else:
#                     row_0.append(one_hot(answer_matrix[i, r], self.N_CLASSES)[0, :])
#                     row_1.append(one_hot(answer_matrix[i, r], self.N_CLASSES)[0, :])
#             answers_bin_missings_0.append(row_0)
#             answers_bin_missings_1.append(row_1)
#         answers_bin_missings_0 = np.array(answers_bin_missings_0, dtype=np.float32).swapaxes(1, 0) # worker, task, class
#         answers_bin_missings_1 = np.array(answers_bin_missings_1).swapaxes(1, 2)  # task, class, worker
#         # answers_bin_missings = np.array(answers_bin_missings)  # task, worker, class
#         # print(answers_bin_missings.shape)
#         return task_feature, answers, answer_matrix, answers_bin_missings_0, answers_bin_missings_1, truth
#
#     def identity_init(self, shape):
#         out = np.ones(shape, dtype=np.float32) * 0
#         if len(shape) == 3:
#             for r in range(shape[0]):
#                 for i in range(shape[1]):
#                     out[r, i, i] = 2
#         elif len(shape) == 2:
#             for i in range(shape[1]):
#                 out[i, i] = 2
#         return out
#
#     def classifier(self, input, training=None):
#         flatten_input = self.flatten(input)
#         x = self.bn(flatten_input)
#         x = self.Dropout(self.fc1(x), training)
#         x = self.bn1(x)
#         x = self.fc2(x)
#         cls_out = tf.nn.softmax(x, axis=-1)
#         return flatten_input, x, cls_out
#
#     def decoder(self, x, training=None):
#         x = self.fc3(x)
#         # x = self.bn1(x)
#         x = self.Dropout(self.fc4(x), training)
#         # x = self.bn(x)
#         return x
#
#
#     # def common_module(self, input):
#     #     instance_difficulty = self.fc3(input)
#     #     instance_difficulty = tf.math.l2_normalize(self.fc4(instance_difficulty), axis=-1)
#     #
#     #     # instance_difficulty = F.normalize(instance_difficulty)
#     #     worker_feature = tf.math.l2_normalize(self.fc5(self.worker_feature), axis=-1)
#     #     # user_feature = F.normalize(user_feature)
#     #     # common_rate = torch.einsum('ij,kj->ik', (instance_difficulty, user_feature))
#     #     common_rate = tf.matmul(instance_difficulty, worker_feature, transpose_b=True)
#     #     common_rate = tf.nn.sigmoid(common_rate)
#     #     return common_rate
#
#     def call(self, input=None, crowds_label=None, training=None):
#         flatten_input, x, cls_out = self.classifier(input, training)
#         # recons_x = self.decoder(x)
#         # common_rate = self.common_module(flatten_input)
#         # common_prob = tf.matmul(cls_out, self.common_kernel)
#
#         # crowd_emb = tf.matmul(crowds_label, self.kernel)
#         # agg_emb = tf.reduce_sum(crowd_emb, axis=0)
#         # agg_out = agg_emb + tf.math.log(cls_out + 0.001) + tf.math.log(self.p_pure)
#
#         indivi_prob = tf.keras.backend.dot(cls_out, self.kernel) # task, worker, class
#         agg_out = tf.reduce_sum(indivi_prob, axis=1) + tf.math.log(cls_out + 0.001) + tf.math.log(self.p_pure)
#         # a = tf.argmax(indivi_prob, axis=-1)
#         # print(a)
#         # crowds_out = common_rate[:, :, None] * common_prob[:, None, :] + (1 - common_rate[:, :, None]) * indivi_prob
#         # indivi_prob = self.BNN(cls_out)
#         return cls_out, tf.transpose(indivi_prob, [0, 2, 1]), tf.nn.softmax(agg_out, axis=-1), self.kernel
#
# class SP_model(tf.keras.Model):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.NUM_RUNS = 30
#         self.DATA_PATH = "./dataset/SP/"
#         self.N_CLASSES = 2
#         self.train_data, self.answers, self.answers_bin_missings, self.labels_train = self.load_SP_dataset()
#         self.N_ANNOT = self.answers.shape[1]
#         self.worker_feature = np.eye(self.N_ANNOT)
#
#         self.flatten = Flatten()
#         self.fc1 = Dense(128, activation='relu')
#         self.fc2 = Dense(self.N_CLASSES)
#         self.Dropout = Dropout(0.5)
#         self.bn = tf.keras.layers.BatchNormalization(center=False, scale=False)
#         self.bn1 = tf.keras.layers.BatchNormalization(center=False, scale=False)
#
#         self.fc3 = Dense(128, activation=None)
#         self.fc4 = Dense(20, activation=None)
#         self.fc5 = Dense(20, activation=None)
#
#         self.kernel = tf.Variable(self.identity_init((self.N_ANNOT, self.N_CLASSES, self.N_CLASSES)))
#         self.common_kernel = tf.Variable(self.identity_init((self.N_CLASSES, self.N_CLASSES)))
#
#     def load_SP_dataset(self):
#         truth_head = pd.read_csv('%s%s' % (self.DATA_PATH, 'truth.csv'), nrows=0)
#         truth = pd.read_csv('%s%s' % (self.DATA_PATH, 'truth.csv'), usecols=truth_head).values[:, -1]
#         # print(truth)
#
#         task_feature_head = pd.read_csv('%s%s' % (self.DATA_PATH, 'task_feature.csv'), nrows=0)
#         # print(task_feature_head)
#         task_feature = pd.read_csv('%s%s' % (self.DATA_PATH, 'task_feature.csv'), usecols=task_feature_head).values
#         # print(task_feature)
#
#         answers_head = pd.read_csv('%s%s' % (self.DATA_PATH, 'answer.csv'), nrows=0)
#         answers = pd.read_csv('%s%s' % (self.DATA_PATH, 'answer.csv'), usecols=answers_head).values
#         task_num = max(answers[:, 0]) + 1
#         worker_num = max(answers[:, 1]) + 1
#         answer_matrix = -1 * np.ones((task_num, worker_num), dtype=np.int32)
#
#         for i in range(answers.shape[0]):
#             answer_matrix[answers[i][0], answers[i][1]] = answers[i][2]
#         # print(answer_matrix[-1])
#         answers_bin_missings = []
#         for i in range(len(answer_matrix)):
#             row = []
#             for r in range(worker_num):
#                 if answer_matrix[i, r] == -1:
#                     row.append(-1 * np.ones(self.N_CLASSES))
#                 else:
#                     row.append(one_hot(answer_matrix[i, r], self.N_CLASSES)[0, :])
#             answers_bin_missings.append(row)
#         answers_bin_missings = np.array(answers_bin_missings).swapaxes(1, 2)  # task, class, worker
#         # print(answers_bin_missings.shape)
#         return task_feature, answer_matrix, answers_bin_missings, truth
#
#     def identity_init(self, shape):
#         out = np.ones(shape, dtype=np.float32) * 0
#         if len(shape) == 3:
#             for r in range(shape[0]):
#                 for i in range(shape[1]):
#                     out[r, i, i] = 2
#         elif len(shape) == 2:
#             for i in range(shape[1]):
#                 out[i, i] = 2
#         return out
#
#     def classifier(self, input, training=None):
#         # base_model = Sequential()
#         # base_model.add(Flatten(input_shape=input.shape[1:]))
#         # # base_model.add(Dense(1024, activation='relu'))
#         # base_model.add(Dense(128, activation='relu'))
#         # base_model.add(Dropout(0.5))
#         # base_model.add(Dense(self.N_CLASSES))
#         # base_model.add(Activation("softmax"))
#         # base_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
#         #                    loss='categorical_crossentropy')  ## for EM-NN
#         flatten_input = self.flatten(input)
#         x = self.bn(flatten_input)
#         x = self.Dropout(self.fc1(x), training)
#         x = self.bn1(x)
#         cls_out = tf.nn.softmax(self.fc2(x), axis=-1)
#         return flatten_input, cls_out
#
#     def common_module(self, input):
#         instance_difficulty = self.fc3(input)
#         instance_difficulty = tf.math.l2_normalize(self.fc4(instance_difficulty), axis=-1)
#
#         # instance_difficulty = F.normalize(instance_difficulty)
#         worker_feature = tf.math.l2_normalize(self.fc5(self.worker_feature), axis=-1)
#         # user_feature = F.normalize(user_feature)
#         # common_rate = torch.einsum('ij,kj->ik', (instance_difficulty, user_feature))
#         common_rate = tf.matmul(instance_difficulty, worker_feature, transpose_b=True)
#         common_rate = tf.nn.sigmoid(common_rate)
#         return common_rate
#
#     def call(self, input=None, training=None):
#         flatten_input, cls_out = self.classifier(input, training)
#         common_rate = self.common_module(flatten_input)
#         common_prob = tf.matmul(cls_out, self.common_kernel)
#         indivi_prob = tf.keras.backend.dot(cls_out, self.kernel)
#         crowds_out = common_rate[:, :, None] * common_prob[:, None, :] + (1 - common_rate[:, :, None]) * indivi_prob
#         return cls_out, tf.transpose(crowds_out, [0, 2, 1])
#
# class BCD_model(tf.keras.Model):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.NUM_RUNS = 30
#         self.DATA_PATH = "./dataset/BCD/"
#         self.N_CLASSES = 2
#         self.train_data, self.answers, self.answers_bin_missings, self.labels_train = self.load_BCD_dataset()
#         self.N_ANNOT = self.answers.shape[1]
#         self.worker_feature = np.eye(self.N_ANNOT)
#
#         self.flatten = Flatten()
#         self.fc1 = Dense(128, activation='relu')
#         self.fc2 = Dense(self.N_CLASSES)
#         self.Dropout = Dropout(0.5)
#         self.bn = tf.keras.layers.BatchNormalization(center=False, scale=False)
#         self.bn1 = tf.keras.layers.BatchNormalization(center=False, scale=False)
#
#         self.fc3 = Dense(128, activation=None)
#         self.fc4 = Dense(20, activation=None)
#         self.fc5 = Dense(20, activation=None)
#
#         self.kernel = tf.Variable(self.identity_init((self.N_ANNOT, self.N_CLASSES, self.N_CLASSES)))
#         self.common_kernel = tf.Variable(self.identity_init((self.N_CLASSES, self.N_CLASSES)))
#
#     def load_BCD_dataset(self):
#         truth_head = pd.read_csv('%s%s' % (self.DATA_PATH, 'truth.csv'), nrows=0)
#         truth = pd.read_csv('%s%s' % (self.DATA_PATH, 'truth.csv'), usecols=truth_head).values[:, -1]
#         # print(truth)
#
#         task_feature_head = pd.read_csv('%s%s' % (self.DATA_PATH, 'task_feature.csv'), nrows=0)
#         # print(task_feature_head)
#         task_feature = pd.read_csv('%s%s' % (self.DATA_PATH, 'task_feature.csv'), usecols=task_feature_head).values
#         # print(task_feature)
#
#         answers_head = pd.read_csv('%s%s' % (self.DATA_PATH, 'answer.csv'), nrows=0)
#         answers = pd.read_csv('%s%s' % (self.DATA_PATH, 'answer.csv'), usecols=answers_head).values
#         task_num = max(answers[:, 0]) + 1
#         worker_num = max(answers[:, 1]) + 1
#         answer_matrix = -1 * np.ones((task_num, worker_num), dtype=np.int32)
#
#         for i in range(answers.shape[0]):
#             answer_matrix[answers[i][0], answers[i][1]] = answers[i][2]
#         # print(answer_matrix[-1])
#         answers_bin_missings = []
#         for i in range(len(answer_matrix)):
#             row = []
#             for r in range(worker_num):
#                 if answer_matrix[i, r] == -1:
#                     row.append(-1 * np.ones(self.N_CLASSES))
#                 else:
#                     row.append(one_hot(answer_matrix[i, r], self.N_CLASSES)[0, :])
#             answers_bin_missings.append(row)
#         answers_bin_missings = np.array(answers_bin_missings).swapaxes(1, 2)  # task, class, worker
#         # print(answers_bin_missings.shape)
#         return task_feature, answer_matrix, answers_bin_missings, truth
#
#     def identity_init(self, shape):
#         out = np.ones(shape, dtype=np.float32) * 0
#         if len(shape) == 3:
#             for r in range(shape[0]):
#                 for i in range(shape[1]):
#                     out[r, i, i] = 2
#         elif len(shape) == 2:
#             for i in range(shape[1]):
#                 out[i, i] = 2
#         return out
#
#     def classifier(self, input, training=None):
#         # base_model = Sequential()
#         # base_model.add(Flatten(input_shape=input.shape[1:]))
#         # # base_model.add(Dense(1024, activation='relu'))
#         # base_model.add(Dense(128, activation='relu'))
#         # base_model.add(Dropout(0.5))
#         # base_model.add(Dense(self.N_CLASSES))
#         # base_model.add(Activation("softmax"))
#         # base_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
#         #                    loss='categorical_crossentropy')  ## for EM-NN
#         flatten_input = self.flatten(input)
#         # x = self.bn(flatten_input)
#         x = self.Dropout(self.fc1(flatten_input), training)
#         # x = self.bn1(x)
#         cls_out = tf.nn.softmax(self.fc2(x), axis=-1)
#         return flatten_input, cls_out
#
#     def common_module(self, input):
#         instance_difficulty = self.fc3(input)
#         instance_difficulty = tf.math.l2_normalize(self.fc4(instance_difficulty), axis=-1)
#
#         # instance_difficulty = F.normalize(instance_difficulty)
#         worker_feature = tf.math.l2_normalize(self.fc5(self.worker_feature), axis=-1)
#         # user_feature = F.normalize(user_feature)
#         # common_rate = torch.einsum('ij,kj->ik', (instance_difficulty, user_feature))
#         common_rate = tf.matmul(instance_difficulty, worker_feature, transpose_b=True)
#         common_rate = tf.nn.sigmoid(common_rate)
#         return common_rate
#
#     def call(self, input=None, training=None):
#         flatten_input, cls_out = self.classifier(input, training)
#         common_rate = self.common_module(flatten_input)
#         common_prob = tf.matmul(cls_out, self.common_kernel)
#         indivi_prob = tf.keras.backend.dot(cls_out, self.kernel)
#         crowds_out = common_rate[:, :, None] * common_prob[:, None, :] + (1 - common_rate[:, :, None]) * indivi_prob
#         return cls_out, tf.transpose(crowds_out, [0, 2, 1])
#
# def run_LableMe():
#     best_acc = 0
#     trainer = LableMe_model()
#     batch_size = 64
#     train_data, answers, answers_bin_missings, labels_train = trainer.load_LabelMe_dataset()
#     shuffle_train_data, shuffle_answers_bin_missings = shuffle_data(train_data, answers_bin_missings, batch_size)
#     learning_rate = 1e-3
#     optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
#
#     if train_data.shape[0] % batch_size == 0:
#         steps = int(train_data.shape[0] / batch_size)
#     else:
#         steps = int((train_data.shape[0] / batch_size) + 1)
#     for epoch in range(60):
#         loss = 0
#         print('epoch:', epoch)
#         for step in range(steps):
#             # print('step:', step)
#             batch_train_data = shuffle_train_data[step * batch_size:(step + 1) * batch_size, :]
#             batch_answers_bin_missings = shuffle_answers_bin_missings[step * batch_size:(step + 1) * batch_size, :]
#             with tf.GradientTape() as tape:
#                 _, crowds_out = trainer(input=batch_train_data, training=True)
#                 # print(crowds_out.shape)
#                 # print(answers_bin_missings.shape)
#                 # s = tf.math.l2_normalize(tf.reshape(trainer.kernel - trainer.common_kernel, (trainer.N_ANNOT, -1)), axis=-1)
#                 # print(s)
#
#                 loss = CEloss(y_true=batch_answers_bin_missings, y_pred=crowds_out) #- (0.00001 * tf.reduce_sum(s))
#
#                 vars = tape.watched_variables()
#                 grads = tape.gradient(loss, vars)
#                 optimizer.apply_gradients(zip(grads, vars))
#
#         cls_out, _ = trainer(input=train_data, training=False)
#
#         flag = tf.compat.v1.to_int32(tf.equal(tf.argmax(cls_out, axis=-1), labels_train))
#         acc = tf.reduce_sum(flag) / labels_train.shape[0]
#         if acc > best_acc:
#             best_acc = acc
#         # print('Acc:', tf.reduce_sum(flag) / truths.shape[0])
#         print("epoch = {}\tloss = {}\tbest_accuracy = {}\tacc = {}".format(epoch, loss, best_acc, acc))
#         # print('.................')
#
# def run_Music():
#     best_acc = 0
#     trainer = Music_model()
#     batch_size = 700
#     train_data, answers, answer_matrix, answers_bin_missings_0, answers_bin_missings_1, labels_train = trainer.load_Music_dataset()
#     shuffle_train_data, shuffle_answers_bin_missings_1 = shuffle_data(train_data, answers_bin_missings_1, batch_size)
#
#     learning_rate = 1e-3
#     optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
#
#     if train_data.shape[0] % batch_size == 0:
#         steps = int(train_data.shape[0] / batch_size)
#     else:
#         steps = int((train_data.shape[0] / batch_size) + 1)
#     for epoch in range(1000):
#         loss = 0
#         print('epoch:', epoch)
#         for step in range(steps):
#             # print('step:', step)
#             batch_train_data = shuffle_train_data[step * batch_size:(step + 1) * batch_size, :]
#             batch_answers_bin_missings_1 = shuffle_answers_bin_missings_1[step * batch_size:(step + 1) * batch_size, :]
#             with tf.GradientTape() as tape:
#                 cls_out, crowds_out, agg_out, kernel = trainer(input=batch_train_data, crowds_label=answers_bin_missings_0, training=True)
#                 # print(tf.transpose(crowds_out, [2,0,1]).shape)
#                 # print(answers_bin_missings.shape)
#                 # s = tf.math.l2_normalize(tf.reshape(trainer.kernel - trainer.common_kernel, (trainer.N_ANNOT, -1)), axis=-1)
#                 # print(s)
#
#                 loss = CEloss(y_true=batch_answers_bin_missings_1, y_pred=crowds_out) #+ MIG_loss(cls_out, agg_out) #- (0.00001 * tf.reduce_sum(s))
#
#                 vars = tape.watched_variables()
#                 grads = tape.gradient(loss, vars)
#                 optimizer.apply_gradients(zip(grads, vars))
#
#         cls_out, crowds_out, agg_out, kernel = trainer(input=train_data, crowds_label=answers_bin_missings_0, training=False)
#         # print(tf.reduce_sum(tf.compat.v1.to_int32(tf.equal(tf.argmax(cls_out, axis=-1), tf.argmax(agg_out, axis=-1)))))
#
#         flag1 = tf.compat.v1.to_int32(tf.equal(tf.argmax(cls_out, axis=-1), labels_train))
#         acc1 = tf.reduce_sum(flag1) / labels_train.shape[0]
#
#         flag2 = tf.compat.v1.to_int32(tf.equal(tf.argmax(agg_out, axis=-1), labels_train))
#         acc2 = tf.reduce_sum(flag2) / labels_train.shape[0]
#
#         if acc1 > best_acc:
#             best_acc = acc1
#
#         # print('Acc:', tf.reduce_sum(flag) / truths.shape[0])
#         print("epoch = {}\tloss = {}\tbest_accuracy = {}\tacc1 = {}\tacc2 = {}".format(epoch, loss, best_acc, acc1, acc2))
#         # print('.................')
#     # np.save('kernel.npy', kernel)
#     # np.save('pred_y.npy', tf.argmax(cls_out, axis=-1))
#     # np.save('crowds_out.npy', crowds_out)
#     # mask = tf.equal(answers_bin_missings[:, :, :], -1)
#     # zer = tf.ones_like(crowds_out)
#     # np.save('crowds_out_mask.npy',tf.where(mask, x=-zer, y=crowds_out))
#
# def run_SP():
#     best_acc = 0
#     trainer = SP_model()
#     batch_size = 5000
#     train_data, answers, answers_bin_missings, labels_train = trainer.load_SP_dataset()
#     shuffle_train_data, shuffle_answers_bin_missings = shuffle_data(train_data, answers_bin_missings, batch_size)
#     learning_rate = 1e-3
#     optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
#
#     if train_data.shape[0] % batch_size == 0:
#         steps = int(train_data.shape[0] / batch_size)
#     else:
#         steps = int((train_data.shape[0] / batch_size) + 1)
#     for epoch in range(1000):
#         loss = 0
#         print('epoch:', epoch)
#         for step in range(steps):
#             # print('step:', step)
#             batch_train_data = shuffle_train_data[step * batch_size:(step + 1) * batch_size, :]
#             batch_answers_bin_missings = shuffle_answers_bin_missings[step * batch_size:(step + 1) * batch_size, :]
#             with tf.GradientTape() as tape:
#                 _, crowds_out = trainer(input=batch_train_data, training=True)
#                 # print(crowds_out.shape)
#                 # print(answers_bin_missings.shape)
#                 s = tf.math.l2_normalize(tf.reshape(trainer.kernel - trainer.common_kernel, (trainer.N_ANNOT, -1)), axis=-1)
#                 # print(s)
#
#                 loss = CEloss(y_true=batch_answers_bin_missings, y_pred=crowds_out) - (0.00001 * tf.reduce_sum(s))
#
#                 vars = tape.watched_variables()
#                 grads = tape.gradient(loss, vars)
#                 optimizer.apply_gradients(zip(grads, vars))
#
#         cls_out, _ = trainer(input=train_data, training=False)
#
#         flag = tf.compat.v1.to_int32(tf.equal(tf.argmax(cls_out, axis=-1), labels_train))
#         acc = tf.reduce_sum(flag) / labels_train.shape[0]
#         if acc > best_acc:
#             best_acc = acc
#         # print('Acc:', tf.reduce_sum(flag) / truths.shape[0])
#         print("step = {}\tloss = {}\tbest_accuracy = {}\tacc = {}".format(step, loss, best_acc, acc))
#         # print('.................')
#
# def run_BCD():
#     best_acc = 0
#     trainer = BCD_model()
#     batch_size = 1000
#     train_data, answers, answers_bin_missings, labels_train = trainer.load_BCD_dataset()
#     shuffle_train_data, shuffle_answers_bin_missings = shuffle_data(train_data, answers_bin_missings, batch_size)
#     learning_rate = 1e-3
#     optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
#
#     if train_data.shape[0] % batch_size == 0:
#         steps = int(train_data.shape[0] / batch_size)
#     else:
#         steps = int((train_data.shape[0] / batch_size) + 1)
#     for epoch in range(1000):
#         loss = 0
#         print('epoch:', epoch)
#         for step in range(steps):
#             # print('step:', step)
#             batch_train_data = shuffle_train_data[step * batch_size:(step + 1) * batch_size, :]
#             batch_answers_bin_missings = shuffle_answers_bin_missings[step * batch_size:(step + 1) * batch_size, :]
#             with tf.GradientTape() as tape:
#                 _, crowds_out = trainer(input=batch_train_data, training=True)
#                 # print(crowds_out.shape)
#                 # print(answers_bin_missings.shape)
#                 s = tf.math.l2_normalize(tf.reshape(trainer.kernel - trainer.common_kernel, (trainer.N_ANNOT, -1)), axis=-1)
#                 # print(s)
#
#                 loss = CEloss(y_true=batch_answers_bin_missings, y_pred=crowds_out) - (0.00001 * tf.reduce_sum(s))
#
#                 vars = tape.watched_variables()
#                 grads = tape.gradient(loss, vars)
#                 optimizer.apply_gradients(zip(grads, vars))
#
#         cls_out, _ = trainer(input=train_data, training=False)
#
#         flag = tf.compat.v1.to_int32(tf.equal(tf.argmax(cls_out, axis=-1), labels_train))
#         acc = tf.reduce_sum(flag) / labels_train.shape[0]
#         if acc > best_acc:
#             best_acc = acc
#         # print('Acc:', tf.reduce_sum(flag) / truths.shape[0])
#         print("step = {}\tloss = {}\tbest_accuracy = {}\tacc = {}".format(step, loss, best_acc, acc))
#         # print('.................')
#
# # run_LableMe()
# run_Music()
# # run_SP()
# # run_BCD()

