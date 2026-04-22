import tensorflow.keras.backend
import time
from data_loader import load_SP_dataset, load_Music_dataset, load_LabelMe_dataset, load_BCD_dataset, load_Reuters_dataset, \
                        load_Bill_dataset, load_Head_dataset, load_Shape_dataset, load_Forehead_dataset, load_Throat_dataset, load_Underpart_dataset,\
                        load_Breast_dataset, shuffle_data
import tensorflow as tf
import numpy as np
import os
import random
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
from sklearn.mixture import GaussianMixture as GMM
# import matplotlib.pyplot as plt
# import tf_geometric as tfg
# from model import Dual_Tower

def seed_tensorflow(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '0' # `pip install tensorflow-determinism` first,使用与tf>2.1

# seed_tensorflow(40)

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

class Music_model(tf.keras.Model):
    def __init__(self, task_num, feature_size, worker_num, class_num, hidden_size, answer_num, a=0.01, l=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_num = task_num
        self.feature_size = feature_size
        self.worker_num = worker_num
        self.class_num = class_num
        self.answer_num = answer_num
        self.hidden_size = hidden_size
        self.a = a
        self.l = l

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
        # crowd_bias = self.gaussian_distribution_density(masked_task_feature, self.masked_worker_rho, self.masked_worker_mu)
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
        crowd_bias = self.worker_NN(sample_task_feature, answers)
        return crowd_bias, sample_task_feature, recons_task_feature, task_mu, task_log_sigma

    def kl_Qwtheta_Pw(self, w, mu, sigma):
        Qwtheta = tf.math.log(self.gaussian_distribution_density(w, mu, sigma))  # poster
        Pw = tf.math.log(self.gaussian_distribution_density(w, 0.0, self.a))
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

        poster = self.kl_Qwtheta_Pw(self.worker_mu, tf.math.reduce_sum(self.task_mu / tf.math.exp(2 * self.task_log_sigma), axis=0)/tf.math.reduce_sum(1 / tf.math.exp(2 * self.task_log_sigma), axis=0),
                                    tf.math.reduce_sum(tf.math.exp(2 * self.task_log_sigma), axis=0)) \
                 + \
                 self.kl_Qwtheta_Pw(self.worker_rho, tf.math.reduce_sum(self.task_mu / tf.math.exp(2 * self.task_log_sigma), axis=0)/tf.math.reduce_sum(1 / tf.math.exp(2 * self.task_log_sigma), axis=0),
                                    tf.math.reduce_sum(tf.math.exp(2 * self.task_log_sigma), axis=0))
        #recons_loss
        mse_loss = self.MSE_loss(task_feature, recons_task_feature)

        # KL_loss
        kl_loss = self.KL_loss(task_mu, task_log_sigma) #+ self.KL_loss(self.worker_mu, self.worker_rho)

        total_loss = (1 - self.l) * (mse_loss + kl_loss) + self.l * (tf.reduce_sum(EC_loss) + poster)

        #loss 来自 KL，与MIG相反数
        return total_loss
        # return tf.reduce_sum(EC_loss), mig_loss, kl_loss
print()
class SP_model(tf.keras.Model):
    def __init__(self, task_num, feature_size, worker_num, class_num, hidden_size, answer_num, a, l, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_num = task_num
        self.feature_size = feature_size
        self.worker_num = worker_num
        self.class_num = class_num
        self.answer_num = answer_num
        self.hidden_size = hidden_size
        self.a = a
        self.l = l

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

        # s = tf.range(10)
        crowd_bias = masked_task_feature * tf.nn.softplus(self.masked_worker_rho) + self.masked_worker_mu # tf.nn.sigmoid(masked_worker_rho) tf.math.log(1+tf.math.exp(masked_worker_rho))
        # crowd_bias = self.gaussian_distribution_density(masked_task_feature, self.masked_worker_mu, tf.nn.softplus(self.masked_worker_rho))
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
        Pw = tf.math.log(self.gaussian_distribution_density(w, 0.0, self.a))
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
        # # 定义模型
        # gmm = GMM(n_components=self.class_num)
        # # 模型拟合
        # gmm.fit(self.task_mu)
        # # 为每个示例分配一个集群
        # yhat = gmm.predict(self.task_mu)

        # sigma_2 = 1 / tf.math.reduce_sum(tf.math.exp(-2 * self.task_log_sigma), axis=0)
        # mu_ = sigma_2 * tf.math.reduce_sum(self.task_mu / tf.math.exp(2 * self.task_log_sigma), axis=0)
        a = self.task_mu / tf.math.exp(2 * self.task_log_sigma)
        b = 1 / tf.math.exp(2 * self.task_log_sigma)
        c = tf.math.exp(2 * self.task_log_sigma)
        # poster

        # poster = self.kl_Qwtheta_Pw(self.masked_worker_mu, self.masked_task_mu, self.masked_task_sigma) \
        #          + \
        #          self.kl_Qwtheta_Pw(self.masked_worker_rho, self.masked_task_mu, self.masked_task_sigma)
        # poster = self.kl_Qwtheta_Pw(self.worker_mu, tf.math.reduce_sum(self.task_mu, axis=0),
        #                             tf.math.reduce_sum(tf.math.exp(2 * self.task_log_sigma), axis=0)) \
        #          + \
        #          self.kl_Qwtheta_Pw(self.worker_rho, tf.math.reduce_sum(self.task_mu, axis=0),
        #                             tf.math.reduce_sum(tf.math.exp(2 * self.task_log_sigma), axis=0))
        poster = self.kl_Qwtheta_Pw(self.worker_mu,
                                    tf.math.reduce_sum(a, axis=0) / tf.math.reduce_sum(b, axis=0),
                                    tf.math.reduce_sum(c, axis=0)) \
                 + \
                 self.kl_Qwtheta_Pw(self.worker_rho,
                                    tf.math.reduce_sum(a, axis=0) / tf.math.reduce_sum(b, axis=0),
                                    tf.math.reduce_sum(c, axis=0))
        # poster = self.kl_Qwtheta_Pw(self.worker_mu,
        #                             mu_,
        #                             sigma_2) \
        #          + \
        #          self.kl_Qwtheta_Pw(self.worker_rho,
        #                             mu_,
        #                             sigma_2)

        #recons_loss
        mse_loss = self.MIG_loss(task_feature, recons_task_feature)

        # recons_loss
        # recons_loss = tf.reduce_mean(tf.sqrt(tf.square(task_feature - recons_task_feature))) * 0
        # print(recons_loss)

        # KL_loss
        kl_loss = self.KL_loss(task_mu, task_log_sigma) #+ self.KL_loss(self.worker_mu, self.worker_rho)

        # CE_loss
        EC_loss = self.CE_loss(crowd_bias, answers)

        total_loss = (1 - self.l) * (mse_loss + kl_loss) + self.l * (tf.reduce_sum(EC_loss) + poster)

        #loss 来自 KL，与MIG相反数
        return total_loss
        # return tf.reduce_sum(EC_loss), mig_loss, kl_loss

class BCD_model(tf.keras.Model):
    def __init__(self, task_num, feature_size, worker_num, class_num, hidden_size, answer_num, a=0.01, l=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_num = task_num
        self.feature_size = feature_size
        self.worker_num = worker_num
        self.class_num = class_num
        self.answer_num = answer_num
        self.hidden_size = hidden_size
        self.a = a
        self.l = l

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
        self.right_bn1 = tf.keras.layers.BatchNormalization(center=False, scale=False)
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
        Pw = tf.math.log(self.gaussian_distribution_density(w, 0.0, self.a))
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

        # poster = self.kl_Qwtheta_Pw(self.masked_worker_mu, self.masked_task_mu, self.masked_task_sigma) \
        #          + \
        #          self.kl_Qwtheta_Pw(self.masked_worker_rho, self.masked_task_mu, self.masked_task_sigma)
        # poster = self.kl_Qwtheta_Pw(self.worker_mu, tf.math.reduce_sum(self.task_mu, axis=0),
        #                             tf.math.reduce_sum(tf.math.exp(2 * self.task_log_sigma), axis=0)) \
        #          + \
        #          self.kl_Qwtheta_Pw(self.worker_rho, tf.math.reduce_sum(self.task_mu, axis=0),
        #                             tf.math.reduce_sum(tf.math.exp(2 * self.task_log_sigma), axis=0))
        poster = self.kl_Qwtheta_Pw(self.worker_mu,
                                    tf.math.reduce_sum(self.task_mu / tf.math.exp(2 * self.task_log_sigma),
                                                       axis=0) / tf.math.reduce_sum(
                                        1 / tf.math.exp(2 * self.task_log_sigma), axis=0),
                                    tf.math.reduce_sum(tf.math.exp(2 * self.task_log_sigma), axis=0)) \
                 + \
                 self.kl_Qwtheta_Pw(self.worker_rho,
                                    tf.math.reduce_sum(self.task_mu / tf.math.exp(2 * self.task_log_sigma),
                                                       axis=0) / tf.math.reduce_sum(
                                        1 / tf.math.exp(2 * self.task_log_sigma), axis=0),
                                    tf.math.reduce_sum(tf.math.exp(2 * self.task_log_sigma), axis=0))

        # recons_loss
        mse_loss = self.MIG_loss(task_feature, recons_task_feature)

        # recons_loss
        # recons_loss = tf.reduce_mean(tf.sqrt(tf.square(task_feature - recons_task_feature))) * 0
        # print(recons_loss)

        # KL_loss
        kl_loss = self.KL_loss(task_mu, task_log_sigma)  # + self.KL_loss(self.worker_mu, self.worker_rho)

        # CE_loss
        EC_loss = self.CE_loss(crowd_bias, answers)

        total_loss = (1 - self.l) * (mse_loss + kl_loss) + self.l * (tf.reduce_sum(EC_loss) + poster)

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

def run_Music(inf='SI', sample_num=100, a=0.01, l=0.5):
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
    hidden_size=128
    # time = 4
    trainer = Music_model(task_num, feature_size, worker_num, class_num, hidden_size, answer_num, a, l)
    # print(9.57+10.29+10.57+10.29+9.00+10.14+10.72+9.14+10.00+10.29)

    # print(tf.math.unsorted_segment_sum(data=tf.ones_like(truths), segment_ids=truths, num_segments=max(truths)+1)/len(truths))

    shuffle_train_data, shuffle_answers_bin_missings = shuffle_data(task_feature, answers_bin_missings, batch_size)
    learning_rate = 5e-3
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    if task_feature.shape[0] % batch_size == 0:
        steps = int(task_feature.shape[0] / batch_size)
    else:
        steps = int((task_feature.shape[0] / batch_size) + 1)

    acc_cache = []
    f1_cache = []
    is_nan = False
    for epoch in tqdm(range(1000)):
        # print('epoch:', epoch)
        for step in range(steps):
            # print('step:', step)
            # batch_train_data = shuffle_train_data[step * batch_size:(step + 1) * batch_size, :]
            # batch_answers_bin_missings = shuffle_answers_bin_missings[step * batch_size:(step + 1) * batch_size, :]

            with tf.GradientTape() as tape:
                crowd_bias, sample_task_feature, recons_task_feature, task_mu, task_log_sigma = trainer([task_feature, answers], training=True)

                loss = trainer.loss_fuction(task_feature, recons_task_feature, crowd_bias, answers[:, -1], task_mu, task_log_sigma)
                # if tf.math.is_nan(loss):
                #     # print()
                #     is_nan = True
                #     break
                vars = tape.watched_variables()
                grads = tape.gradient(loss, vars)
                optimizer.apply_gradients(zip(grads, vars))

        if is_nan:
            break

        crowd_bias, sample_task_feature, recons_task_feature, task_mu, task_log_sigma = trainer([task_feature, answers], training=False)
        if inf == 'SI':
            y = sample_task_feature
            for _ in range(sample_num-1):
                y + trainer.sample(task_mu, task_log_sigma)
            y = y / sample_num
        elif inf == 'PI':
            y = task_mu
        elif inf == 'PCI':
            y = task_mu - task_log_sigma

        flag = tf.compat.v1.to_int32(tf.equal(tf.argmax(y, axis=-1), truths))
        acc = tf.reduce_sum(flag) / truths.shape[0]

        flag_hard = tf.compat.v1.to_int32(tf.equal(tf.argmax(tf.gather(y, hard_example[:, 0]), axis=-1), hard_example[:, -1]))
        acc_hard = tf.reduce_sum(flag_hard) / hard_example.shape[0]
        # print(hard_example[:, 0], hard_example[:, -1])

        # if acc>0.96:
        #     np.savetxt('./tSNE/BABNN_Music_embedding_acc-{}.csv'.format(acc), sample_task_feature)
        #     np.save('{}_task_mu.npy'.format(acc), task_mu)
        #     np.save('{}_task_log_sigma.npy'.format(acc), task_log_sigma)
        #     np.save('{}_worker_mu.npy'.format(acc), trainer.worker_mu)
        #     np.save('{}_worker_rho.npy'.format(acc), trainer.worker_rho)

        # m = tf.keras.metrics.AUC().update_state([1,0,0,1], [0, 0.5, 0.5, 0.9])

        # m = tf.keras.metrics.AUC(num_thresholds=200)
        # m.update_state(tf.one_hot(tf.argmax(sample_task_feature, axis=-1), class_num), tf.one_hot(truths, class_num))

        # my_macro_f1 = MacroF1(len(truths), len(set(truths)))
        # macro = my_macro_f1.macro_f1_score(truths,
        #                                    tf.argmax(sample_task_feature, axis=-1))[0]
        # my_macro_f1 = MacroF1(len(hard_example[:, -1]), len(set(hard_example[:, -1].numpy())))
        # macro_hard = my_macro_f1.macro_f1_score(hard_example[:, -1],
        #                                         tf.argmax(tf.gather(sample_task_feature, hard_example[:, 0]), axis=-1))[0]

        macro = f1_score(y_true=truths, y_pred=tf.argmax(y, axis=-1), average='macro')
        # micro = f1_score(y_true=truths, y_pred=tf.argmax(sample_task_feature, axis=-1), average='micro')
        macro_hard = f1_score(y_true=hard_example[:, -1], y_pred=tf.argmax(tf.gather(y, hard_example[:, 0]), axis=-1), average='macro')
        # if macro>0.83:
        #     np.savetxt('./tSNE/BABNN_Music_embedding_f1-{}.csv'.format(acc), sample_task_feature)
        acc_cache.append(acc)
        # f1_cache.append(macro)

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
        print('Acc:', tf.reduce_sum(flag) / truths.shape[0])
        print("step = {}\tloss = {}".format(epoch, loss))
        print("best_acc = {}\tacc1 = {}".format(best_acc, acc))
        # print("best_auc = {}\tauc = {}".format(best_auc, m.result().numpy()))
        print("best_macro = {}\tmacro = {}".format(best_macro, macro))
        # print("best_micro = {}\tmicro = {}".format(best_micro, micro))
        print("best_acc_hard = {}\tacc_hard = {}".format(best_acc_hard, acc_hard))
        print("best_macro_hard = {}\tmacro_hard = {}".format(best_macro_hard, macro_hard))
        print('.................')
    # np.savetxt('AccLine_music_{}_{}.csv'.format(str(sample_num), str(time)), np.array(acc_cache))
    # np.savetxt('MacroLine_music_{}_{}.csv'.format(str(sample_num), str(time)), np.array(f1_cache))
    # np.savetxt('./tSNE/BABNN_Music_embedding.csv', sample_task_feature)
    print("hard example number:", hard_example.shape[0])
    return best_acc, best_macro, is_nan
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
def run_BCD(inf='SI', sample_num=100, a=0.01, l=0.5):
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
    hidden_size = 64
    time = 5
    trainer = BCD_model(task_num, feature_size, worker_num, class_num, hidden_size, answer_num, a, l)

    # print(tf.math.unsorted_segment_sum(data=tf.ones_like(truths), segment_ids=truths, num_segments=max(truths) + 1) / len(truths))

    shuffle_train_data, shuffle_answers_bin_missings = shuffle_data(task_feature, answers_bin_missings, batch_size)
    learning_rate = 5e-3
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    if task_feature.shape[0] % batch_size == 0:
        steps = int(task_feature.shape[0] / batch_size)
    else:
        steps = int((task_feature.shape[0] / batch_size) + 1)

    acc_cache = []
    f1_cache = []
    is_nan = False
    for epoch in tqdm(range(100)):
        # print('epoch:', epoch)
        for step in range(steps):
            # print('step:', step)
            # batch_train_data = shuffle_train_data[step * batch_size:(step + 1) * batch_size, :]
            # batch_answers_bin_missings = shuffle_answers_bin_missings[step * batch_size:(step + 1) * batch_size, :]

            with tf.GradientTape() as tape:
                crowd_bias, sample_task_feature, recons_task_feature, task_mu, task_log_sigma = trainer([task_feature, answers], training=True)

                loss = trainer.loss_fuction(task_feature, recons_task_feature, crowd_bias, answers[:, -1], task_mu, task_log_sigma)

                if tf.math.is_nan(loss):
                    print()
                    is_nan = True
                    break
                vars = tape.watched_variables()
                grads = tape.gradient(loss, vars)
                optimizer.apply_gradients(zip(grads, vars))

        if is_nan:
            break

        crowd_bias, sample_task_feature, recons_task_feature, task_mu, task_log_sigma = trainer([task_feature, answers], training=False)
        if inf == 'SI':
            y = sample_task_feature
            for _ in range(sample_num-1):
                y + trainer.sample(task_mu, task_log_sigma)
            y = y / sample_num
        elif inf == 'PI':
            y = task_mu
        elif inf == 'PCI':
            y = task_mu - task_log_sigma

        # flag = tf.compat.v1.to_int32(tf.equal(tf.argmax(cls_out, axis=-1), truths))
        # acc = tf.reduce_sum(flag) / truths.shape[0]

        flag = tf.compat.v1.to_int32(tf.equal(tf.argmax(y, axis=-1), truths))
        acc = tf.reduce_sum(flag) / truths.shape[0]

        flag_hard = tf.compat.v1.to_int32(
            tf.equal(tf.argmax(tf.gather(y, hard_example[:, 0]), axis=-1), hard_example[:, -1]))
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

        macro = f1_score(y_true=truths, y_pred=tf.argmax(y, axis=-1), average='macro')
        # micro = f1_score(y_true=truths, y_pred=tf.argmax(sample_task_feature, axis=-1), average='micro')
        macro_hard = f1_score(y_true=hard_example[:, -1], y_pred=tf.argmax(tf.gather(y, hard_example[:, 0]), axis=-1), average='macro')

        acc_cache.append(acc)
        # f1_cache.append(macro)

        if acc > best_acc:
            best_acc = acc
            # np.savetxt('./tSNE/BABNN_BCD_embedding.csv', sample_task_feature)
        if acc_hard > best_acc_hard:
            best_acc_hard = acc_hard
            # np.savetxt('./tSNE/BABNN_BCD_embedding.csv', sample_task_feature)
        # if m.result().numpy() > best_auc:
        #     best_auc = m.result().numpy()
        if macro > best_macro:
            best_macro = macro
        if macro_hard > best_macro_hard:
            best_macro_hard = macro_hard
    #     # if micro > best_micro:
    #     #     best_micro = micro
    #     # print('Acc:', tf.reduce_sum(flag) / truths.shape[0])
    #     print("step = {}\tloss = {}".format(epoch, loss))
    #     print("best_acc = {}\tacc1 = {}".format(best_acc, acc))
    #     # print("best_auc = {}\tauc = {}".format(best_auc, m.result().numpy()))
    #     print("best_macro = {}\tmacro = {}".format(best_macro, macro))
    #     # print("best_micro = {}\tmicro = {}".format(best_micro, micro))
    #     print("best_acc_hard = {}\tacc_hard = {}".format(best_acc_hard, acc_hard))
    #     print("best_macro_hard = {}\tmacro_hard = {}".format(best_macro_hard, macro_hard))
    #     # print('.................')
    # # np.savetxt('AccLine_BCD_{}_{}.csv'.format(str(hidden_size), str(time)), np.array(acc_cache))
    # # np.savetxt('MacroLine_BCD_{}_{}.csv'.format(str(hidden_size), str(time)), np.array(f1_cache))
    print("hard example number:", hard_example.shape[0])
    return best_acc, best_macro, is_nan

def run_Reuters(inf='SI', sample_num=100, a=0.01, l=0.5):
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
    hidden_size = 256
    time = 9
    trainer = SP_model(task_num, feature_size, worker_num, class_num, hidden_size, answer_num, a, l)

    # print(54.71+28.78+3.08+1.01+3.86+3.86+2.52+2.18)
    # print(
    #     tf.math.unsorted_segment_sum(data=tf.ones_like(truths), segment_ids=truths, num_segments=max(truths) + 1) / len(
    #         truths))

    shuffle_train_data, shuffle_answers_bin_missings = shuffle_data(task_feature, answers_bin_missings, batch_size)
    learning_rate = 5e-3
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    if task_feature.shape[0] % batch_size == 0:
        steps = int(task_feature.shape[0] / batch_size)
    else:
        steps = int((task_feature.shape[0] / batch_size) + 1)

    acc_cache = []
    f1_cache = []
    is_nan = False
    for epoch in tqdm(range(100)):
        # print('epoch:', epoch)
        for step in range(steps):
            # print('step:', step)
            # batch_train_data = shuffle_train_data[step * batch_size:(step + 1) * batch_size, :]
            # batch_answers_bin_missings = shuffle_answers_bin_missings[step * batch_size:(step + 1) * batch_size, :]

            with tf.GradientTape() as tape:
                crowd_bias, sample_task_feature, recons_task_feature, task_mu, task_log_sigma = trainer([task_feature, answers], training=True)

                loss = trainer.loss_fuction(task_feature, recons_task_feature, crowd_bias, answers[:, -1], task_mu,
                                            task_log_sigma)

                if tf.math.is_nan(loss):
                    print()
                    is_nan = True
                    break
                vars = tape.watched_variables()
                grads = tape.gradient(loss, vars)
                optimizer.apply_gradients(zip(grads, vars))

        if is_nan:
            break
        crowd_bias, sample_task_feature, recons_task_feature, task_mu, task_log_sigma = trainer([task_feature, answers], training=False)

        if inf == 'SI':
            y = sample_task_feature
            for _ in range(sample_num-1):
                y + trainer.sample(task_mu, task_log_sigma)
            y = y / sample_num
        elif inf == 'PI':
            y = task_mu
        elif inf == 'PCI':
            y = task_mu - task_log_sigma

        # flag = tf.compat.v1.to_int32(tf.equal(tf.argmax(cls_out, axis=-1), truths))
        # acc = tf.reduce_sum(flag) / truths.shape[0]

        flag = tf.compat.v1.to_int32(tf.equal(tf.argmax(y, axis=-1), truths))
        acc = tf.reduce_sum(flag) / truths.shape[0]

        flag_hard = tf.compat.v1.to_int32(
            tf.equal(tf.argmax(tf.gather(y, hard_example[:, 0]), axis=-1), hard_example[:, -1]))
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
        # # micro = f1_score(y_true=truths, y_pred=tf.argmax(sample_task_feature, axis=-1), average='micro')
        macro_hard = f1_score(y_true=hard_example[:, -1], y_pred=tf.argmax(tf.gather(sample_task_feature, hard_example[:, 0]), axis=-1), average='macro')

        acc_cache.append(acc)
        f1_cache.append(macro)

        if acc > best_acc:
            best_acc = acc
        if acc_hard > best_acc_hard:
            best_acc_hard = acc_hard
        # if m.result().numpy() > best_auc:
        #     best_auc = m.result().numpy()
        if macro > best_macro:
            best_macro = macro
            print(classification_report(y_true=truths, y_pred=tf.argmax(sample_task_feature, axis=-1)))
        if macro_hard > best_macro_hard:
            best_macro_hard = macro_hard
        # if micro > best_micro:
        #     best_micro = micro
        # # print('Acc:', tf.reduce_sum(flag) / truths.shape[0])
        # print("step = {}\tloss = {}".format(epoch, loss))
        # print("best_acc = {}\tacc1 = {}".format(best_acc, acc))
        # # print("best_auc = {}\tauc = {}".format(best_auc, m.result().numpy()))
        # print("best_macro = {}\tmacro = {}".format(best_macro, macro))
        # # print("best_micro = {}\tmicro = {}".format(best_micro, micro))
        # print("best_acc_hard = {}\tacc_hard = {}".format(best_acc_hard, acc_hard))
        # print("best_macro_hard = {}\tmacro_hard = {}".format(best_macro_hard, macro_hard))
        # # print('.................')
    # np.savetxt('AccLine_Reuters_{}_{}.csv'.format(str(hidden_size), str(time)), np.array(acc_cache))
    # np.savetxt('MacroLine_Reuters_{}_{}.csv'.format(str(hidden_size), str(time)), np.array(f1_cache))
    # np.savetxt('./tSNE/BABNN_Reuters_embedding.csv', sample_task_feature)
    print("hard example number:", hard_example.shape[0])
    return best_acc, best_macro, is_nan


def run_CUB(inf='SI', sample_num=100, a=0.01, l=0.5, data_func=None):
    batch_size = 6033
    best_acc = 0
    best_auc = 0
    best_macro = 0
    best_acc_hard = 0
    best_macro_hard = 0

    # load_Bill_dataset, load_Head_dataset, load_Shape_dataset, load_Forehead_dataset, load_Throat_dataset, load_Underpart_dataset, load_Breast_dataset

    task_feature, answers, answer_matrix, answers_bin_missings, truths, hard_example = data_func # load_Underpart_dataset
    task_num, worker_num, class_num = answers_bin_missings.shape
    feature_size = task_feature.shape[1]
    answer_num = answers.shape[0]
    hidden_size = 128
    # time = 9
    trainer = SP_model(task_num, feature_size, worker_num, class_num, hidden_size, answer_num, a, l)

    # print(
    #     tf.math.unsorted_segment_sum(data=tf.ones_like(truths), segment_ids=truths, num_segments=max(truths) + 1) / len(
    #         truths))

    shuffle_train_data, shuffle_answers_bin_missings = shuffle_data(task_feature, answers_bin_missings, batch_size)
    learning_rate = 1e-3
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    if task_feature.shape[0] % batch_size == 0:
        steps = int(task_feature.shape[0] / batch_size)
    else:
        steps = int((task_feature.shape[0] / batch_size) + 1)
    acc_cache = []
    f1_cache = []
    is_nan = False
    start_time = time.time()
    for epoch in tqdm(range(100)):
            # print('epoch:', epoch)
            for step in range(steps):
                # print('step:', step)
                # batch_train_data = shuffle_train_data[step * batch_size:(step + 1) * batch_size, :]
                # batch_answers_bin_missings = shuffle_answers_bin_missings[step * batch_size:(step + 1) * batch_size, :]

                with tf.GradientTape() as tape:
                    crowd_bias, sample_task_feature, recons_task_feature, task_mu, task_log_sigma = trainer([task_feature, answers], training=True)

                    loss = trainer.loss_fuction(task_feature, recons_task_feature, crowd_bias, answers[:, -1], task_mu, task_log_sigma)

                    # if tf.math.is_nan(loss):
                    #     print()
                    #     is_nan = True
                    #     break
                    vars = tape.watched_variables()
                    grads = tape.gradient(loss, vars)
                    optimizer.apply_gradients(zip(grads, vars))

            # if is_nan:
            #     break

            # crowd_bias, sample_task_feature, recons_task_feature, task_mu, task_log_sigma = trainer([task_feature, answers], training=False)
            # if inf == 'SI':
            #     y = sample_task_feature
            #     for _ in range(sample_num-1):
            #         y + trainer.sample(task_mu, task_log_sigma)
            #     y = y / sample_num
            # elif inf == 'PI':
            #     y = task_mu
            # elif inf == 'PCI':
            y = task_mu - task_log_sigma
    end_time = time.time()
    print(end_time - start_time)
    #     flag = tf.compat.v1.to_int32(tf.equal(tf.argmax(y, axis=-1), truths))
    #     acc = tf.reduce_sum(flag) / truths.shape[0]
    #
    #     flag_hard = tf.compat.v1.to_int32(
    #         tf.equal(tf.argmax(tf.gather(y, hard_example[:, 0]), axis=-1), hard_example[:, -1]))
    #     acc_hard = tf.reduce_sum(flag_hard) / hard_example.shape[0]
    #     # print(hard_example[:, 0], hard_example[:, -1])
    #
    #     # if acc1>0.83:
    #     #     np.save('{}_task_mu.npy'.format(acc1), task_mu)
    #     #     np.save('{}_task_log_sigma.npy'.format(acc1), task_log_sigma)
    #     #     np.save('{}_worker_mu.npy'.format(acc1), trainer.worker_mu)
    #     #     np.save('{}_worker_rho.npy'.format(acc1), trainer.worker_rho)
    #
    #     # m = tf.keras.metrics.AUC().update_state([1,0,0,1], [0, 0.5, 0.5, 0.9])
    #
    #     # m = tf.keras.metrics.AUC(num_thresholds=200)
    #     # m.update_state(tf.one_hot(tf.argmax(sample_task_feature, axis=-1), class_num), tf.one_hot(truths, class_num))
    #
    #     macro = f1_score(y_true=truths, y_pred=tf.argmax(y, axis=-1), average='macro')
    #     # micro = f1_score(y_true=truths, y_pred=tf.argmax(sample_task_feature, axis=-1), average='micro')
    #     macro_hard = f1_score(y_true=hard_example[:, -1], y_pred=tf.argmax(tf.gather(y, hard_example[:, 0]), axis=-1), average='macro')
    #
    #     acc_cache.append(acc)
    #     f1_cache.append(macro)
    #
    #     if acc > best_acc:
    #         best_acc = acc
    #     if acc_hard > best_acc_hard:
    #         best_acc_hard = acc_hard
    #         # np.savetxt('./tSNE/BABNN_Underpart_embedding.csv', sample_task_feature)
    #     # if m.result().numpy() > best_auc:
    #     #     best_auc = m.result().numpy()
    #     if macro > best_macro:
    #         best_macro = macro
    #     if macro_hard > best_macro_hard:
    #         best_macro_hard = macro_hard
    #     # if micro > best_micro:
    #     #     best_micro = micro
    #     # # print('Acc:', tf.reduce_sum(flag) / truths.shape[0])
    #     # print("step = {}\tloss = {}".format(epoch, loss))
    #     # print("best_acc = {}\tacc1 = {}".format(best_acc, acc))
    #     # # print("best_auc = {}\tauc = {}".format(best_auc, m.result().numpy()))
    #     # print("best_macro = {}\tmacro = {}".format(best_macro, macro))
    #     # # print("best_micro = {}\tmicro = {}".format(best_micro, micro))
    #     # print("best_acc_hard = {}\tacc_hard = {}".format(best_acc_hard, acc_hard))
    #     # print("best_macro_hard = {}\tmacro_hard = {}".format(best_macro_hard, macro_hard))
    #     # # print('.................')
    # # np.savetxt('AccLine_Underpart_{}_{}.csv'.format(str(hidden_size), str(time)), np.array(acc_cache))
    # # np.savetxt('MacroLine_Underpart_{}_{}.csv'.format(str(hidden_size), str(time)), np.array(f1_cache))
    # print("hard example number:", hard_example.shape[0])
    # # np.savetxt('./tSNE/BABNN_Underpart_embedding.csv', sample_task_feature)
    return best_acc, best_macro, is_nan

# music
# PI: total 0.8271428571428572; hard 0.1797752808988764
# PCI: total 0.8042857142857143; hard 0.29213483146067415
# SI: total 0.8214285714285714; hard 0.25842696629213485

# BCD
# PI: total 0.683; hard 0.28
# PCI: total 0.673; hard 0.28
# SI: total 0.681; hard 0.301

# Reuters
# PI: total 0.9176931690929452; hard 0.7095238095238096
# PCI: total 0.9141332586786114; hard 0.7380952380952381
# SI: total 0.9148936170212766; hard 0.7380952380952381

# load_Underpart_dataset
# PI: total 0.9461296204210178; hard 0.1785714285714286
# PCI: total 0.9448035803083044; hard 0.22857142857142856
# SI: total 0.9441405602519476; hard 0.24

# load_Bill_dataset
# PI: total 0.8433615116857285; hard 0.1544973544973545
# PCI: total 0.8412066965025692; hard 0.2238095238095238
# SI: total 0.8445217967843527; hard 0.2386243386243386

# load_Breast_dataset
# PI: total 0.8030830432620587; hard 0.4386138613861386
# PCI: total 0.8128625890933201; hard 0.5
# SI: total 0.7984419028675618; hard 0.4900990099009901

# Forehead
# PI: total 0.8896782695176529; hard 0.0043859649122807015
# PCI: total 0.8771755345599205; hard 0.043859649122807015
# SI: total 0.8865453339963534; hard 0.06578947368421052

# Head
# PI: total 0.8970661362506216; hard 0.10833333333333334
# PCI: total 0.8860152494612962; hard 0.14166666666666666
# SI: total 0.8995524614619592; hard 0.21666666666666667

# load_Bill_dataset, load_Head_dataset, load_Shape_dataset, load_Forehead_dataset, load_Throat_dataset, load_Breast_dataset， load_Underpart_dataset

import itertools



a = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3]
l = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
aa, ll = np.meshgrid(a, l)
a = aa.ravel()
l = ll.ravel()
inf='PCI'

# # 1
# [0.8314, 0.8228, 0.8328, 0.83, 0.8328] # 0.82138 + 0.0048
# [0.8257, 0.8157, 0.8142, 0.8185, 0.8328]
#
# # 50
# [0.8385, 0.8242, 0.8257, 0.8314, 0.8242] # 0.8288 + 0.0055
# [0.8171, 0.8142, 0.8185, 0.8142, 0.8114]
#
# # 100
# [0.8185, 0.8303, 0.8214, 0.8328, 0.8257] # 82.57 + 0.053
# [0.8071, 0.8192, 0.8014, 0.8071, 0.82]
#
# # 150
# [0.8271, 0.8285, 0.83, 0.8185, 0.8285] # 0.8265 + 0.0041
# [0.8085, 0.8171, 0.8014, 0.7971, 0.7971]
# [0.8272, 0.8299]
# [0.8151, 0.8008]
# best_acc, best_macro, is_nan = run_Music(inf, sample_num=150) #OK
# best_acc, best_macro, is_nan = run_BCD(inf)  # OK
# best_acc, best_macro, is_nan = run_Reuters(inf)  # OK
best_acc, best_macro, is_nan = run_CUB(inf, data_func=load_Breast_dataset())  # OK

# acc_list = []
# marco_list = []
# for i in range(len(a)):
#     aaa = a[i]
#     lll = l[i]
#     print(f'a: {aaa}, l: {lll}')
#     # run_LableMe(inf)
#     # best_acc, best_macro, is_nan = run_Music(inf, a=aaa, l=lll) #OK
#     # best_acc, best_macro, is_nan = run_BCD(inf, a=aaa, l=lll)  # OK
#     # best_acc, best_macro, is_nan = run_Reuters(inf, a=aaa, l=lll)  # OK
#     best_acc, best_macro, is_nan = run_CUB(inf, data_func=load_Head_dataset(), a=aaa, l=lll)  # OK
#     if is_nan:
#         acc_list.append(acc_list[-1])
#         marco_list.append(marco_list[-1])
#     else:
#         acc_list.append(best_acc)
#         marco_list.append(best_macro)
#     # run_SP(inf)
#     # run_BCD(inf) #OK
#     # run_Reuters(inf) # OK
#     # run_CUB(inf=inf, data_func=load_Head_dataset()) # underpart OK
# print('acc_list')
# print([acc_list[i].numpy() for i in range(len(acc_list))])
# print('marco_list')
# print(marco_list)

