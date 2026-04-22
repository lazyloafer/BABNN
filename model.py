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

    def encoder(self, task_feature, training=False):
        # task_feature = self.left_bn_1(task_feature)
        h = self.Efc(task_feature)
        # h = self.left_bn_2(h)
        # mean = self.Efc_mean(h)
        # log_std = self.Efc_log_std(h)
        cls_out = self.Efc_mean(h)
        # return mean, log_std
        return tf.nn.softmax(cls_out)

    # def sample_z(self, mean, log_std):
    #     std = tf.math.exp(log_std)
    #     eps = tf.random.normal(shape=(self.task_num, self.class_num), mean=0, stddev=0.01)
    #     return mean + eps * std
    #
    def decoder(self, z, training=False):
        h = self.Dfc_1(z)
        _x = self.Dfc_2(h)
        return _x

    def call(self, x, training=False):
        # mean, log_std = self.encoder(x, training=training)
        # z = self.sample_z(mean, log_std)
        # _x = self.decoder(z, training=training)
        cls_out = self.encoder(x, training=training)
        # return _x, mean, log_std
        return cls_out

class VAE_crowds(tf.keras.Model):
    def __init__(self, task_num, feature_size, worker_num, class_num, answer_num, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_num = task_num
        self.feature_size = feature_size
        self.worker_num = worker_num
        self.class_num = class_num
        self.answer_num = answer_num
        self.hidden_size = 128

        self.worker_feature = tf.Variable(tf.random.normal((self.worker_num, self.hidden_size)))

        self.Efc = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        self.right_bn = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        self.gnn_share = tfg.layers.GCN(self.hidden_size, activation=tf.nn.relu)
        self.gnn_mean = tfg.layers.GCN(self.class_num, activation=None)
        self.gnn_log_std = tfg.layers.GCN(self.class_num, activation=None)
        # self.Efc_mean = tf.keras.layers.Dense(self.class_num, activation=None)
        # self.Efc_log_std = tf.keras.layers.Dense(self.class_num, activation=None)
        self.left_bn_1 = tf.keras.layers.BatchNormalization(center=False, scale=False)
        self.left_bn_2 = tf.keras.layers.BatchNormalization(center=False, scale=False)

        self.Dfc_1 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        self.Dfc_2 = tf.keras.layers.Dense(self.feature_size, activation=None)

    def encoder(self, task_feature, answers, training=None):

        task_ids = answers[:, 0]
        worker_ids = answers[:, 1]
        label_ids = answers[:, 2]

        row = task_ids
        col = worker_ids + self.task_num
        Row = np.concatenate([row, col], axis=-1)
        Col = np.concatenate([col, row], axis=-1)
        edge_index = np.concatenate([[Row], [Col]], axis=0)

        # task_feature = self.left_bn_1(task_feature)
        task_feature = self.Efc(task_feature)
        # worker_feature = self.right_bn(self.worker_feature)
        # h = self.left_bn_2(h)

        node_feature = tf.concat([task_feature, self.worker_feature], axis=0)
        # node_feature = self.gnn_share([node_feature, np.array(edge_index, dtype=np.int32)], training=training)
        mean = self.gnn_mean([node_feature, np.array(edge_index, dtype=np.int32)], training=training)
        log_std = self.gnn_log_std([node_feature, np.array(edge_index, dtype=np.int32)], training=training)

        # mean = self.Efc_mean(gnn_out)
        # log_std = self.Efc_log_std(gnn_out)

        return mean, log_std

    def sample_z(self, mean, log_std):
        std = tf.math.exp(log_std)
        eps = tf.random.normal(shape=(self.task_num + self.worker_num, self.class_num), mean=0, stddev=0.01)
        return mean + eps * std

    def decoder(self, z, answers):
        task_ids = answers[:, 0]
        worker_ids = answers[:, 1]

        row = task_ids
        col = worker_ids + self.task_num

        masked_task_feature = tf.gather(z, row)
        masked_worker_feature = tf.gather(z, col)

        crowd_out = masked_task_feature * masked_worker_feature
        return crowd_out

    def call(self, task_feature, answers, training=False):
        mean, log_std = self.encoder(task_feature, answers)
        z = self.sample_z(mean, log_std)
        crowd_out = self.decoder(z, answers)
        return crowd_out, mean, log_std

class Dual_Tower(tf.keras.Model):
    def __init__(self, task_num, feature_size, worker_num, class_num, answer_num, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_num = task_num
        self.feature_size = feature_size
        self.worker_num = worker_num
        self.class_num = class_num
        self.answer_num = answer_num
        self.hidden_size = 128
        self.Task_VAE = VAE_no_crowds(task_num, feature_size, worker_num, class_num, answer_num)
        self.Crowds_VAE = VAE_crowds(task_num, feature_size, worker_num, class_num, answer_num)
        self.kl_loss = tf.keras.losses.KLDivergence()

    def sample_z(self, crowd_mean, crowd_log_std, answers):
        task_ids = answers[:, 0]
        worker_ids = answers[:, 1]
        row = task_ids
        col = worker_ids + self.task_num

        crowd_std = tf.math.exp(crowd_log_std)

        masked_worker_crowd_mean = tf.gather(crowd_mean, col)
        agg_worker_crowd_mean = tf.math.unsorted_segment_sum(data=masked_worker_crowd_mean, segment_ids=task_ids, num_segments=self.task_num)

        masked_worker_crowd_std = tf.gather(crowd_std, col)
        agg_worker_crowd_std = tf.math.unsorted_segment_sum(data=masked_worker_crowd_std, segment_ids=task_ids, num_segments=self.task_num)

        # de_bias_mean = tf.gather(crowd_mean, range(self.task_num)) - agg_worker_crowd_mean
        # de_bias_std = tf.gather(crowd_std, range(self.task_num)) / agg_worker_crowd_std

        eps = tf.random.normal(shape=(self.task_num, self.class_num), mean=0, stddev=0.01)
        # return de_bias_mean + eps * de_bias_std, de_bias_mean, de_bias_std
        return (tf.gather(crowd_mean, range(self.task_num))-agg_worker_crowd_mean) + eps * (tf.gather(crowd_std, range(self.task_num)) - agg_worker_crowd_std), \
               (tf.gather(crowd_mean, range(self.task_num))-agg_worker_crowd_mean), (tf.gather(crowd_std, range(self.task_num)) - agg_worker_crowd_std)

    def loss_function(self, task_feature, cls_out, crowd_out, de_bias_mean, de_bias_std, z, de_bias_task_feature, answers):

        # re_1 = self.kl_loss(task_feature, reconstructed_task_feature)
        re_2 = self.kl_loss(task_feature, de_bias_task_feature)

        # KL_1 = -0.5*tf.reduce_mean(1 + 2*task_log_std - tf.square(task_mean) - tf.square(tf.exp(task_log_std)))
        KL_2 = -0.5*tf.reduce_sum(1 + 2*tf.math.log(de_bias_std) - tf.square(de_bias_mean) - tf.square(de_bias_std))

        batch_num = cls_out.shape[0]
        I = tf.cast(np.eye(batch_num), dtype=tf.float32)
        E = tf.cast(np.ones((batch_num, batch_num)), dtype=tf.float32)
        normalize_1 = batch_num
        normalize_2 = batch_num * (batch_num - 1)

        new_output = cls_out  # / self.p_pure
        m = tf.matmul(new_output, z, transpose_b=True)
        noise = np.random.rand(1) * 0.0001
        m1 = tf.math.log(m * I + I * noise + E - I)  # i<->i + i<->j. 此处E - I是为了让log为0，以便最大化P(i,i)互信息
        # m1 = tf.math.log(m * I + E - I)  # i<->i + i<->j. 此处E - I是为了让log为0，以便最大化P(i,i)互信息
        m2 = m * (E - I)  # i<->j，最小化P(i,j)互信息
        mig_loss = -(tf.reduce_sum(tf.reduce_sum(m1)) + batch_num) / normalize_1 + tf.reduce_sum(
            tf.reduce_sum(m2)) / normalize_2


        EC_loss = tf.nn.softmax_cross_entropy_with_logits(logits=crowd_out,
                                                          labels=tf.one_hot(indices=answers, depth=self.class_num),
                                                          axis=1)
        # print('EC_loss:', EC_loss)
        return tf.reduce_sum(EC_loss) + mig_loss + re_2 + KL_2

    def call(self, x, answers, training=False):
        cls_out = self.Task_VAE(x, training=training)

        # reconstructed_task_feature, task_mean, task_log_std = self.Task_VAE(x, training=training)
        crowd_out, crowd_mean, crowd_log_std = self.Crowds_VAE(x, answers, training=training)

        z, de_bias_mean, de_bias_std = self.sample_z(crowd_mean, crowd_log_std, answers)
        de_bias_task_feature = self.Task_VAE.decoder(z, training=training)

        return cls_out, crowd_out, de_bias_mean, de_bias_std, tf.nn.softmax(z), de_bias_task_feature

        # return [reconstructed_task_feature, task_mean, task_log_std], [de_bias_task_feature, de_bias_mean, de_bias_std], crowd_out, z
