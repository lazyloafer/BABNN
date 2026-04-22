import numpy as np
import pandas as pd
import tensorflow as tf

def one_hot(target, n_classes):
    targets = np.array([target]).reshape(-1)
    one_hot_targets = np.eye(n_classes)[targets]
    return one_hot_targets

def load_LabelMe_dataset(DATA_PATH="../dataset/LabelMe/prepared/", N_CLASSES=8):
    def load_data(filename):
        with open(filename, 'rb') as f:
            data = np.load(f)
        return data

    # print("\nLoading train data...")

    # images processed by VGG16
    data_train_vgg16 = load_data(DATA_PATH + "data_train_vgg16.npy")

    # ground truth labels
    labels_train = load_data(DATA_PATH + "labels_train.npy")

    # data from Amazon Mechanical Turk
    # print("\nLoading AMT data...")
    answer_matrix = load_data(DATA_PATH + "answers.npy")
    N_ANNOT = answer_matrix.shape[1]

    answers = []

    answers_bin_missings = []
    for i in range(len(answer_matrix)):
        row = []
        for r in range(N_ANNOT):
            if answer_matrix[i, r] == -1:
                row.append(0 * np.ones(N_CLASSES))
            else:
                row.append(one_hot(answer_matrix[i, r], N_CLASSES)[0, :])
                answers.append([[i, r, answer_matrix[i, r]]])
        answers_bin_missings.append(row)
    answers_bin_missings = np.array(answers_bin_missings, dtype=np.float32)  # task, worker, class
    # answers_bin_missings = np.array(answers_bin_missings).swapaxes(1, 2)  # task, class, worker
    # print(answers_bin_missings.shape)
    return tf.keras.layers.Flatten()(data_train_vgg16), np.concatenate(answers, axis=0), answer_matrix, answers_bin_missings, labels_train

def load_Music_dataset(DATA_PATH="../dataset/music/", N_CLASSES=10):
    truth_head = pd.read_csv('%s%s' % (DATA_PATH, 'truth.csv'), nrows=0)
    truth = pd.read_csv('%s%s' % (DATA_PATH, 'truth.csv'), usecols=truth_head).values.astype(np.int64)
    # print(truth)

    task_feature_head = pd.read_csv('%s%s' % (DATA_PATH, 'task_feature.csv'), nrows=0)
    # print(task_feature_head)
    task_feature = pd.read_csv('%s%s' % (DATA_PATH, 'task_feature.csv'), usecols=task_feature_head).values
    # task_feature = (task_feature - np.mean(task_feature, axis=0, keepdims=True)) / np.std(task_feature, axis=0, keepdims=True)
    # print(task_feature)

    answers_head = pd.read_csv('%s%s' % (DATA_PATH, 'answer.csv'), nrows=0)
    answers = pd.read_csv('%s%s' % (DATA_PATH, 'answer.csv'), usecols=answers_head).values.astype(np.int64)
    task_num = max(answers[:, 0]) + 1
    worker_num = max(answers[:, 1]) + 1
    answer_matrix = -1 * np.ones((task_num, worker_num), dtype=np.int32)

    task_id_crowd = answers[:, 0]
    answer_crowd = np.eye(N_CLASSES)[answers[:, -1]]
    aggre_mv = tf.math.unsorted_segment_sum(data=answer_crowd, segment_ids=task_id_crowd, num_segments=task_num)
    truth_onehot = np.eye(N_CLASSES)[truth[:, -1]]

    truth_rate = tf.reduce_sum(truth_onehot * aggre_mv, axis=-1) / tf.reduce_sum(aggre_mv, axis=-1)
    mask = np.where(truth_rate==0.0)
    hard_example = tf.gather(truth, mask)[0]
    # print(mask)

    for i in range(answers.shape[0]):
        answer_matrix[answers[i][0], answers[i][1]] = answers[i][2]

    # Q = np.zeros((task_num, self.N_CLASSES))
    # for i in range(task_num):
    #     crowd_labels = answer_matrix[i]
    #     unique_crowd_labels = np.unique(crowd_labels)
    #     for j in range(len(unique_crowd_labels)):
    #         if unique_crowd_labels[j] == -1:
    #             continue
    #         else:
    #             count = len(np.where(crowd_labels == unique_crowd_labels[j])[0])
    #             Q[i, unique_crowd_labels[j]] = count
    # Q /= worker_num
    #
    # Pi = []
    # for i in range(worker_num):
    #     pi = np.zeros((self.N_CLASSES, self.N_CLASSES))
    #     worker_answers = answer_matrix[:, i]
    #     for j in range(self.N_CLASSES):
    #         Q_j = Q[:, j]
    #         for k in range(self.N_CLASSES):
    #             mask = np.where(worker_answers == k, 1, 0)
    #             if np.sum(Q_j) == 0:
    #                 # pi[j][k] = np.log(np.sum(Q_j * mask) / np.sum(Q_j))
    #                 pi[j][k] = 0
    #             else:
    #                 pi[j][k] = np.sum(Q_j * mask) / np.sum(Q_j)
    #     Pi.append(pi)
    # # Pi = np.concatenate(Pi, axis=-1)
    # Pi = np.array(Pi, dtype=np.float32)

    answers_bin_missings = []
    for i in range(len(answer_matrix)):
        row = []
        for r in range(worker_num):
            if answer_matrix[i, r] == -1:
                row.append(0 * np.ones(N_CLASSES))
            else:
                row.append(one_hot(answer_matrix[i, r], N_CLASSES)[0, :])
        answers_bin_missings.append(row)
    answers_bin_missings = np.array(answers_bin_missings, dtype=np.float32)  # task, worker, class
    # print(answers_bin_missings.shape)
    answers_pd = pd.DataFrame(answers)
    hard_answers_pd = answers_pd[answers_pd.iloc[:, 0].isin(hard_example[:, 0].numpy().tolist())]

    crowd_count = answers_pd[2].value_counts()
    hard_crowd_count = hard_answers_pd[2].value_counts()

    hard_workers = hard_answers_pd.iloc[:, 1].unique()
    hard_truth_per = pd.DataFrame(hard_example.numpy())[1].value_counts()
    truth_per = pd.DataFrame(truth)[1].value_counts()

    df_merged = answers_pd.merge(
        pd.DataFrame(truth).rename(columns={1: 'new_val'}),  # 把 valid_df 的 1 列重命名为 new_val
        on=0,
        how='left'
    )
    num_diff = (df_merged[2] != df_merged['new_val']).sum()
    noisy_rate = num_diff / len(answers)

    return np.array(task_feature, dtype=np.float32), answers, answer_matrix, answers_bin_missings, truth[:, -1], hard_example

def load_SP_dataset(DATA_PATH="../dataset/SP/", N_CLASSES=2):
    truth_head = pd.read_csv('%s%s' % (DATA_PATH, 'truth.csv'), nrows=0)
    truth = pd.read_csv('%s%s' % (DATA_PATH, 'truth.csv'), usecols=truth_head).values[:, -1]
    # print(truth)

    task_feature_head = pd.read_csv('%s%s' % (DATA_PATH, 'task_feature.csv'), nrows=0)
    # print(task_feature_head)
    task_feature = pd.read_csv('%s%s' % (DATA_PATH, 'task_feature.csv'), usecols=task_feature_head).values
    # print(task_feature)

    answers_head = pd.read_csv('%s%s' % (DATA_PATH, 'answer.csv'), nrows=0)
    answers = pd.read_csv('%s%s' % (DATA_PATH, 'answer.csv'), usecols=answers_head).values
    task_num = max(answers[:, 0]) + 1
    worker_num = max(answers[:, 1]) + 1
    answer_matrix = -1 * np.ones((task_num, worker_num), dtype=np.int32)

    for i in range(answers.shape[0]):
        answer_matrix[answers[i][0], answers[i][1]] = answers[i][2]

    # Q = np.zeros((task_num, self.N_CLASSES))
    # for i in range(task_num):
    #     crowd_labels = answer_matrix[i]
    #     unique_crowd_labels = np.unique(crowd_labels)
    #     for j in range(len(unique_crowd_labels)):
    #         if unique_crowd_labels[j] == -1:
    #             continue
    #         else:
    #             count = len(np.where(crowd_labels == unique_crowd_labels[j])[0])
    #             Q[i, unique_crowd_labels[j]] = count
    # Q /= worker_num
    #
    # Pi = []
    # for i in range(worker_num):
    #     pi = np.zeros((self.N_CLASSES, self.N_CLASSES))
    #     worker_answers = answer_matrix[:, i]
    #     for j in range(self.N_CLASSES):
    #         Q_j = Q[:, j]
    #         for k in range(self.N_CLASSES):
    #             mask = np.where(worker_answers == k, 1, 0)
    #             if np.sum(Q_j) == 0:
    #                 # pi[j][k] = np.log(np.sum(Q_j * mask) / np.sum(Q_j))
    #                 pi[j][k] = 0
    #             else:
    #                 pi[j][k] = np.sum(Q_j * mask) / np.sum(Q_j)
    #     Pi.append(pi)
    # # Pi = np.concatenate(Pi, axis=-1)
    # Pi = np.array(Pi, dtype=np.float32)

    answers_bin_missings = []
    for i in range(len(answer_matrix)):
        row = []
        for r in range(worker_num):
            if answer_matrix[i, r] == -1:
                row.append(0 * np.ones(N_CLASSES))
            else:
                row.append(one_hot(answer_matrix[i, r], N_CLASSES)[0, :])
        answers_bin_missings.append(row)
    answers_bin_missings = np.array(answers_bin_missings, dtype=np.float32)  # task, worker, class
    # print(answers_bin_missings.shape)
    return np.array(task_feature, dtype=np.float32), answers, answer_matrix, answers_bin_missings, truth

def load_BCD_dataset(DATA_PATH="../dataset/BCD/", N_CLASSES=2):
    truth_head = pd.read_csv('%s%s' % (DATA_PATH, 'truth.csv'), nrows=0)
    truth = pd.read_csv('%s%s' % (DATA_PATH, 'truth.csv'), usecols=truth_head).values
    # print(truth)

    task_feature_head = pd.read_csv('%s%s' % (DATA_PATH, 'task_feature.csv'), nrows=0)
    # print(task_feature_head)
    task_feature = pd.read_csv('%s%s' % (DATA_PATH, 'task_feature.csv'), usecols=task_feature_head).values
    # print(task_feature)

    answers_head = pd.read_csv('%s%s' % (DATA_PATH, 'answer.csv'), nrows=0)
    answers = pd.read_csv('%s%s' % (DATA_PATH, 'answer.csv'), usecols=answers_head).values
    task_num = max(answers[:, 0]) + 1
    worker_num = max(answers[:, 1]) + 1
    answer_matrix = -1 * np.ones((task_num, worker_num), dtype=np.int32)

    task_id_crowd = answers[:, 0]
    answer_crowd = np.eye(N_CLASSES)[answers[:, -1]]
    aggre_mv = tf.math.unsorted_segment_sum(data=answer_crowd, segment_ids=task_id_crowd, num_segments=task_num)
    truth_onehot = np.eye(N_CLASSES)[truth[:, -1]]

    truth_rate = tf.reduce_sum(truth_onehot * aggre_mv, axis=-1) / tf.reduce_sum(aggre_mv, axis=-1)
    mask = np.where(truth_rate == 0.0)
    hard_example = tf.gather(truth, mask)[0]

    for i in range(answers.shape[0]):
        answer_matrix[answers[i][0], answers[i][1]] = answers[i][2]

    answers_bin_missings = []
    for i in range(len(answer_matrix)):
        row = []
        for r in range(worker_num):
            if answer_matrix[i, r] == -1:
                row.append(0 * np.ones(N_CLASSES))
            else:
                row.append(one_hot(answer_matrix[i, r], N_CLASSES)[0, :])
        answers_bin_missings.append(row)
    answers_bin_missings = np.array(answers_bin_missings, dtype=np.float32)  # task, worker, class
    # print(answers_bin_missings.shape)
    answers_pd = pd.DataFrame(answers)
    hard_answers_pd = answers_pd[answers_pd.iloc[:, 0].isin(hard_example[:, 0].numpy().tolist())]
    hard_workers = hard_answers_pd.iloc[:, 1].unique()
    hard_truth_per = pd.DataFrame(hard_example.numpy())[1].value_counts()
    truth_per = pd.DataFrame(truth)[1].value_counts()

    crowd_count = answers_pd[2].value_counts()
    hard_crowd_count = hard_answers_pd[2].value_counts()
    df_merged = answers_pd.merge(
        pd.DataFrame(truth).rename(columns={1: 'new_val'}),  # 把 valid_df 的 1 列重命名为 new_val
        on=0,
        how='left'
    )
    num_diff = (df_merged[2] != df_merged['new_val']).sum()
    noisy_rate = num_diff / len(answers)
    return np.array(task_feature, dtype=np.float32), answers, answer_matrix, answers_bin_missings, truth[:, -1], hard_example

def load_Reuters_dataset(DATA_PATH="../dataset/Reuters/", N_CLASSES=8):
    truth_head = pd.read_csv('%s%s' % (DATA_PATH, 'truth.csv'), nrows=0)
    truth = pd.read_csv('%s%s' % (DATA_PATH, 'truth.csv'), usecols=truth_head).values
    # print(truth)

    task_feature_head = pd.read_csv('%s%s' % (DATA_PATH, 'task_feature.csv'), nrows=0)
    # print(task_feature_head)
    task_feature = pd.read_csv('%s%s' % (DATA_PATH, 'task_feature.csv'), usecols=task_feature_head).values
    # print(task_feature)

    answers_head = pd.read_csv('%s%s' % (DATA_PATH, 'answer.csv'), nrows=0)
    answers = pd.read_csv('%s%s' % (DATA_PATH, 'answer.csv'), usecols=answers_head).values
    task_num = max(answers[:, 0]) + 1
    worker_num = max(answers[:, 1]) + 1
    answer_matrix = -1 * np.ones((task_num, worker_num), dtype=np.int32)

    task_id_crowd = answers[:, 0]
    answer_crowd = np.eye(N_CLASSES)[answers[:, -1]]
    aggre_mv = tf.math.unsorted_segment_sum(data=answer_crowd, segment_ids=task_id_crowd, num_segments=task_num)
    truth_onehot = np.eye(N_CLASSES)[truth[:, -1]]

    truth_rate = tf.reduce_sum(truth_onehot * aggre_mv, axis=-1) / tf.reduce_sum(aggre_mv, axis=-1)
    mask = np.where(truth_rate == 0.0)
    hard_example = tf.gather(truth, mask)[0]
    # print(mask)

    for i in range(answers.shape[0]):
        answer_matrix[answers[i][0], answers[i][1]] = answers[i][2]

    answers_bin_missings = []
    for i in range(len(answer_matrix)):
        row = []
        for r in range(worker_num):
            if answer_matrix[i, r] == -1:
                row.append(0 * np.ones(N_CLASSES))
            else:
                row.append(one_hot(answer_matrix[i, r], N_CLASSES)[0, :])
        answers_bin_missings.append(row)
    answers_bin_missings = np.array(answers_bin_missings, dtype=np.float32)  # task, worker, class
    # print(answers_bin_missings.shape)
    answers_pd = pd.DataFrame(answers)
    hard_answers_pd = answers_pd[answers_pd.iloc[:, 0].isin(hard_example[:, 0].numpy().tolist())]
    hard_workers = hard_answers_pd.iloc[:, 1].unique()
    hard_truth_per = pd.DataFrame(hard_example.numpy())[1].value_counts()
    truth_per = pd.DataFrame(truth)[1].value_counts()
    crowd_count = answers_pd[2].value_counts()
    hard_crowd_count = hard_answers_pd[2].value_counts()

    df_merged = answers_pd.merge(
        pd.DataFrame(truth).rename(columns={1: 'new_val'}),  # 把 valid_df 的 1 列重命名为 new_val
        on=0,
        how='left'
    )
    num_diff = (df_merged[2] != df_merged['new_val']).sum()
    # answers_pd[df_merged[2] != df_merged['new_val']][2].value_counts()
    noisy_rate = num_diff / len(answers)
    return np.array(task_feature, dtype=np.float32), answers, answer_matrix, answers_bin_missings, truth[:, -1], hard_example

def load_Bill_dataset(DATA_PATH="../dataset/Bill/", N_CLASSES=2):
    truth_head = pd.read_csv('%s%s' % (DATA_PATH, 'truth.csv'), nrows=0)
    truth = pd.read_csv('%s%s' % (DATA_PATH, 'truth.csv'), usecols=truth_head).values
    # print(truth)

    task_feature_head = pd.read_csv('%s%s' % (DATA_PATH, 'task_feature.csv'), nrows=0)
    # print(task_feature_head)
    task_feature = pd.read_csv('%s%s' % (DATA_PATH, 'task_feature.csv'), usecols=task_feature_head).values
    # print(task_feature)

    answers_head = pd.read_csv('%s%s' % (DATA_PATH, 'answer.csv'), nrows=0)
    answers = pd.read_csv('%s%s' % (DATA_PATH, 'answer.csv'), usecols=answers_head).values
    task_num = max(answers[:, 0]) + 1
    worker_num = max(answers[:, 1]) + 1
    answer_matrix = -1 * np.ones((task_num, worker_num), dtype=np.int32)

    task_id_crowd = answers[:, 0]
    answer_crowd = np.eye(N_CLASSES)[answers[:, -1]]
    aggre_mv = tf.math.unsorted_segment_sum(data=answer_crowd, segment_ids=task_id_crowd, num_segments=task_num)
    truth_onehot = np.eye(N_CLASSES)[truth[:, -1]]

    truth_rate = tf.reduce_sum(truth_onehot * aggre_mv, axis=-1) / tf.reduce_sum(aggre_mv, axis=-1)
    mask = np.where(truth_rate == 0.0)
    hard_example = tf.gather(truth, mask)[0]

    for i in range(answers.shape[0]):
        answer_matrix[answers[i][0], answers[i][1]] = answers[i][2]

    answers_bin_missings = []
    for i in range(len(answer_matrix)):
        row = []
        for r in range(worker_num):
            if answer_matrix[i, r] == -1:
                row.append(0 * np.ones(N_CLASSES))
            else:
                row.append(one_hot(answer_matrix[i, r], N_CLASSES)[0, :])
        answers_bin_missings.append(row)
    answers_bin_missings = np.array(answers_bin_missings, dtype=np.float32)  # task, worker, class
    # print(answers_bin_missings.shape)

    answers_pd = pd.DataFrame(answers)
    hard_answers_pd = answers_pd[answers_pd.iloc[:, 0].isin(hard_example[:, 0].numpy().tolist())]
    hard_workers = hard_answers_pd.iloc[:, 1].unique()
    hard_truth_per = pd.DataFrame(hard_example.numpy())[1].value_counts()
    truth_per = pd.DataFrame(truth)[1].value_counts()
    crowd_count = answers_pd[2].value_counts()
    hard_crowd_count = hard_answers_pd[2].value_counts()
    df_merged = answers_pd.merge(
        pd.DataFrame(truth).rename(columns={1: 'new_val'}),  # 把 valid_df 的 1 列重命名为 new_val
        on=0,
        how='left'
    )
    num_diff = (df_merged[2] != df_merged['new_val']).sum()
    noisy_rate = num_diff / len(answers)
    return np.array(task_feature, dtype=np.float32), answers, answer_matrix, answers_bin_missings, truth[:, -1], hard_example

def load_Head_dataset(DATA_PATH="../dataset/Head/", N_CLASSES=2):
    truth_head = pd.read_csv('%s%s' % (DATA_PATH, 'truth.csv'), nrows=0)
    truth = pd.read_csv('%s%s' % (DATA_PATH, 'truth.csv'), usecols=truth_head).values
    # print(truth)

    task_feature_head = pd.read_csv('%s%s' % (DATA_PATH, 'task_feature.csv'), nrows=0)
    # print(task_feature_head)
    task_feature = pd.read_csv('%s%s' % (DATA_PATH, 'task_feature.csv'), usecols=task_feature_head).values
    # print(task_feature)

    answers_head = pd.read_csv('%s%s' % (DATA_PATH, 'answer.csv'), nrows=0)
    answers = pd.read_csv('%s%s' % (DATA_PATH, 'answer.csv'), usecols=answers_head).values
    task_num = max(answers[:, 0]) + 1
    worker_num = max(answers[:, 1]) + 1
    answer_matrix = -1 * np.ones((task_num, worker_num), dtype=np.int32)

    task_id_crowd = answers[:, 0]
    answer_crowd = np.eye(N_CLASSES)[answers[:, -1]]
    aggre_mv = tf.math.unsorted_segment_sum(data=answer_crowd, segment_ids=task_id_crowd, num_segments=task_num)
    truth_onehot = np.eye(N_CLASSES)[truth[:, -1]]

    truth_rate = tf.reduce_sum(truth_onehot * aggre_mv, axis=-1) / tf.reduce_sum(aggre_mv, axis=-1)
    mask = np.where(truth_rate == 0.0)
    hard_example = tf.gather(truth, mask)[0]

    for i in range(answers.shape[0]):
        answer_matrix[answers[i][0], answers[i][1]] = answers[i][2]

    answers_bin_missings = []
    for i in range(len(answer_matrix)):
        row = []
        for r in range(worker_num):
            if answer_matrix[i, r] == -1:
                row.append(0 * np.ones(N_CLASSES))
            else:
                row.append(one_hot(answer_matrix[i, r], N_CLASSES)[0, :])
        answers_bin_missings.append(row)
    answers_bin_missings = np.array(answers_bin_missings, dtype=np.float32)  # task, worker, class
    # print(answers_bin_missings.shape)
    answers_pd = pd.DataFrame(answers)
    hard_answers_pd = answers_pd[answers_pd.iloc[:, 0].isin(hard_example[:, 0].numpy().tolist())]
    hard_workers = hard_answers_pd.iloc[:, 1].unique()
    hard_truth_per = pd.DataFrame(hard_example.numpy())[1].value_counts()
    truth_per = pd.DataFrame(truth)[1].value_counts()
    crowd_count = answers_pd[2].value_counts()
    hard_crowd_count = hard_answers_pd[2].value_counts()
    df_merged = answers_pd.merge(
        pd.DataFrame(truth).rename(columns={1: 'new_val'}),  # 把 valid_df 的 1 列重命名为 new_val
        on=0,
        how='left'
    )
    num_diff = (df_merged[2] != df_merged['new_val']).sum()
    noisy_rate = num_diff / len(answers)
    return np.array(task_feature, dtype=np.float32), answers, answer_matrix, answers_bin_missings, truth[:,
                                                                                                   -1], hard_example

def load_Shape_dataset(DATA_PATH="../dataset/Shape/", N_CLASSES=2):
    truth_head = pd.read_csv('%s%s' % (DATA_PATH, 'truth.csv'), nrows=0)
    truth = pd.read_csv('%s%s' % (DATA_PATH, 'truth.csv'), usecols=truth_head).values
    # print(truth)

    task_feature_head = pd.read_csv('%s%s' % (DATA_PATH, 'task_feature.csv'), nrows=0)
    # print(task_feature_head)
    task_feature = pd.read_csv('%s%s' % (DATA_PATH, 'task_feature.csv'), usecols=task_feature_head).values
    # print(task_feature)

    answers_head = pd.read_csv('%s%s' % (DATA_PATH, 'answer.csv'), nrows=0)
    answers = pd.read_csv('%s%s' % (DATA_PATH, 'answer.csv'), usecols=answers_head).values
    task_num = max(answers[:, 0]) + 1
    worker_num = max(answers[:, 1]) + 1
    answer_matrix = -1 * np.ones((task_num, worker_num), dtype=np.int32)

    task_id_crowd = answers[:, 0]
    answer_crowd = np.eye(N_CLASSES)[answers[:, -1]]
    aggre_mv = tf.math.unsorted_segment_sum(data=answer_crowd, segment_ids=task_id_crowd, num_segments=task_num)
    truth_onehot = np.eye(N_CLASSES)[truth[:, -1]]

    truth_rate = tf.reduce_sum(truth_onehot * aggre_mv, axis=-1) / tf.reduce_sum(aggre_mv, axis=-1)
    mask = np.where(truth_rate == 0.0)
    hard_example = tf.gather(truth, mask)[0]

    for i in range(answers.shape[0]):
        answer_matrix[answers[i][0], answers[i][1]] = answers[i][2]

    answers_bin_missings = []
    for i in range(len(answer_matrix)):
        row = []
        for r in range(worker_num):
            if answer_matrix[i, r] == -1:
                row.append(0 * np.ones(N_CLASSES))
            else:
                row.append(one_hot(answer_matrix[i, r], N_CLASSES)[0, :])
        answers_bin_missings.append(row)
    answers_bin_missings = np.array(answers_bin_missings, dtype=np.float32)  # task, worker, class
    # print(answers_bin_missings.shape)
    return np.array(task_feature, dtype=np.float32), answers, answer_matrix, answers_bin_missings, truth[:,
                                                                                                   -1], hard_example

def load_Forehead_dataset(DATA_PATH="../dataset/Forehead/", N_CLASSES=2):
    truth_head = pd.read_csv('%s%s' % (DATA_PATH, 'truth.csv'), nrows=0)
    truth = pd.read_csv('%s%s' % (DATA_PATH, 'truth.csv'), usecols=truth_head).values
    # print(truth)

    task_feature_head = pd.read_csv('%s%s' % (DATA_PATH, 'task_feature.csv'), nrows=0)
    # print(task_feature_head)
    task_feature = pd.read_csv('%s%s' % (DATA_PATH, 'task_feature.csv'), usecols=task_feature_head).values
    # print(task_feature)

    answers_head = pd.read_csv('%s%s' % (DATA_PATH, 'answer.csv'), nrows=0)
    answers = pd.read_csv('%s%s' % (DATA_PATH, 'answer.csv'), usecols=answers_head).values
    task_num = max(answers[:, 0]) + 1
    worker_num = max(answers[:, 1]) + 1
    answer_matrix = -1 * np.ones((task_num, worker_num), dtype=np.int32)

    task_id_crowd = answers[:, 0]
    answer_crowd = np.eye(N_CLASSES)[answers[:, -1]]
    aggre_mv = tf.math.unsorted_segment_sum(data=answer_crowd, segment_ids=task_id_crowd, num_segments=task_num)
    truth_onehot = np.eye(N_CLASSES)[truth[:, -1]]

    truth_rate = tf.reduce_sum(truth_onehot * aggre_mv, axis=-1) / tf.reduce_sum(aggre_mv, axis=-1)
    mask = np.where(truth_rate == 0.0)
    hard_example = tf.gather(truth, mask)[0]

    for i in range(answers.shape[0]):
        answer_matrix[answers[i][0], answers[i][1]] = answers[i][2]

    answers_bin_missings = []
    for i in range(len(answer_matrix)):
        row = []
        for r in range(worker_num):
            if answer_matrix[i, r] == -1:
                row.append(0 * np.ones(N_CLASSES))
            else:
                row.append(one_hot(answer_matrix[i, r], N_CLASSES)[0, :])
        answers_bin_missings.append(row)
    answers_bin_missings = np.array(answers_bin_missings, dtype=np.float32)  # task, worker, class
    # print(answers_bin_missings.shape)

    answers_pd = pd.DataFrame(answers)
    hard_answers_pd = answers_pd[answers_pd.iloc[:, 0].isin(hard_example[:, 0].numpy().tolist())]
    hard_workers = hard_answers_pd.iloc[:, 1].unique()
    hard_truth_per = pd.DataFrame(hard_example.numpy())[1].value_counts()
    truth_per = pd.DataFrame(truth)[1].value_counts()
    crowd_count = answers_pd[2].value_counts()
    hard_crowd_count = hard_answers_pd[2].value_counts()
    df_merged = answers_pd.merge(
        pd.DataFrame(truth).rename(columns={1: 'new_val'}),  # 把 valid_df 的 1 列重命名为 new_val
        on=0,
        how='left'
    )
    num_diff = (df_merged[2] != df_merged['new_val']).sum()
    noisy_rate = num_diff / len(answers)
    return np.array(task_feature, dtype=np.float32), answers, answer_matrix, answers_bin_missings, truth[:,
                                                                                                   -1], hard_example

def load_Throat_dataset(DATA_PATH="../dataset/Throat/", N_CLASSES=2):
    truth_head = pd.read_csv('%s%s' % (DATA_PATH, 'truth.csv'), nrows=0)
    truth = pd.read_csv('%s%s' % (DATA_PATH, 'truth.csv'), usecols=truth_head).values
    # print(truth)

    task_feature_head = pd.read_csv('%s%s' % (DATA_PATH, 'task_feature.csv'), nrows=0)
    # print(task_feature_head)
    task_feature = pd.read_csv('%s%s' % (DATA_PATH, 'task_feature.csv'), usecols=task_feature_head).values
    # print(task_feature)

    answers_head = pd.read_csv('%s%s' % (DATA_PATH, 'answer.csv'), nrows=0)
    answers = pd.read_csv('%s%s' % (DATA_PATH, 'answer.csv'), usecols=answers_head).values
    task_num = max(answers[:, 0]) + 1
    worker_num = max(answers[:, 1]) + 1
    answer_matrix = -1 * np.ones((task_num, worker_num), dtype=np.int32)

    task_id_crowd = answers[:, 0]
    answer_crowd = np.eye(N_CLASSES)[answers[:, -1]]
    aggre_mv = tf.math.unsorted_segment_sum(data=answer_crowd, segment_ids=task_id_crowd, num_segments=task_num)
    truth_onehot = np.eye(N_CLASSES)[truth[:, -1]]

    truth_rate = tf.reduce_sum(truth_onehot * aggre_mv, axis=-1) / tf.reduce_sum(aggre_mv, axis=-1)
    mask = np.where(truth_rate == 0.0)
    hard_example = tf.gather(truth, mask)[0]

    for i in range(answers.shape[0]):
        answer_matrix[answers[i][0], answers[i][1]] = answers[i][2]

    answers_bin_missings = []
    for i in range(len(answer_matrix)):
        row = []
        for r in range(worker_num):
            if answer_matrix[i, r] == -1:
                row.append(0 * np.ones(N_CLASSES))
            else:
                row.append(one_hot(answer_matrix[i, r], N_CLASSES)[0, :])
        answers_bin_missings.append(row)
    answers_bin_missings = np.array(answers_bin_missings, dtype=np.float32)  # task, worker, class
    # print(answers_bin_missings.shape)
    return np.array(task_feature, dtype=np.float32), answers, answer_matrix, answers_bin_missings, truth[:,
                                                                                                   -1], hard_example

def load_Underpart_dataset(DATA_PATH="../dataset/Underpart/", N_CLASSES=2):
    truth_head = pd.read_csv('%s%s' % (DATA_PATH, 'truth.csv'), nrows=0)
    truth = pd.read_csv('%s%s' % (DATA_PATH, 'truth.csv'), usecols=truth_head).values
    # print(truth)

    task_feature_head = pd.read_csv('%s%s' % (DATA_PATH, 'task_feature.csv'), nrows=0)
    # print(task_feature_head)
    task_feature = pd.read_csv('%s%s' % (DATA_PATH, 'task_feature.csv'), usecols=task_feature_head).values
    # print(task_feature)

    answers_head = pd.read_csv('%s%s' % (DATA_PATH, 'answer.csv'), nrows=0)
    answers = pd.read_csv('%s%s' % (DATA_PATH, 'answer.csv'), usecols=answers_head).values
    task_num = max(answers[:, 0]) + 1
    worker_num = max(answers[:, 1]) + 1
    answer_matrix = -1 * np.ones((task_num, worker_num), dtype=np.int32)

    task_id_crowd = answers[:, 0]
    answer_crowd = np.eye(N_CLASSES)[answers[:, -1]]
    aggre_mv = tf.math.unsorted_segment_sum(data=answer_crowd, segment_ids=task_id_crowd, num_segments=task_num)
    truth_onehot = np.eye(N_CLASSES)[truth[:, -1]]

    truth_rate = tf.reduce_sum(truth_onehot * aggre_mv, axis=-1) / tf.reduce_sum(aggre_mv, axis=-1)
    mask = np.where(truth_rate == 0.0)
    hard_example = tf.gather(truth, mask)[0]

    for i in range(answers.shape[0]):
        answer_matrix[answers[i][0], answers[i][1]] = answers[i][2]

    answers_bin_missings = []
    for i in range(len(answer_matrix)):
        row = []
        for r in range(worker_num):
            if answer_matrix[i, r] == -1:
                row.append(0 * np.ones(N_CLASSES))
            else:
                row.append(one_hot(answer_matrix[i, r], N_CLASSES)[0, :])
        answers_bin_missings.append(row)
    answers_bin_missings = np.array(answers_bin_missings, dtype=np.float32)  # task, worker, class
    # print(answers_bin_missings.shape)

    answers_pd = pd.DataFrame(answers)
    hard_answers_pd = answers_pd[answers_pd.iloc[:, 0].isin(hard_example[:, 0].numpy().tolist())]
    hard_workers = hard_answers_pd.iloc[:, 1].unique()
    hard_truth_per = pd.DataFrame(hard_example.numpy())[1].value_counts()
    truth_per = pd.DataFrame(truth)[1].value_counts()
    crowd_count = answers_pd[2].value_counts()
    hard_crowd_count = hard_answers_pd[2].value_counts()
    df_merged = answers_pd.merge(
        pd.DataFrame(truth).rename(columns={1: 'new_val'}),  # 把 valid_df 的 1 列重命名为 new_val
        on=0,
        how='left'
    )
    num_diff = (df_merged[2] != df_merged['new_val']).sum()
    noisy_rate = num_diff / len(answers)
    return np.array(task_feature, dtype=np.float32), answers, answer_matrix, answers_bin_missings, truth[:, -1], hard_example

def load_Breast_dataset(DATA_PATH="../dataset/Breast/", N_CLASSES=2):
    truth_head = pd.read_csv('%s%s' % (DATA_PATH, 'truth.csv'), nrows=0)
    truth = pd.read_csv('%s%s' % (DATA_PATH, 'truth.csv'), usecols=truth_head).values
    # print(truth)

    task_feature_head = pd.read_csv('%s%s' % (DATA_PATH, 'task_feature.csv'), nrows=0)
    # print(task_feature_head)
    task_feature = pd.read_csv('%s%s' % (DATA_PATH, 'task_feature.csv'), usecols=task_feature_head).values
    # print(task_feature)

    answers_head = pd.read_csv('%s%s' % (DATA_PATH, 'answer.csv'), nrows=0)
    answers = pd.read_csv('%s%s' % (DATA_PATH, 'answer.csv'), usecols=answers_head).values
    task_num = max(answers[:, 0]) + 1
    worker_num = max(answers[:, 1]) + 1
    answer_matrix = -1 * np.ones((task_num, worker_num), dtype=np.int32)

    task_id_crowd = answers[:, 0]
    answer_crowd = np.eye(N_CLASSES)[answers[:, -1]]
    aggre_mv = tf.math.unsorted_segment_sum(data=answer_crowd, segment_ids=task_id_crowd, num_segments=task_num)
    truth_onehot = np.eye(N_CLASSES)[truth[:, -1]]

    truth_rate = tf.reduce_sum(truth_onehot * aggre_mv, axis=-1) / tf.reduce_sum(aggre_mv, axis=-1)
    mask = np.where(truth_rate == 0.0)
    hard_example = tf.gather(truth, mask)[0]

    for i in range(answers.shape[0]):
        answer_matrix[answers[i][0], answers[i][1]] = answers[i][2]

    answers_bin_missings = []
    for i in range(len(answer_matrix)):
        row = []
        for r in range(worker_num):
            if answer_matrix[i, r] == -1:
                row.append(0 * np.ones(N_CLASSES))
            else:
                row.append(one_hot(answer_matrix[i, r], N_CLASSES)[0, :])
        answers_bin_missings.append(row)
    answers_bin_missings = np.array(answers_bin_missings, dtype=np.float32)  # task, worker, class
    # print(answers_bin_missings.shape)

    answers_pd = pd.DataFrame(answers)
    hard_answers_pd = answers_pd[answers_pd.iloc[:, 0].isin(hard_example[:, 0].numpy().tolist())]
    hard_workers = hard_answers_pd.iloc[:, 1].unique()
    hard_truth_per = pd.DataFrame(hard_example.numpy())[1].value_counts()
    truth_per = pd.DataFrame(truth)[1].value_counts()
    crowd_count = answers_pd[2].value_counts()
    hard_crowd_count = hard_answers_pd[2].value_counts()
    df_merged = answers_pd.merge(
        pd.DataFrame(truth).rename(columns={1: 'new_val'}),  # 把 valid_df 的 1 列重命名为 new_val
        on=0,
        how='left'
    )
    num_diff = (df_merged[2] != df_merged['new_val']).sum()
    noisy_rate = num_diff / len(answers)
    return np.array(task_feature, dtype=np.float32), answers, answer_matrix, answers_bin_missings, truth[:,
                                                                                                   -1], hard_example

def shuffle_data(train_data, answers_bin_missings, batch_size):
    data_num = train_data.shape[0]
    data_index = list(range(data_num))
    # random.shuffle(data_index)
    # if data_num % batch_size == 0:
    #     flag = int(data_num/batch_size)
    # else:
    #     flag = int(data_num / batch_size) + 1
    shuffle_train_data = train_data[data_index]
    shuffle_answers_bin_missings = answers_bin_missings[data_index]
    # for i in range(flag):
    return shuffle_train_data, shuffle_answers_bin_missings
# data_train_vgg16, answers, answers, answers_bin_missings, labels_train = load_SP_dataset()
# load_Reuters_dataset()
