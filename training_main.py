# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 18:05:44 2020

@author: Xiwu Zhao
"""

import os
import tensorflow as tf
from tensorflow.keras import optimizers
from utils_for_miRNA_mRNA import loadTxtData
from utils_for_miRNA_mRNA import padding_sequence, MyModel
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def pre_processing_data(file_name):

    pre_data1 = loadTxtData(file_name, 1)
    pre_x, pre_y, max_len, _ = padding_sequence(pre_data1)

    # convert the data to tensor and shuffle
    xs1 = tf.convert_to_tensor(pre_x, dtype=tf.int32)
    ys1 = tf.convert_to_tensor(pre_y, dtype=tf.int32)
    # shuffle the whole data
    perm = tf.random.shuffle(tf.range(tf.shape(xs1)[0]))

    xs = tf.gather(xs1, perm, axis=0)
    ys = tf.gather(ys1, perm, axis=0)

    # split the data into training,val,test sets
    x_training = xs[0:int(len(ys) * 0.6), :]
    y_training = ys[0:int(len(ys) * 0.6)]
    x_val = xs[int(len(ys) * 0.6):int(len(ys) * 0.8), :]
    y_val = ys[int(len(ys) * 0.6):int(len(ys) * 0.8)]
    x_test = xs[int(len(ys) * 0.8):, :]
    y_test = ys[int(len(ys) * 0.8):]

    # convert them to tf.data.Dataset
    db_training = tf.data.Dataset.from_tensor_slices((x_training, y_training)).shuffle(10000).batch(64)
    db_val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(128)
    db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128)
    return db_training, db_val, db_test


def my_loop(file_name, my_learning_rate=0.001):
    epoch_number = 500
    epoch_count = -1
    epoch_loss = 0
    loss_threshold = [np.inf]
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    db_training, db_val, db_test = pre_processing_data(file_name)

    my_saved_model_name = "saved_model_weights_at_" + str(my_learning_rate)
    my_model = MyModel()
    optimizer = optimizers.Adam(learning_rate=my_learning_rate)

    for epoch in range(epoch_number):
        epoch_count += 1

        # training set
        my_training_loss_list_in_epoch = []
        my_training_metric_list_in_epoch = []
        for step, (x, y) in enumerate(db_training):
            with tf.GradientTape() as tape:
                my_training_forwards = my_model(x, training=True)
                y_one_hot = tf.one_hot(y, 2)
                train_loss = tf.reduce_mean(
                    (tf.losses.categorical_crossentropy(y_one_hot, my_training_forwards, from_logits=True)))
            grads = tape.gradient(train_loss, my_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, my_model.trainable_variables))

            my_training_loss_list_in_epoch.append(train_loss)
            predictions = tf.math.argmax(my_training_forwards, axis=1)
            my_training_metric = tf.keras.metrics.Accuracy()
            my_training_metric.update_state(y, predictions, sample_weight=None)
            my_training_metric_list_in_epoch.append(my_training_metric.result().numpy())
        my_training_loss = tf.math.reduce_mean(my_training_loss_list_in_epoch)
        my_training_accuracy = tf.math.reduce_mean(my_training_metric_list_in_epoch)
        train_loss_list.append(my_training_loss)
        train_acc_list.append(my_training_accuracy)

        # validation set
        my_val_loss_list_in_epoch = []
        my_val_metric_list_in_epoch = []
        for step, (x_val, y_val) in enumerate(db_val):
            y_val_forwards = my_model(x_val, training=False)
            # validation loss
            y_onehot = tf.one_hot(y_val, 2)
            val_loss = tf.reduce_mean((tf.losses.categorical_crossentropy(y_onehot, y_val_forwards, from_logits=True)))
            my_val_loss_list_in_epoch.append(val_loss)
            # validation accuracy
            predictions = tf.math.argmax(y_val_forwards, axis=1)
            my_val_metric = tf.keras.metrics.Accuracy()
            my_val_metric.update_state(y_val, predictions, sample_weight=None)
            my_val_metric_list_in_epoch.append(my_val_metric.result().numpy())

        my_val_loss = tf.math.reduce_mean(val_loss)
        my_val_accuracy = tf.math.reduce_mean(my_val_metric_list_in_epoch)
        val_loss_list.append(my_val_loss)
        val_acc_list.append(my_val_accuracy)
        print(epoch, 'training_loss = ', float(my_training_loss), ' training_accuracy = ', float(my_training_accuracy),
              '     validation_loss = ', float(my_val_loss), '  validation_accuracy = ', float(my_val_accuracy))

        # set up the early stopping
        if (loss_threshold[-1] - my_val_loss) / my_val_loss > 0.001:
            loss_threshold.append(my_val_loss)
            epoch_loss = epoch_count
        if epoch_count - epoch_loss > 30:
            my_model.save_weights(my_saved_model_name)
            print("=" * 50)
            print("Stopped at epoch", epoch_count)
            print("=" * 50)
            return float(my_training_loss), float(my_training_accuracy), float(my_val_loss), float(my_val_accuracy),\
                   train_loss_list, train_acc_list, val_loss_list, val_acc_list

    my_model.save_weights(my_saved_model_name)
    my_model.summary()

    return float(my_training_loss), float(my_training_accuracy), float(my_val_loss), float(my_val_accuracy), \
           train_loss_list, train_acc_list, val_loss_list, val_acc_list


def main():
    learning_rate_list = [0.01, 0.005, 0.001, 0.0005]
    projects_cwd = os.getcwd()
    with open(projects_cwd + "/results/" + "result_for_miRNA_and_mRNA.txt", "w") as f:
        f.write("learning_rate" + "\t" + "train_loss" + "\t" + "training_accuracy" + "\t"
                + "validation_loss" + "\t" + "validation_accuracy" + "\n")

    for my_learning_rate in learning_rate_list:
        result = my_loop("data_DeepMirTar_removeMisMissing_remained_seed1122.txt", my_learning_rate)
        with open(projects_cwd + "/results/" + "result_for_miRNA_and_mRNA.txt", "a") as f:
            f.write(str(my_learning_rate) + "\t" + str(result[0]) + "\t" + str(
                result[1]) + "\t" + str(result[2]) + "\t" + str(result[3]) + "\n")
        with open(projects_cwd + "/results/" + "loss_and_corr_for_all.txt", "a") as ff:
            ff.write(str(my_learning_rate) + "\t" + str(result[4]) + "\n" +
                     str(my_learning_rate) + "\t" + str(result[5]) + "\n" +
                     str(my_learning_rate) + "\t" + str(result[6]) + "\n" +
                     str(my_learning_rate) + "\t" + str(result[7]) + "\n")


if __name__ == '__main__':
    main()
