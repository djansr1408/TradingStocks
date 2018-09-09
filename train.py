import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

import model
import dataset
from config import config

if __name__ == "__main__":
    tf.reset_default_graph()
    with tf.Graph().as_default():
        # define placeholders
        inputs = tf.placeholder(dtype=tf.float32, shape=[None, config['num_time_steps'], config['input_length']],
                                name='inputs')
        targets = tf.placeholder(dtype=tf.float32, shape=[None, config['input_length']], name='targets')
        is_training = tf.placeholder(dtype=tf.bool, name='is_training')
        print(inputs)
        print(targets)
        print(is_training)
        # create LSTM nn model
        predictions = model.create_model(inputs, config)
        # calculate total loss
        loss = tf.reduce_mean(tf.square(predictions - targets), name="loss")
        t_vars = tf.trainable_variables()
        reg_loss = config['reg_strength'] * tf.reduce_sum([tf.nn.l2_loss(v) for v in t_vars
                                                           if not 'bias' in v.name], name='reg_loss')
        total_loss = tf.add(loss, reg_loss, name="total_loss")

        # define global step
        global_step = tf.train.get_or_create_global_step()

        # data
        data = dataset.load_data('SPY.csv')
        spy_dataset = dataset.Dataset(data, config=config)

        # define learning rate with decay
        num_steps_per_epoch = spy_dataset.num_train // config['batch_size']
        num_steps_before_decay = config['num_epochs_before_decay'] * num_steps_per_epoch
        learning_rate = tf.train.exponential_decay(learning_rate=config['initial_learning_rate'],
                                                   global_step=global_step, decay_steps=num_steps_before_decay,
                                                   decay_rate=config['decay_rate'], staircase=True)
        # minimize total loss
        minimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss,
                                                                                 global_step=global_step,
                                                                                 name='adam_minimizer')

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            test_feed = {inputs: spy_dataset.X_test,
                         targets: spy_dataset.y_test,
                         is_training: False}
            train_losses = []
            lr_values = []
            test_losses = []
            train_losses_eval = []
            train_loss_per_epoch = []
            pred = 0
            for epoch in range(config['num_epochs']):
                for X_batch, y_batch in spy_dataset.get_next_batch(config['batch_size']):
                    train_feed = {inputs: X_batch, targets: y_batch, is_training: True}
                    train_loss, _ = sess.run([total_loss, minimizer], train_feed)
                    train_losses.append(train_loss)
                    train_loss_per_epoch.append(train_loss)
                    # print(loss_val)
                    gs = sess.run(global_step)
                    print("Global step: ", gs)
                    lr = sess.run(learning_rate)
                    lr_values.append(lr)
                if epoch % config['test_model_every'] == 0:
                    test_loss, pred = sess.run([total_loss, predictions], test_feed)
                    test_losses.append(test_loss)
                    train_losses_eval.append(np.mean(np.array(train_loss_per_epoch)))
                    train_loss_per_epoch = []

            plt.plot(train_losses)
            plt.title('Train losses')
            plt.show()

            plt.plot(lr_values)
            plt.title('Learning rate')
            plt.show()

            plt.plot(test_losses, 'r')
            plt.plot(train_losses_eval, 'b')
            plt.legend(['Validation loss', 'Train loss'])
            plt.show()

            pred = np.array(pred)
            print(pred.shape)
            pred_flattened = pred.flatten()
            plt.plot(pred_flattened)
            plt.show()

            pred_averaged = np.mean(pred, axis=1)
            print(pred_averaged.shape)
            targets_averaged = np.mean(spy_dataset.y_test, axis=1)
            plt.plot(targets_averaged)
            plt.plot(pred_averaged)
            plt.title('Predictions averaged')
            plt.show()

            checkpoint_dir = os.path.join(os.getcwd(), config['log_dir'])
            print(checkpoint_dir)
            if not os.path.exists(checkpoint_dir):
                os.mkdir(checkpoint_dir)

            saver = tf.train.Saver()
            saver.save(sess, os.path.join(checkpoint_dir, config['model_name'] + '.ckpt'))
