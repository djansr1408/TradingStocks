import tensorflow as tf
import os
from dataset import load_data
from config import config
import dataset
import matplotlib.pyplot as plt
import os
import numpy as np


def trading_strategy(X, y, true_prices, total_amount_of_money, num_time_steps_ahead, fixed_amount, checkpoint_dir, graph_name):
    with tf.Session() as sess:
        graph_path = os.path.join(checkpoint_dir, graph_name + '.ckpt.meta')
        print(graph_path)
        saver = tf.train.import_meta_graph(graph_path)
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
        graph = tf.get_default_graph()
        for n in graph.as_graph_def().node:
            if n.name == 'output_layer/outputs:0':
                print(n)
        print(graph.get_tensor_by_name('inputs:0'))
        print(graph.get_tensor_by_name('targets:0'))
        print(graph.get_tensor_by_name('is_training:0'))
        test_feed = {graph.get_tensor_by_name('inputs:0'): X,
                     graph.get_tensor_by_name('targets:0'): y,
                     graph.get_tensor_by_name('is_training:0'): False
                     }
        outputs = graph.get_tensor_by_name('output_layer/outputs:0')
        pred = sess.run([outputs], test_feed)
        pred = np.array(pred[0])
        pred_mean = np.mean(pred, axis=1) if pred.shape[1] > 1 else pred
        true_values = y
        true_values_mean = np.mean(y, axis=1) if true_values.shape[1] > 1 else true_values
        plt.plot(true_values_mean, 'b')
        plt.plot(pred_mean, 'r')
        plt.title("Predictions test data")
        plt.show()

        true_prices_mean = np.mean(true_prices, axis=1) if true_prices.shape[1] > 1 else true_prices
        plt.plot(true_prices_mean)
        plt.show()

        pred_flattened = pred.flatten()
        targets_flattened = y.flatten()
        prices_flattened = np.array(true_prices).flatten()
        money_amount = total_amount_of_money
        bought_stocks = []
        for i in range(len(pred_flattened)-num_time_steps_ahead):
            if pred_flattened[i+num_time_steps_ahead] > targets_flattened[i]:
                if money_amount > fixed_amount:
                    bought_stocks.append(fixed_amount / prices_flattened[i])
                    money_amount -= fixed_amount + fixed_amount * 0.0025
                elif money_amount > 0:
                    bought_stocks.append(0.9975 * money_amount / prices_flattened[i])
                    money_amount = 0
            else:
                for s in bought_stocks:
                    money_amount += s * prices_flattened[i]
                bought_stocks = []
        print("Profit: ", money_amount - total_amount_of_money)


if __name__ == "__main__":
    config['train_ratio'] = 0
    checkpoint_dir = os.path.join(os.getcwd(), config['log_dir'])
    data = dataset.load_data('SPY_test.csv')
    spy_dataset = dataset.Dataset(data, config)
    X_test, y_test = spy_dataset.X_test, spy_dataset.y_test
    print(X_test.shape)
    print(y_test.shape)

    print("Ovo: ", spy_dataset.data_grouped[spy_dataset.num_train+spy_dataset.num_time_steps:].shape)
    print(y_test.shape)
    trading_strategy(X_test, y_test, spy_dataset.data_grouped[spy_dataset.num_train+spy_dataset.num_time_steps:],
                     total_amount_of_money=100000,
                     num_time_steps_ahead=2,
                     fixed_amount=5000,
                     checkpoint_dir=checkpoint_dir,
                     graph_name='checkpoint_00')

