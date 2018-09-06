import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

config = {'input_length': 3,
          'num_time_steps': 20,
          'batch_size': 50,
          'train_ratio': 0.9
         }

class Dataset:
    def __init__(self, data, config):
        self.data = data
        self.input_length = config['input_length']
        self.num_time_steps = config['num_time_steps']
        self.train_ratio = config['train_test_ratio']
        self._format_data()

    def _format_data(self):
        # split data to groups of equal number of samples
        data_grouped = [np.array(self.data[i*self.input_length:(i+1)*self.input_length])
                        for i in range(len(self.data)//self.input_length)]

        # normalize data
        self.normalized_data = []
        self.normalized_data.append(data_grouped[0] / data_grouped[0][0] - 1.0)
        for i in range(1, len(data_grouped)):
            self.normalized_data.append(data_grouped[i]/data_grouped[i-1][-1] - 1.0)

        X = np.array([self.normalized_data[i:i+self.num_time_steps]
                      for i in range(len(self.normalized_data)-self.num_time_steps)])
        y = np.array([self.normalized_data[i + self.num_time_steps]
                      for i in range(len(self.normalized_data)-self.num_time_steps)])
        num_train = int(self.train_ratio * y.shape[0])
        self.X_train = X[:num_train]
        self.y_train = y[:num_train]
        self.X_test = X[num_train:]
        self.y_test = y[num_train:]

    def get_next_batch(self, batch_size, allow_smaller_last_batch=False):
        num_batches = int(len(self.y_train) // batch_size)
        if num_batches * batch_size < len(self.y_train) and allow_smaller_last_batch:
            num_batches += 1
        indices = np.arange(num_batches)
        random.shuffle(indices)
        for i in indices:
            X_batch = self.X_train[i*batch_size:(i+1)*batch_size]
            y_batch = self.y_train[i*batch_size:(i+1)*batch_size]
            yield X_batch, y_batch


if __name__ == "__main__":
    df = pd.read_csv('SPY.csv')
    close_prices = df['Close'].values.tolist()
    close_prices = np.array(close_prices)
    # plt.plot(close_prices)
    # plt.title("Original data")
    # plt.show()

    # averaged for num_steps = 50
    num_steps = 50
    averaged = []
    for i in range(len(close_prices)-num_steps):
        averaged.append(np.mean(close_prices[i:i+num_steps]))


    #plt.plot(averaged)
    #plt.title("Averaged for num_steps=" + str(num_steps))
    #plt.show()

    dataset = Dataset(close_prices, config=config)
    dataset.format_data()














