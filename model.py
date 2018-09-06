import tensorflow as tf

config = {
    'input_length': 3,
    'num_time_steps': 20,
    'lstm_size': 200,
    'num_lstm_cells': 3,
    'dropout_prob': 0.5
}


def create_lstm_cell(lstm_size, dropout_prob=None):
    lstm = tf.contrib.rnn.LSTMCell(num_units=lstm_size, initializer=tf.truncated_normal_initializer(stddev=1e-2))
    if dropout_prob is not None:
        lstm = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=dropout_prob)
    return lstm


def create_model(config):
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, config['num_time_steps'], config['input_length']],
                            name='inputs')
    targets = tf.placeholder(dtype=tf.float32, shape=[None, config['input_length']], name='targets')
    with tf.name_scope("lstm_layers"):
        if config['num_lstm_cells'] > 1:
            lstm_cells = tf.contrib.rnn.MultiRNNCell([create_lstm_cell(config['lstm_size'],
                                                                       dropout_prob=config['dropout_prob'])
                                                      for i in range(config['num_lstm_cells'])])
        else:
            lstm_cells = create_lstm_cell(config['lstm_size'], dropout_prob=config['dropout_prob'])
        lstm_outputs, lstm_states = tf.nn.dynamic_rnn(lstm_cells, inputs, dtype=tf.float32, scope='lstm_cells')
        lstm_outputs_transposed = tf.transpose(lstm_outputs, [1, 0, 2])

    with tf.name_scope("output_layer"):
        last_time_step = tf.gather(lstm_outputs_transposed, lstm_outputs_transposed.get_shape()[0] - 1, name='last_time_step')

        # FC layer after lstm
        weight = tf.Variable(dtype=tf.float32, shape=[config['lstm_size'], config['input_length']],
                             initializer=tf.truncated_normal_initializer())
        biases = tf.Variable(dtype=tf.float32, shape=[config['input_size']], initializer= tf.zeros_initializer())

        outputs = tf.add(tf.matmul(last_time_step, weight), biases)


if __name__ == "__main__":
    create_model(config)
