import tensorflow as tf
from config import config


def create_lstm_cell(lstm_size, dropout_prob=None, is_training=True):
    """
        This function returns single LSTM cell.
    :param lstm_size:
    :param dropout_prob:
    :return: lstm cell
    """
    lstm = tf.contrib.rnn.LSTMCell(num_units=lstm_size, initializer=tf.truncated_normal_initializer(stddev=1e-2))
    if is_training and dropout_prob is not None:
        lstm = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=dropout_prob)
    elif is_training is False:
        lstm = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=1.0)
    return lstm


def create_model(inputs, config, is_training=True):
    """
        This function builds LSTM neural network and returns its outputs
    :param inputs: shape (batch_size, num_time_steps, input_length)
    :param config:
    :return: outputs: shape (batch_size, input_length)
    """
    with tf.name_scope("lstm_layers"):
        if config['num_lstm_cells'] > 1:
            lstm_cells = tf.contrib.rnn.MultiRNNCell([create_lstm_cell(config['lstm_size'],
                                                                       dropout_prob=config['dropout_prob'],
                                                                       is_training=is_training)
                                                      for i in range(config['num_lstm_cells'])])
        else:
            lstm_cells = create_lstm_cell(config['lstm_size'], dropout_prob=config['dropout_prob'], is_training=is_training)
        lstm_outputs, lstm_states = tf.nn.dynamic_rnn(lstm_cells, inputs, dtype=tf.float32, scope='lstm_cells')
        lstm_outputs_transposed = tf.transpose(lstm_outputs, [1, 0, 2])

    with tf.name_scope("output_layer"):
        last_time_step = tf.gather(lstm_outputs_transposed, lstm_outputs_transposed.get_shape()[0] - 1, name='last_time_step')

        # FC layer after lstm
        w = tf.get_variable(name="weight", shape=[config['lstm_size'], config['input_length']],
                            dtype=tf.float32, initializer=tf.truncated_normal_initializer(), trainable=True)
        b = tf.get_variable(name="biases", shape=[config['input_length']], dtype=tf.float32,
                            initializer=tf.zeros_initializer(), trainable=True)
        outputs = tf.add(tf.matmul(last_time_step, w), b, name='outputs')
        print("Outputs ", outputs)
    return outputs


if __name__ == "__main__":
    tf.reset_default_graph()
    with tf.Graph().as_default():
        inputs = tf.placeholder(dtype=tf.float32, shape=[None, config['num_time_steps'], config['input_length']],
                                name='inputs')
        targets = tf.placeholder(dtype=tf.float32, shape=[None, config['input_length']], name='targets')
        o = create_model(inputs, config)
        t_vars = tf.trainable_variables()
        reg_loss = tf.reduce_sum([tf.nn.l2_loss(v) for v in t_vars if not 'bias' in v.name])


