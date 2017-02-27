import tensorflow as tf
import functools
from input_data import InputData


def doublewrap(function):
    """
    A decorator decorator,allowing to use the decorator to be used
    without parenthess if not arguments are provided

    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper
from enum import Enum


class HyperParameter(Enum):
    batch_size = 50
    learn_rate = 0.0001
    drop_out = 0.5
    epoch = 50
    num_hidden = 256
    print_iter = 20
    max_length = 75
    num_classes = 28
    data_size = 12996
    use_softmax = 1  # 0 for No 1 for yes
    beam_search = 1  # 1 yes 0 no
    beam_width = 200
    top_paths = 1


class LipNetModel(object):

    def __init__(self, data, target, num_hidden=HyperParameter.num_hidden):
        self._data = data
        self._max_length = HyperParameter.max_length
        self._target = target
        self._weights = None
        self._biases = None
        self._seq_len = [HyperParameter.max_length] * HyperParameter.batch_size
        self._num_hidden = num_hidden
        self.create_weight_and_biases()
        self.prediction
        self.ctc_loss
        self.optimize
        self.decoded

    def create_weight_and_biases(self):
        self._weights = {
            'w1': tf.Variable(tf.random_normal([3, 5, 5, 3, 32])),
            'w2': tf.Variable(tf.random_normal([3, 5, 5, 32, 64])),
            'w3': tf.Variable(tf.random_normal([3, 3, 3, 64, 96]))
        }
        self._biases = {
            'b1': tf.Variable(tf.constant(0.1, shape=[32])),
            'b2': tf.Variable(tf.constant(0.1, shape=[64])),
            'b3': tf.Variable(tf.constant(0.1, shape=[96]))
        }

    def stcnn3x(self, input_data, weights, biases, dropout=0.5):
        padded_data1 = tf.pad(input_data, [[0, 0], [1, 1], [
                              2, 2], [2, 2], [0, 0]], "CONSTANT")
        conv1 = tf.nn.conv3d(padded_data1, weights['w1'], strides=[
                             1, 1, 2, 2, 1], padding='VALID', name='conv1')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, biases['b1']), 'relu1')
        pool1 = tf.nn.max_pool3d(relu1, ksize=[1, 1, 2, 2, 1], strides=[
                                 1, 1, 2, 2, 1], padding='VALID', name='pool1')
        print "========", pool1.get_shape()
        pool1_droped = tf.nn.dropout(pool1, dropout, noise_shape=[
                                     HyperParameter.batch_size, 75, 12, 25, 1])
        padded_data2 = tf.pad(pool1_droped, [[0, 0], [1, 1], [
                              2, 2], [2, 2], [0, 0]], "CONSTANT")
        conv2 = tf.nn.conv3d(padded_data2, weights['w2'], strides=[
                             1, 1, 1, 1, 1], padding='VALID', name='conv2')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, biases['b2']), 'relu2')
        pool2 = tf.nn.max_pool3d(relu2, ksize=[1, 1, 2, 2, 1], strides=[
                                 1, 1, 2, 2, 1], padding='VALID', name='pool2')
        print "========", pool2.get_shape()
        pool2_droped = tf.nn.dropout(pool2, dropout, noise_shape=[
                                     HyperParameter.batch_size, 75, 6, 12, 1])
        padded_data3 = tf.pad(pool2_droped, [[0, 0], [1, 1], [
                              1, 1], [1, 1], [0, 0]], "CONSTANT")
        conv3 = tf.nn.conv3d(padded_data3, weights['w3'], strides=[
                             1, 1, 1, 1, 1], padding='VALID', name='conv3')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, biases['b3']), 'relu3')
        pool3 = tf.nn.max_pool3d(relu3, ksize=[1, 1, 2, 2, 1], strides=[
                                 1, 1, 2, 2, 1], padding='VALID', name='pool3')
        print "========", pool3.get_shape()
        pool3_droped = tf.nn.dropout(pool3, dropout, noise_shape=[
                                     HyperParameter.batch_size, 75, 3, 6, 1])
        return pool3_droped

    def weight_and_bias(self, in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)

    def biGRux2(self, input_sequence):
        gru_cell_fw_1 = tf.nn.rnn_cell.GRUCell(self._num_hidden)
        gru_cell_bw_1 = tf.nn.rnn_cell.GRUCell(self._num_hidden)
        result_tuple_1 = tf.nn.bidirectional_rnn(
            gru_cell_fw_1, gru_cell_bw_1, input_sequence, dtype=tf.float32, scope='BiRNN1')
        gru_cell_fw_2 = tf.nn.rnn_cell.GRUCell(self._num_hidden)
        gru_cell_bw_2 = tf.nn.rnn_cell.GRUCell(self._num_hidden)
        inner_input = result_tuple_1[0]
        result_tuple_2 = tf.nn.bidirectional_rnn(
            gru_cell_fw_2, gru_cell_bw_2, inner_input, dtype=tf.float32, scope='BiRNN2')
        return result_tuple_2[0]

    @lazy_property
    def prediction(self):
        cnn_out_put = self.stcnn3x(
            self._data, self._weights, self._biases, dropout=HyperParameter.drop_out)
        in_put = tf.reshape(cnn_out_put, [-1, 75, 96 * 3 * 6])
        tmajor_tensor = tf.transpose(in_put, [1, 0, 2])
        times = tmajor_tensor.get_shape()[0]
        inputs = [tmajor_tensor[i, :, :] for i in range(times)]
        out_put = self.biGRux2(inputs)
        weight, bias = self.weight_and_bias(self._num_hidden * 2, 28)
        out_put = tf.reshape(out_put, [-1, self._num_hidden * 2])
        if HyperParameter.use_softmax == 1:
            output = tf.nn.softmax(tf.matmul(out_put, weight) + bias)
        prediction = tf.matmul(out_put, weight) + bias
        prediction = tf.reshape(prediction, [-1, self._max_length, 28])
        prediction = tf.transpose(prediction, [1, 0, 2])
        return prediction

    @lazy_property
    def ctc_loss(self):
        loss = tf.nn.ctc_loss(self.prediction, self._target,
                              sequence_length=self._seq_len)
        return tf.reduce_mean(loss)

    @lazy_property
    def optimize(self):
        '''
        this way of learning_rate initialization need change
        '''
        learning_rate = HyperParameter.learn_rate
        optimizer = tf.train.AdamOptimizer(learning_rate)
        return optimizer.minimize(self.ctc_loss)

    @lazy_property
    def decoded(self):
        result = None
        if HyperParameter.beam_search == 1:
            result, _ = tf.nn.ctc_beam_search_decoder(self.prediction, sequence_length=self._seq_len,
                                                      beam_width=HyperParameter.beam_width, top_paths=HyperParameter.top_paths, merge_repeated=True)
        else:
            result, _ = tf.nn.ctc_greedy_decoder(
                self.prediction, sequence_length=self._seq_len, merge_repeated=True)
        return result

    @lazy_property
    def accuracy(self):
        return tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self._target))
if __name__ == '__main__':
    data = tf.placeholder(tf.float32, [None, 75, 50, 100, 3])
    target = tf.sparse_placeholder(tf.int32)
    model = LipNetModel(data, target)
    sess = tf.Session()
    dataInput = InputData()
    initOp = tf.global_variables_initializer()
    sess.run(initOp)
    per_epoch_batches = HyperParameter.data_size / HyperParameter.batch_size
    print "batches_per_epoch", per_epoch_batches
    for epoch in range(HyperParameter.epoch):
        for i in range(per_epoch_batches):
            image_batch, target_batch = dataInput.get_bacth_data(
                HyperParameter.batch_size, per_epoch_batches)
            sess.run(model.optimize, feed_dict={
                     data: image_batch, target: target_batch})
            if i % HyperParameter.print_iter == 0:
                loss = sess.run(model.ctc_loss, feed_dict={
                                data: image_batch, target: target_batch})
                inaccuracy = sess.run(model.accuracy, feed_dict={
                                      data: image_batch, target: target_batch})
                print("eppoch {:2d} iteration {:2d} loss {:.3f} accuracy:{:.3f}".format(
                    epoch, i, loss, 1.0 - inaccuracy))
            d = session.run(self.decoded[0], feed_dict=feed)
            dense_decoded = tf.sparse_tensor_to_dense(
                d, default_value=-1).eval(session=session)
            for i, seq in enumerate(dense_decoded):
                seq = [s for s in seq if s != -1]
                print('Sequence %d' % i)
                print('\t Original:\n%s' % target_batch[i])
                print('\t Decoded:\n%s' % seq)
    sess.close()
