# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn
from tensorflow.python.framework import ops

linear_parent_scope = "linear"
dnn_parent_scope = "dnn"


def add_variables_from_scope(scope, collections):
    train_vars = tf.trainable_variables()
    vars_in_scope = [var for var in train_vars if scope in var.name]
    for var in vars_in_scope:
        for collection in collections:
            print("add {} to {}".format(var.name, collection))
            tf.add_to_collection(collection, var)


# rnn
def rnn_layer(input, mask, rnn_unit, hidden_unit, keep_prob, num_layers, reuse, scope):
    with tf.variable_scope(scope):
        if rnn_unit == 'lstm':
            _cell = rnn.BasicLSTMCell(hidden_unit, forget_bias=1., state_is_tuple=True, reuse=reuse)
        elif rnn_unit == 'gru':
            _cell = rnn.GRUCell(hidden_unit, reuse=reuse)
        else:
            raise ValueError('rnn_unit must in (lstm, gru)!')
        _cells = []
        for _ in range(num_layers):
            _cells.append(tf.nn.rnn_cell.DropoutWrapper(_cell, output_keep_prob=keep_prob))
        cell = tf.nn.rnn_cell.MultiRNNCell(_cells)
        sequence_actual_length = tf.reduce_sum(mask, axis=1, keep_dims=False)
        rnn_outputs, state = tf.nn.dynamic_rnn(
            cell, input, scope=rnn_unit,
            dtype=tf.float32, sequence_length=sequence_actual_length)
        return rnn_outputs, state


# attention_pooling
INF = 1e30


def softmax_mask(val, mask):
    return -INF * (1 - tf.cast(mask, tf.float32)) + val


def attention_pooling(inputs, memory, mask, reuse, scope='pool_attention'):
    """
    :param inputs: pooling feature, dim (batch_size, sequence_dim, vector_dim)
    :param memory: attention query, dim (batch_size, vector_dim)
    :param mask: sequence mask, dim (batch_size, sequence_dim)
    :param hidden: hidden dim
    :param scope: special the layer scope
    :return: pool_attention vector for inputs, dim (batch_size, vector_dim)
    """
    with tf.variable_scope(scope, reuse=reuse):
        JX = tf.shape(inputs)[1]
        memory = tf.tile(tf.expand_dims(memory, axis=1), [1, JX, 1])
        with tf.variable_scope("attention"):
            u = tf.concat([memory, inputs], axis=2)
            u = layers.fully_connected(u, 64, activation_fn=tf.nn.relu, scope='att_dense1',
                                       variables_collections=[dnn_parent_scope])
            u = layers.fully_connected(u, 32, activation_fn=tf.nn.relu, scope='att_dense2',
                                       variables_collections=[dnn_parent_scope])
            s = layers.fully_connected(u, 1, activation_fn=None, scope='att_dense3',
                                       variables_collections=[dnn_parent_scope])
            if mask is not None:
                s = softmax_mask(tf.squeeze(s, [2]), mask)
                a = tf.expand_dims(tf.nn.softmax(s), axis=2)
            else:
                a = tf.expand_dims(tf.nn.softmax(tf.squeeze(s, [2])), axis=2)
            res = tf.squeeze(tf.matmul(inputs, a, transpose_a=True), axis=2)
            return res


def time_attention_pooling(inputs, memory, time, mask, reuse, scope='pool_attention'):
    with tf.variable_scope(scope, reuse=reuse):
        _, JX, dim = inputs.get_shape().as_list()
        with tf.variable_scope("time"):
            time = tf.expand_dims(tf.log(time + 2), axis=2)
            # time = tf.expand_dims(time, axis=2)
            time = layers.fully_connected(time, 8, activation_fn=tf.nn.tanh, scope='att_time',
                                          variables_collections=[dnn_parent_scope])
        memory = tf.tile(tf.expand_dims(memory, axis=1), [1, JX, 1])
        with tf.variable_scope("attention"):
            u = tf.concat([memory, inputs, time], axis=2)
            u = layers.fully_connected(u, 64, activation_fn=tf.nn.relu, scope='att_dense1',
                                       variables_collections=[dnn_parent_scope])
            u = layers.fully_connected(u, 32, activation_fn=tf.nn.relu, scope='att_dense2',
                                       variables_collections=[dnn_parent_scope])
            s = layers.fully_connected(u, 1, activation_fn=None, scope='att_dense3',
                                       variables_collections=[dnn_parent_scope])
            if mask is not None:
                s = softmax_mask(tf.squeeze(s, [2]), mask)
                a = tf.expand_dims(tf.nn.softmax(s), axis=2)
            else:
                a = tf.expand_dims(tf.nn.softmax(tf.squeeze(s, [2])), axis=2)
            res = tf.squeeze(tf.matmul(inputs, a, transpose_a=True), axis=2)
            return res

def select_block(input, name, activation='sigmoid'):
    output = tf.keras.layers.Dense(input.get_shape().as_list()[1], name, use_bias=True, activation=activation)(
        input)
    output = input * output
    return output

def select_block2(input, name, units, activation='sigmoid'):
    output = tf.keras.layers.Dense(units, name, use_bias=True, activation=activation)(
        input)
    return output

class InteractionAwareFG:
    """InteractionAwareFM class"""

    def __init__(self, field_dim, field_num, feature_dim, att_dim):
        self.field_dim = field_dim
        self.field_num = field_num
        self.feature_dim = feature_dim
        self.att_dim = att_dim
        self.activation_fn = tf.nn.relu
        self.tao = 10.0

    def build(self, inputs, reuse, scope):
        with tf.variable_scope(scope, reuse=reuse):
            num_inputs = len(inputs)
            row = []
            col = []
            for i in range(num_inputs - 1):
                for j in range(i + 1, num_inputs):
                    row.append(i)
                    col.append(j)
            with tf.variable_scope("feature_aspect"):
                vi = tf.concat([tf.expand_dims(inputs[idx], axis=1) for idx in row], axis=1)  # batch num_pairs k
                vj = tf.concat([tf.expand_dims(inputs[idx], axis=1) for idx in col], axis=1)
                inner_product = vi * vj  # batch num_pairs k
                att = layers.fully_connected(inner_product, self.att_dim, activation_fn=self.activation_fn,
                                             variables_collections=[dnn_parent_scope])
                att = layers.fully_connected(att, 1, activation_fn=None, biases_initializer=None,
                                             variables_collections=[dnn_parent_scope])
                T = tf.nn.softmax(att * self.tao)  # batch num_pairs 1
                feature_aspect = inner_product * T
            with tf.variable_scope("field_aspect"):
                U = tf.get_variable("U", shape=[self.field_num, self.field_dim], trainable=True,
                                    collections=[dnn_parent_scope, ops.GraphKeys.GLOBAL_VARIABLES,
                                                 ops.GraphKeys.MODEL_VARIABLES])
                row = tf.convert_to_tensor(row)  # [num_pairs]
                col = tf.convert_to_tensor(col)

                ui = tf.nn.embedding_lookup(U, row)  # [num_pairs, field_dim]
                uj = tf.nn.embedding_lookup(U, col)
                field_product = ui * uj  # [num_pairs, field_dim]

                # [num_pairs, k]
                field_aspect = layers.fully_connected(field_product, self.feature_dim, activation_fn=None,
                                                      biases_initializer=None,
                                                      variables_collections=[dnn_parent_scope])
        return feature_aspect * field_aspect  # batch num_pairs k


class FM:
    """FM class"""

    def __init__(self):
        pass

    def build(self, inputs):
        # [batch_size, field_dim, field_num]
        concated_embeds_value = tf.concat([tf.expand_dims(input, axis=1) for input in inputs], axis=1)
        square_of_sum = tf.square(tf.reduce_sum(
            concated_embeds_value, axis=1, keep_dims=True))
        sum_of_square = tf.reduce_sum(
            concated_embeds_value * concated_embeds_value, axis=1, keep_dims=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * tf.reduce_sum(cross_term, axis=2, keep_dims=False)
        return cross_term


class Transformer:
    """Transformer class"""

    def __init__(self,
                 num_heads,
                 key_mask,
                 query_mask,
                 length,
                 embed_dim,
                 linear_key_dim,
                 linear_value_dim,
                 output_dim,
                 hidden_dim,
                 num_layer,
                 keep_prob):
        """
        :param key_mask: mask matrix for key
        :param query_mask: mask matrix for query
        :param num_heads: number of multi-attention head
        :param linear_key_dim: key, query forward dim
        :param linear_value_dim: val forward dim
        :param output_dim: fnn output dim
        :param hidden_dim: fnn hidden dim
        :param num_layer: number of multi-attention layer
        :param keep_prob: keep probability
        """
        assert linear_key_dim % num_heads == 0
        assert linear_value_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.length = length
        self.num_layers = num_layer
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.key_mask = key_mask
        self.query_mask = query_mask
        self.linear_key_dim = linear_key_dim
        self.linear_value_dim = linear_value_dim
        self.keep_prob = keep_prob

    def build(self, inputs, reuse, scope):
        with tf.variable_scope(scope, reuse=reuse):
            # o1 = self._positional_add(inputs)
            o1 = inputs
            for i in range(1, self.num_layers + 1):
                with tf.variable_scope("layer_{}".format(i)):
                    o1_ = self.multi_head(o1, o1, o1, 'multi_head')
                    o2 = self._add_and_norm(o1, o1_, 'norm_1')
                    o2_ = self._positional_feed_forward(o2, self.hidden_dim, self.output_dim, 'forward')
                    o3 = self._add_and_norm(o2, o2_, 'norm_2')
                    o1 = o3
            return o1

    def _positional_add(self, inputs):
        return inputs

    def _positional_feed_forward(self, output, hidden_dim, output_dim, scope):
        with tf.variable_scope(scope):
            output = layers.fully_connected(output, hidden_dim, activation_fn=tf.nn.relu,
                                            variables_collections=[dnn_parent_scope])
            output = layers.fully_connected(output, output_dim, activation_fn=None,
                                            variables_collections=[dnn_parent_scope])
            return tf.nn.dropout(output, self.keep_prob)

    def _add_and_norm(self, x, sub_layer_x, scope):
        with tf.variable_scope(scope):
            return layers.layer_norm(tf.add(x, sub_layer_x), variables_collections=[dnn_parent_scope])

    def multi_head(self, q, k, v, scope):
        with tf.variable_scope(scope):
            q, k, v = self._linear_projection(q, k, v)
            qs, ks, vs = self._split_heads(q, k, v)
            outputs = self._scaled_dot_product(qs, ks, vs)
            output = self._concat_heads(outputs)
            return tf.nn.dropout(output, self.keep_prob)

    def _linear_projection(self, q, k, v):
        q = layers.fully_connected(q, self.linear_key_dim, biases_initializer=None, activation_fn=None,
                                   variables_collections=[dnn_parent_scope])
        k = layers.fully_connected(k, self.linear_key_dim, biases_initializer=None, activation_fn=None,
                                   variables_collections=[dnn_parent_scope])
        v = layers.fully_connected(v, self.linear_value_dim, biases_initializer=None, activation_fn=None,
                                   variables_collections=[dnn_parent_scope])
        return q, k, v

    def _split_heads(self, q, k, v):
        def split_last_dimension_then_transpose(tensor, num_heads, dim):
            t_shape = tensor.get_shape().as_list()
            tensor = tf.reshape(tensor, [-1] + t_shape[1:-1] + [num_heads, dim // num_heads])
            return tf.transpose(tensor, [0, 2, 1, 3])  # [batch_size, num_heads, max_seq_len, dim]

        qs = split_last_dimension_then_transpose(q, self.num_heads, self.linear_key_dim)
        ks = split_last_dimension_then_transpose(k, self.num_heads, self.linear_key_dim)
        vs = split_last_dimension_then_transpose(v, self.num_heads, self.linear_value_dim)
        return qs, ks, vs

    def _scaled_dot_product(self, qs, ks, vs):
        key_dim_per_head = self.linear_key_dim // self.num_heads

        o1 = tf.matmul(qs, ks, transpose_b=True)
        o2 = o1 / (key_dim_per_head ** 0.5)  # (batch_size, num_heads, q_length, k_length)
        if self.key_mask is not None:  # (batch_size, k_length)
            # key mask
            padding_num = -2 ** 32 + 1
            # Generate masks
            mask = tf.expand_dims(self.key_mask, 1)  # (batch_size, 1, k_length)
            mask = tf.tile(mask, [1, self.length, 1])  # (batch_size, q_length, k_length)
            mask = tf.expand_dims(mask, 1)
            mask = tf.tile(mask, [1, self.num_heads, 1, 1])
            # Apply masks to inputs
            paddings = tf.ones_like(o2) * padding_num
            o2 = tf.where(tf.equal(mask, 0), paddings, o2)  # (batch_size, num_heads, q_length, k_length)
        o3 = tf.nn.softmax(o2)

        if self.query_mask is not None:
            mask = tf.expand_dims(self.query_mask, 2)  # (batch_size, q_length, 1)
            mask = tf.expand_dims(mask, 1)
            mask = tf.tile(mask, [1, self.num_heads, 1, 1])  # (batch_size, num_heads, q_length, 1)
            o3 = o3 * tf.cast(mask, tf.float32)  # broadcast
        return tf.matmul(o3, vs)

    def _concat_heads(self, outputs):
        def transpose_then_concat_last_two_dimenstion(tensor):
            tensor = tf.transpose(tensor, [0, 2, 1, 3])  # [batch_size, max_seq_len, num_heads, dim]
            t_shape = tensor.get_shape().as_list()
            num_heads, dim = t_shape[-2:]
            return tf.reshape(tensor, [-1] + t_shape[1:-2] + [num_heads * dim])

        return transpose_then_concat_last_two_dimenstion(outputs)


class SelfAttentionPooling:
    """SelfAttentionPooling class"""

    def __init__(self,
                 num_heads,
                 key_mask,
                 query_mask,
                 length,
                 linear_key_dim,
                 linear_value_dim,
                 output_dim,
                 hidden_dim,
                 num_layer,
                 keep_prob):
        """
        :param key_mask: mask matrix for key
        :param query_mask: mask matrix for query
        :param num_heads: number of multi-attention head
        :param linear_key_dim: key, query forward dim
        :param linear_value_dim: val forward dim
        :param output_dim: fnn output dim
        :param hidden_dim: fnn hidden dim
        :param num_layer: number of multi-attention layer
        :param keep_prob: keep probability
        """
        assert linear_key_dim % num_heads == 0
        assert linear_value_dim % num_heads == 0

        self.length = length
        self.num_layers = num_layer
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.key_mask = key_mask
        self.query_mask = query_mask
        self.linear_key_dim = linear_key_dim
        self.linear_value_dim = linear_value_dim
        self.keep_prob = keep_prob

    def build(self, inputs, reuse, scope):
        with tf.variable_scope(scope, reuse=reuse):
            # o1 = self._positional_add(inputs)
            o1 = inputs
            for i in range(1, self.num_layers + 1):
                with tf.variable_scope("layer_{}".format(i)):
                    o1_ = self.multi_head(o1, o1, o1, 'multi_head')
                    o2 = self._add_and_norm(o1, o1_, 'norm_1')
                    o2_ = self._positional_feed_forward(o2, self.hidden_dim, self.output_dim, 'forward')
                    o3 = self._add_and_norm(o2, o2_, 'norm_2')
                    o1 = o3
            return o1

    def _positional_feed_forward(self, output, hidden_dim, output_dim, scope):
        with tf.variable_scope(scope):
            output = layers.fully_connected(output, hidden_dim, activation_fn=tf.nn.relu,
                                            variables_collections=[dnn_parent_scope])
            output = layers.fully_connected(output, output_dim, activation_fn=None,
                                            variables_collections=[dnn_parent_scope])
            return tf.nn.dropout(output, self.keep_prob)

    def _add_and_norm(self, x, sub_layer_x, scope):
        with tf.variable_scope(scope):
            return layers.layer_norm(tf.add(x, sub_layer_x), variables_collections=[dnn_parent_scope])

    def multi_head(self, q, k, v, scope):
        with tf.variable_scope(scope):
            q, k, v = self._linear_projection(q, k, v)
            qs, ks, vs = self._split_heads(q, k, v)
            outputs = self._scaled_dot_product(qs, ks, vs)
            output = self._concat_heads(outputs)
            return tf.nn.dropout(output, self.keep_prob)

    def _linear_projection(self, q, k, v):
        q = layers.fully_connected(q, self.linear_key_dim, biases_initializer=None, activation_fn=None,
                                   variables_collections=[dnn_parent_scope])
        k = layers.fully_connected(k, self.linear_key_dim, biases_initializer=None, activation_fn=None,
                                   variables_collections=[dnn_parent_scope])
        v = layers.fully_connected(v, self.linear_value_dim, biases_initializer=None, activation_fn=None,
                                   variables_collections=[dnn_parent_scope])
        return q, k, v

    def _split_heads(self, q, k, v):
        def split_last_dimension_then_transpose(tensor, num_heads, dim):
            t_shape = tensor.get_shape().as_list()
            tensor = tf.reshape(tensor, [-1] + t_shape[1:-1] + [num_heads, dim // num_heads])
            return tf.transpose(tensor, [0, 2, 1, 3])  # [batch_size, num_heads, max_seq_len, dim]

        qs = split_last_dimension_then_transpose(q, self.num_heads, self.linear_key_dim)
        ks = split_last_dimension_then_transpose(k, self.num_heads, self.linear_key_dim)
        vs = split_last_dimension_then_transpose(v, self.num_heads, self.linear_value_dim)
        return qs, ks, vs

    def _scaled_dot_product(self, qs, ks, vs):
        key_dim_per_head = self.linear_key_dim // self.num_heads

        o1 = tf.matmul(qs, ks, transpose_b=True)
        o2 = o1 / (key_dim_per_head ** 0.5)  # (batch_size, num_heads, q_length, k_length)
        if self.key_mask is not None:  # (batch_size, k_length)
            # key mask
            padding_num = -2 ** 32 + 1
            # Generate masks
            mask = tf.expand_dims(self.key_mask, 1)  # (batch_size, 1, k_length)
            mask = tf.tile(mask, [1, self.length, 1])  # (batch_size, q_length, k_length)
            mask = tf.expand_dims(mask, 1)
            mask = tf.tile(mask, [1, self.num_heads, 1, 1])
            # Apply masks to inputs
            paddings = tf.ones_like(o2) * padding_num
            o2 = tf.where(tf.equal(mask, 0), paddings, o2)  # (batch_size, num_heads, q_length, k_length)
        o3 = tf.nn.softmax(o2)

        if self.query_mask is not None:
            mask = tf.expand_dims(self.query_mask, 2)  # (batch_size, q_length, 1)
            mask = tf.expand_dims(mask, 1)
            mask = tf.tile(mask, [1, self.num_heads, 1, 1])  # (batch_size, num_heads, q_length, 1)
            o3 = o3 * tf.cast(mask, tf.float32)  # broadcast
        return tf.matmul(o3, vs)

    def _concat_heads(self, outputs):
        def transpose_then_concat_last_two_dimenstion(tensor):
            tensor = tf.transpose(tensor, [0, 2, 1, 3])  # [batch_size, max_seq_len, num_heads, dim]
            t_shape = tensor.get_shape().as_list()
            num_heads, dim = t_shape[-2:]
            return tf.reshape(tensor, [-1] + t_shape[1:-2] + [num_heads * dim])

        return transpose_then_concat_last_two_dimenstion(outputs)


class FGCNN:
    """FGCNN class"""

    def __init__(self, is_training, keep_prob):
        self.filters = [5, 5, 5, 5]
        self.num_outputs = [6, 8, 10, 12]
        self.news = [2, 2, 2, 2]
        self.pool_sizes = [2, 2, 2, 2]
        self.pool_strides = [2, 2, 2, 2]
        self.activation_fn = tf.nn.relu
        self.is_training = is_training
        self.keep_prob = keep_prob

    def build(self, inputs, reuse, scope):
        # (batch, field_num, embed_dim, 1)
        gen_features = []
        inputs = tf.expand_dims(inputs, axis=3)
        o1 = inputs
        with tf.variable_scope(scope, reuse=reuse):
            for i in range(len(self.filters)):
                o1_ = self.combine(o1, i)  # conv and pool
                o2 = self.recombine(o1_, i)  # mlp and reshape
                gen_features.append(o2)
                o1 = o1_
        gen_features_res = tf.concat(gen_features, axis=1)
        return gen_features_res

    def recombine(self, inputs, layer_id):
        new = self.news[layer_id]
        _, field, k, num_output = inputs.get_shape().as_list()
        input_dim = field * k * num_output
        output_dim = field * k * new
        # (batch, field_num * embed_dim * num_output)
        inputs = tf.reshape(inputs, [-1, input_dim])
        inputs = layers.fully_connected(inputs, output_dim, activation_fn=None,
                                        variables_collections=[dnn_parent_scope])
        inputs = layers.batch_norm(inputs, is_training=self.is_training, activation_fn=self.activation_fn,
                                   variables_collections=[dnn_parent_scope])
        inputs = tf.reshape(inputs, [-1, field * new, k])
        return inputs

    def combine(self, inputs, layer_id):
        filter = self.filters[layer_id]
        num_output = self.num_outputs[layer_id]
        pool_size = self.pool_sizes[layer_id]
        pool_strides = self.pool_strides[layer_id]
        # (batch, field_num, embed_dim, num_output)
        cnn_output = layers.conv2d(inputs, num_outputs=num_output, kernel_size=[filter, 1], stride=[1, 1],
                                   padding='SAME',
                                   activation_fn=None, scope="cnn_{}".format(layer_id),
                                   variables_collections=[dnn_parent_scope])
        cnn_output = layers.batch_norm(cnn_output, is_training=self.is_training, activation_fn=self.activation_fn,
                                       variables_collections=[dnn_parent_scope])
        # (batch, field_num/pool_size, embed_dim, num_output)
        pool_output = layers.max_pool2d(cnn_output, kernel_size=[pool_size, 1], stride=[pool_strides, 1])
        return pool_output


class Autoint:
    """Autoint class"""

    def __init__(self,
                 num_heads,
                 linear_dim,
                 num_layer,
                 is_training,
                 keep_prob):
        assert linear_dim % num_heads == 0

        self.num_layers = num_layer
        self.num_heads = num_heads
        self.linear_dim = linear_dim
        self.is_training = is_training
        self.keep_prob = keep_prob

    def build(self, inputs, reuse, scope):
        with tf.variable_scope(scope, reuse=reuse):
            # o1 = self._positional_add(inputs)
            o1 = inputs
            for i in range(1, self.num_layers + 1):
                with tf.variable_scope("layer_{}".format(i)):
                    o1_ = self.multi_head(o1, o1, o1, 'multi_head')
                    o1 = self._positional_feed_forward(o1, o1_, 'forward')
            return o1

    def _positional_feed_forward(self, before, after, scope):
        with tf.variable_scope(scope):
            before = layers.fully_connected(before, self.linear_dim, biases_initializer=None, activation_fn=None,
                                            variables_collections=[dnn_parent_scope])
            output = tf.add(before, after)
            output = layers.batch_norm(output, is_training=self.is_training, activation_fn=tf.nn.relu,
                                       variables_collections=[dnn_parent_scope])

            # output = tf.nn.relu(tf.add(before, after))
            return tf.nn.dropout(output, self.keep_prob)

    def multi_head(self, q, k, v, scope):
        with tf.variable_scope(scope):
            q, k, v = self._linear_projection(q, k, v)
            qs, ks, vs = self._split_heads(q, k, v)
            outputs = self._scaled_dot_product(qs, ks, vs)
            output = self._concat_heads(outputs)
            return tf.nn.dropout(output, self.keep_prob)

    def _linear_projection(self, q, k, v):
        q = layers.fully_connected(q, self.linear_dim, biases_initializer=None, activation_fn=None,
                                   variables_collections=[dnn_parent_scope])
        k = layers.fully_connected(k, self.linear_dim, biases_initializer=None, activation_fn=None,
                                   variables_collections=[dnn_parent_scope])
        v = layers.fully_connected(v, self.linear_dim, biases_initializer=None, activation_fn=None,
                                   variables_collections=[dnn_parent_scope])
        return q, k, v

    def _split_heads(self, q, k, v):
        def split_last_dimension_then_transpose(tensor, num_heads, dim):
            t_shape = tensor.get_shape().as_list()
            tensor = tf.reshape(tensor, [-1] + t_shape[1:-1] + [num_heads, dim // num_heads])
            return tf.transpose(tensor, [0, 2, 1, 3])  # [batch_size, num_heads, max_seq_len, dim]

        qs = split_last_dimension_then_transpose(q, self.num_heads, self.linear_dim)
        ks = split_last_dimension_then_transpose(k, self.num_heads, self.linear_dim)
        vs = split_last_dimension_then_transpose(v, self.num_heads, self.linear_dim)
        return qs, ks, vs

    def _scaled_dot_product(self, qs, ks, vs):
        o1 = tf.matmul(qs, ks, transpose_b=True)
        o2 = tf.nn.softmax(o1)
        return tf.matmul(o2, vs)

    def _concat_heads(self, outputs):
        def transpose_then_concat_last_two_dimenstion(tensor):
            tensor = tf.transpose(tensor, [0, 2, 1, 3])  # [batch_size, max_seq_len, num_heads, dim]
            t_shape = tensor.get_shape().as_list()
            num_heads, dim = t_shape[-2:]
            return tf.reshape(tensor, [-1] + t_shape[1:-2] + [num_heads * dim])

        return transpose_then_concat_last_two_dimenstion(outputs)


class DCN:
    """Deep Cross Network class"""

    def __init__(self,
                 num_layer,
                 is_training,
                 keep_prob):
        self.num_layers = num_layer
        self.is_training = is_training
        self.keep_prob = keep_prob
        self.activation_fn = tf.nn.relu

    def build(self, inputs, reuse, scope):
        with tf.variable_scope(scope, reuse=reuse):
            _, dim = inputs.get_shape().as_list()
            inputs = layers.batch_norm(inputs, is_training=self.is_training, activation_fn=None,
                                       variables_collections=[dnn_parent_scope])
            x_0 = tf.expand_dims(inputs, axis=2)
            cross = x_0
            for i in range(self.num_layers):
                kernel = tf.get_variable("kernel_{}".format(i),
                                         shape=[dim, 1],
                                         collections=[dnn_parent_scope, ops.GraphKeys.GLOBAL_VARIABLES,
                                                      ops.GraphKeys.MODEL_VARIABLES],
                                         initializer=tf.glorot_uniform_initializer())
                bias = tf.get_variable("bias_{}".format(i),
                                       shape=[dim, 1],
                                       collections=[dnn_parent_scope, ops.GraphKeys.GLOBAL_VARIABLES,
                                                    ops.GraphKeys.MODEL_VARIABLES],
                                       initializer=tf.zeros_initializer())
                # (batch_size dim 1) * (dim 1) -> (batch_size, 1, 1)
                cross = tf.matmul(x_0, tf.tensordot(cross, kernel, axes=(1, 0))) + bias + cross
                # (batch_size dim 1)
                cross = tf.squeeze(cross, axis=2)
                cross = layers.batch_norm(cross, is_training=self.is_training, activation_fn=self.activation_fn,
                                          variables_collections=[dnn_parent_scope])
                cross = tf.expand_dims(cross, axis=2)
            cross = tf.squeeze(cross, axis=2)
        return cross


class PNN:
    """Product-based Neural Network class"""

    def __init__(self,
                 embedding_dim,
                 field_dim,
                 output_dim):
        self.embedding_dim = embedding_dim
        self.field_dim = field_dim
        self.output_dim = output_dim

    def build(self, inputs, reuse, scope):
        with tf.variable_scope(scope, reuse=reuse):
            with tf.variable_scope("pnn_model_variable"):
                linear_weights = tf.get_variable("linear_weights",
                                                 shape=[self.embedding_dim * self.field_dim, self.output_dim],
                                                 collections=[dnn_parent_scope, ops.GraphKeys.GLOBAL_VARIABLES,
                                                              ops.GraphKeys.MODEL_VARIABLES],
                                                 initializer=layers.xavier_initializer())

                product_weights = tf.get_variable("product_weights",
                                                  shape=[self.field_dim, self.output_dim],
                                                  collections=[dnn_parent_scope, ops.GraphKeys.GLOBAL_VARIABLES,
                                                               ops.GraphKeys.MODEL_VARIABLES],
                                                  initializer=layers.xavier_initializer())

                product_biases = tf.get_variable("product_biases",
                                                 shape=[self.output_dim],
                                                 collections=[dnn_parent_scope, ops.GraphKeys.GLOBAL_VARIABLES,
                                                              ops.GraphKeys.MODEL_VARIABLES],
                                                 initializer=tf.zeros_initializer())
            with tf.variable_scope("pnn_product_layer"):
                # multiply -> matmul
                inputs_linear = tf.reshape(inputs, [-1, self.embedding_dim * self.field_dim])
                linear_part = tf.matmul(inputs_linear, linear_weights)

                inputs_product = tf.transpose(inputs, [0, 2, 1])
                inputs_product = tf.reshape(inputs_product, [-1, self.field_dim])
                product_part = tf.matmul(inputs_product, product_weights)
                product_part = tf.reshape(product_part, [-1, self.embedding_dim, self.output_dim])
                product_part = tf.reduce_sum(product_part, axis=1)
                pnn_part = linear_part + product_part + product_biases
        return pnn_part
