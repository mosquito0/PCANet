# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn
from tensorflow.contrib.layers.python.layers import optimizers
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import state_ops
from utils import *
from fg_worker import FeatureColumnGenerator

import math
import os
import time
import rtp_fg
import json
import copy

linear_parent_scope = "linear"
dnn_parent_scope = "dnn"

def model_block(rid, features, label_1, label_2, label_3, avg_p, avg_q, group_p, group_q, price, group_avg_price, group_avg_jy, group_price, fc_generator, is_training, keep_prob,
                params):

    tf.logging.info("building features...")
    outputs_dict = fc_generator.get_output_dict(features)

    tf.logging.info("finished build features:")
    for key in outputs_dict:
        tf.logging.info(key)
        tf.logging.info(outputs_dict[key])

    tf.logging.info("building dense features:")
    dense_feats = []
    for key in outputs_dict:
        if "is_dense" in key:
            tf.logging.info(key)
            dense_feats.append((key, outputs_dict[key]))

    tf.logging.info("building wide_dense features:")
    wide_dense_feats = []
    group_dense_feats = []
    for key in outputs_dict:
        if "is_wide" in key:
            tf.logging.info(key)
            wide_dense_feats.append((key, outputs_dict[key]))
            if key in ("is_wide_f108", "is_wide_f112", "is_wide_f115", "is_wide_f117", "is_wide_f118", "is_wide_f119",
                       "is_wide_f120", "is_wide_f121", "is_wide_f123"):
                tf.logging.info("group feats: {}".format(key))
                group_dense_feats.append((key, outputs_dict[key]))

    wide_dense_feats = [feat for _, feat in sorted(wide_dense_feats, key=lambda x: x[0])]
    group_dense_feats = [feat for _, feat in sorted(group_dense_feats, key=lambda x: x[0])]

    tf.logging.info("building deep features:")
    rid_ext_feats = []
    for key in outputs_dict:
        if "rid_ext" in key:
            tf.logging.info(key)
            rid_ext_feats.append((key, outputs_dict[key]))

    rid_ext_feats = [feat for _, feat in sorted(rid_ext_feats, key=lambda x: x[0])]
    group_ext_feats = []
    for key in outputs_dict:
        if "group_ext" in key:
            tf.logging.info(key)
            group_ext_feats.append((key, outputs_dict[key]))

    group_ext_feats = [feat for _, feat in sorted(group_ext_feats, key=lambda x: x[0])]

    ################# rid series
    rid_ord_uv_list = outputs_dict["rid_ord_uv_list"]
    rid_jy_list = outputs_dict["rid_jy_list"]
    rid_ord_price_list = outputs_dict["rid_ord_price_list"]
    rid_ipv_uv_list = outputs_dict["rid_ipv_uv_list"]
    rid_search_uv_list = outputs_dict["rid_search_uv_list"]
    rid_sale_price_list = outputs_dict["rid_sale_price_list"]
    rid_vec = outputs_dict["rid_vec"]
    rid_price_vec = outputs_dict["rid_price_vec"]

    group_rid_ord_uv_list = outputs_dict["group_rid_ord_uv_list"]
    group_rid_jy_list = outputs_dict["group_rid_jy_list"]
    group_rid_ord_price_list = outputs_dict["group_rid_ord_price_list"]
    group_rid_ipv_uv_list = outputs_dict["group_rid_ipv_uv_list"]
    group_rid_search_uv_list = outputs_dict["group_rid_search_uv_list"]
    group_rid_sale_price_list = outputs_dict["group_rid_sale_price_list"]
    group_rid_ord_uv_list = tf.expand_dims(group_rid_ord_uv_list, axis=2)
    group_rid_jy_list = tf.expand_dims(group_rid_jy_list, axis=2)
    group_rid_ord_price_list = tf.expand_dims(group_rid_ord_price_list, axis=2)
    group_rid_ipv_uv_list = tf.expand_dims(group_rid_ipv_uv_list, axis=2)
    group_rid_search_uv_list = tf.expand_dims(group_rid_search_uv_list, axis=2)
    group_rid_sale_price_list = tf.expand_dims(group_rid_sale_price_list, axis=2)

    rid_ord_uv_list = tf.expand_dims(rid_ord_uv_list, axis=2)
    rid_jy_list = tf.expand_dims(rid_jy_list, axis=2)
    rid_ord_price_list = tf.expand_dims(rid_ord_price_list, axis=2)
    rid_ipv_uv_list = tf.expand_dims(rid_ipv_uv_list, axis=2)
    rid_search_uv_list = tf.expand_dims(rid_search_uv_list, axis=2)
    rid_sale_price_list = tf.expand_dims(rid_sale_price_list, axis=2)
    rid_internal_pre = tf.expand_dims(rid_vec, axis=1)
    rid_price_internal_pre = tf.expand_dims(rid_price_vec, axis=1)

    rid_series_m = tf.concat(
        [rid_ord_uv_list, rid_jy_list, rid_ord_price_list, rid_ipv_uv_list, rid_search_uv_list, rid_sale_price_list],
        axis=2)
    print('rid_series_m:', rid_series_m.get_shape())
    group_rid_series = tf.concat(
        [group_rid_ord_uv_list, group_rid_jy_list, group_rid_ord_price_list, group_rid_ipv_uv_list,
         group_rid_search_uv_list, group_rid_sale_price_list],
        axis=2)
    #################
    #### rid_day
    rid_jy_day = outputs_dict["shared_rid_jy_day"]
    rid_jy_day = tf.concat([tf.expand_dims(id, axis=1) for id in rid_jy_day], axis=1)

    rid_pv_day = outputs_dict["shared_rid_pv_day"]
    rid_pv_day = tf.concat([tf.expand_dims(id, axis=1) for id in rid_pv_day], axis=1)

    rid_uv_day = outputs_dict["shared_rid_uv_day"]
    rid_uv_day = tf.concat([tf.expand_dims(id, axis=1) for id in rid_uv_day], axis=1)

    rid_search_uv_day = outputs_dict["shared_rid_search_uv_day"]
    rid_search_uv_day = tf.concat([tf.expand_dims(id, axis=1) for id in rid_search_uv_day], axis=1)

    rid_day_seq = tf.concat(
        [rid_jy_day, rid_uv_day, rid_pv_day, rid_search_uv_day], axis=2)

    rid_day_seq = tf.reduce_mean(rid_day_seq, axis=1)

    #### rid_week
    rid_jy_week = outputs_dict["shared_rid_jy_week"]
    rid_jy_week = tf.concat([tf.expand_dims(id, axis=1) for id in rid_jy_week], axis=1)

    rid_pv_week = outputs_dict["shared_rid_pv_week"]
    rid_pv_week = tf.concat([tf.expand_dims(id, axis=1) for id in rid_pv_week], axis=1)

    rid_uv_week = outputs_dict["shared_rid_uv_week"]
    rid_uv_week = tf.concat([tf.expand_dims(id, axis=1) for id in rid_uv_week], axis=1)

    rid_search_uv_week = outputs_dict["shared_rid_search_uv_week"]
    rid_search_uv_week = tf.concat([tf.expand_dims(id, axis=1) for id in rid_search_uv_week], axis=1)

    rid_week_seq = tf.concat(
        [rid_jy_week, rid_uv_week, rid_pv_week, rid_search_uv_week], axis=2)

    rid_week_seq = tf.reduce_mean(rid_week_seq, axis=1)

    #### rid_month
    rid_jy_month = outputs_dict["shared_rid_jy_month"]
    rid_jy_month = tf.concat([tf.expand_dims(id, axis=1) for id in rid_jy_month], axis=1)

    rid_pv_month = outputs_dict["shared_rid_pv_month"]
    rid_pv_month = tf.concat([tf.expand_dims(id, axis=1) for id in rid_pv_month], axis=1)

    rid_uv_month = outputs_dict["shared_rid_uv_month"]
    rid_uv_month = tf.concat([tf.expand_dims(id, axis=1) for id in rid_uv_month], axis=1)

    rid_search_uv_month = outputs_dict["shared_rid_search_uv_month"]
    rid_search_uv_month = tf.concat([tf.expand_dims(id, axis=1) for id in rid_search_uv_month], axis=1)

    rid_month_seq = tf.concat(
        [rid_jy_month, rid_uv_month, rid_pv_month, rid_search_uv_month], axis=2)

    rid_month_seq = tf.reduce_mean(rid_month_seq, axis=1)

    rid_day_query = []
    rid_week_query = []
    rid_month_query = []
    rid_day_key = []
    rid_week_key = []
    rid_month_key = []
    for key in outputs_dict:
        if key in ["rid_day_key"]:
            rid_day_key.append((key, outputs_dict[key]))
        if key in ["rid_week_key"]:
            rid_week_key.append((key, outputs_dict[key]))
        if key in ["rid_month_key"]:
            rid_month_key.append((key, outputs_dict[key]))
        if key in ["rid_day_query"]:
            rid_day_query.append((key, outputs_dict[key]))
        if key in ["rid_week_query"]:
            rid_week_query.append((key, outputs_dict[key]))
        if key in ["rid_month_query"]:
            rid_month_query.append((key, outputs_dict[key]))
    rid_day_query = [feat for _, feat in sorted(rid_day_query, key=lambda x: x[0])]
    rid_week_query = [feat for _, feat in sorted(rid_week_query, key=lambda x: x[0])]
    rid_month_query = [feat for _, feat in sorted(rid_month_query, key=lambda x: x[0])]
    rid_day_key = [feat for _, feat in sorted(rid_day_key, key=lambda x: x[0])]
    rid_week_key = [feat for _, feat in sorted(rid_week_key, key=lambda x: x[0])]
    rid_month_key = [feat for _, feat in sorted(rid_month_key, key=lambda x: x[0])]
    rid_day_query = tf.concat(rid_day_query, axis=1)
    rid_week_query = tf.concat(rid_week_query, axis=1)
    rid_month_query = tf.concat(rid_month_query, axis=1)
    rid_day_key = tf.concat(rid_day_key, axis=1)
    rid_week_key = tf.concat(rid_week_key, axis=1)
    rid_month_key = tf.concat(rid_month_key, axis=1)

    #### group_rid_day
    group_rid_jy_day = outputs_dict["shared_group_rid_jy_day"]
    group_rid_jy_day = tf.concat([tf.expand_dims(id, axis=1) for id in group_rid_jy_day], axis=1)

    group_rid_pv_day = outputs_dict["shared_group_rid_pv_day"]
    group_rid_pv_day = tf.concat([tf.expand_dims(id, axis=1) for id in group_rid_pv_day], axis=1)

    group_rid_uv_day = outputs_dict["shared_group_rid_uv_day"]
    group_rid_uv_day = tf.concat([tf.expand_dims(id, axis=1) for id in group_rid_uv_day], axis=1)

    group_rid_search_uv_day = outputs_dict["shared_group_rid_search_uv_day"]
    group_rid_search_uv_day = tf.concat([tf.expand_dims(id, axis=1) for id in group_rid_search_uv_day], axis=1)

    group_rid_day_seq = tf.concat(
        [group_rid_jy_day, group_rid_uv_day, group_rid_pv_day, group_rid_search_uv_day], axis=2)

    group_day_seq = tf.reduce_mean(group_rid_day_seq, axis=1)

    #### group_rid_week
    group_rid_jy_week = outputs_dict["shared_group_rid_jy_week"]
    group_rid_jy_week = tf.concat([tf.expand_dims(id, axis=1) for id in group_rid_jy_week], axis=1)

    group_rid_pv_week = outputs_dict["shared_group_rid_pv_week"]
    group_rid_pv_week = tf.concat([tf.expand_dims(id, axis=1) for id in group_rid_pv_week], axis=1)

    group_rid_uv_week = outputs_dict["shared_group_rid_uv_week"]
    group_rid_uv_week = tf.concat([tf.expand_dims(id, axis=1) for id in group_rid_uv_week], axis=1)

    group_rid_search_uv_week = outputs_dict["shared_group_rid_search_uv_week"]
    group_rid_search_uv_week = tf.concat([tf.expand_dims(id, axis=1) for id in group_rid_search_uv_week], axis=1)

    group_rid_week_seq = tf.concat(
        [group_rid_jy_week, group_rid_uv_week, group_rid_pv_week, group_rid_search_uv_week], axis=2)

    group_week_seq = tf.reduce_mean(group_rid_week_seq, axis=1)

    #### group_rid_month
    group_rid_jy_month = outputs_dict["shared_group_rid_jy_month"]
    group_rid_jy_month = tf.concat([tf.expand_dims(id, axis=1) for id in group_rid_jy_month], axis=1)

    group_rid_pv_month = outputs_dict["shared_group_rid_pv_month"]
    group_rid_pv_month = tf.concat([tf.expand_dims(id, axis=1) for id in group_rid_pv_month], axis=1)

    group_rid_uv_month = outputs_dict["shared_group_rid_uv_month"]
    group_rid_uv_month = tf.concat([tf.expand_dims(id, axis=1) for id in group_rid_uv_month], axis=1)

    group_rid_search_uv_month = outputs_dict["shared_group_rid_search_uv_month"]
    group_rid_search_uv_month = tf.concat([tf.expand_dims(id, axis=1) for id in group_rid_search_uv_month], axis=1)

    group_rid_month_seq = tf.concat(
        [group_rid_jy_month, group_rid_uv_month, group_rid_pv_month, group_rid_search_uv_month], axis=2)
    group_month_seq = tf.reduce_mean(group_rid_month_seq, axis=1)

    group_day_query = []
    group_week_query = []
    group_month_query = []
    group_day_key = []
    group_week_key = []
    group_month_key = []
    for key in outputs_dict:
        if key in ["group_day_key"]:
            group_day_key.append((key, outputs_dict[key]))
        if key in ["group_week_key"]:
            group_week_key.append((key, outputs_dict[key]))
        if key in ["group_month_key"]:
            group_month_key.append((key, outputs_dict[key]))
        if key in ["group_day_query"]:
            group_day_query.append((key, outputs_dict[key]))
        if key in ["group_week_query"]:
            group_week_query.append((key, outputs_dict[key]))
        if key in ["group_month_query"]:
            group_month_query.append((key, outputs_dict[key]))
    group_day_query = [feat for _, feat in sorted(group_day_query, key=lambda x: x[0])]
    group_week_query = [feat for _, feat in sorted(group_week_query, key=lambda x: x[0])]
    group_month_query = [feat for _, feat in sorted(group_month_query, key=lambda x: x[0])]
    group_day_key = [feat for _, feat in sorted(group_day_key, key=lambda x: x[0])]
    group_week_key = [feat for _, feat in sorted(group_week_key, key=lambda x: x[0])]
    group_month_key = [feat for _, feat in sorted(group_month_key, key=lambda x: x[0])]
    group_day_query = tf.concat(group_day_query, axis=1)
    group_week_query = tf.concat(group_week_query, axis=1)
    group_month_query = tf.concat(group_month_query, axis=1)
    group_day_key = tf.concat(group_day_key, axis=1)
    group_week_key = tf.concat(group_week_key, axis=1)
    group_month_key = tf.concat(group_month_key, axis=1)

    #################

    group_ela = label_1
    jy_label = label_2
    rid_ela = label_3
    rid_s = rid
    group_ela = tf.string_to_number(group_ela, out_type=tf.float32, name=None)
    jy_label = tf.string_to_number(jy_label, out_type=tf.float32, name=None)
    rid_ela = tf.string_to_number(rid_ela, out_type=tf.float32, name=None)
    p_0 = tf.string_to_number(avg_p, out_type=tf.float32, name=None)
    q_0 = tf.string_to_number(avg_q, out_type=tf.float32, name=None)
    group_p = tf.string_to_number(group_p, out_type=tf.float32, name=None)
    group_q = tf.string_to_number(group_q, out_type=tf.float32, name=None)
    group_avg_price = tf.string_to_number(group_avg_price, out_type=tf.float32, name=None)
    group_avg_jy = tf.string_to_number(group_avg_jy, out_type=tf.float32, name=None)
    group_price = tf.string_to_number(group_price, out_type=tf.float32, name=None)
    sale_p = tf.string_to_number(price, out_type=tf.float32, name=None)
    rid_s = tf.string_to_number(rid_s, out_type=tf.int64, name=None)

    group_ela = tf.reshape(group_ela, [-1, 1])
    jy_label = tf.reshape(jy_label, [-1, 1])
    rid_ela = tf.reshape(rid_ela, [-1, 1])
    p_0 = tf.reshape(p_0, [-1, 1])
    group_avg_price = tf.reshape(group_avg_price, [-1, 1])
    group_avg_jy = tf.reshape(group_avg_jy, [-1, 1])
    group_price = tf.reshape(group_price, [-1, 1])
    group_p = tf.reshape(group_p, [-1, 1])
    group_q = tf.reshape(group_q, [-1, 1])
    q_0 = tf.reshape(q_0, [-1, 1])
    sale_p = tf.reshape(sale_p, [-1, 1])
    rid_s = tf.reshape(rid_s, [-1, 1])

    activation_fn = tf.nn.relu6
    # activation_fn = tf.nn.selu
    # activation_fn = tf.nn.softmax
    # activation_fn = tf.nn.leaky_relu
    tf.keras.backend.set_learning_phase(is_training)

    rid_ext_repre = FM().build(rid_ext_feats)
    rid_ext_repre = select_block(rid_ext_repre, "rid_external_sb")
    group_ext_feats = FM().build(group_ext_feats)
    group_ext_feats = select_block(group_ext_feats, "group_external_sb")

    group_static_fea = tf.concat(group_dense_feats, axis=1)
    rid_static_fea = tf.concat(wide_dense_feats, axis=1)

    # ### PRM
    rid_cnn_output = tf.keras.layers.Conv1D(64, 3, strides=1, activation="relu", use_bias=True, padding='same',
                                            input_shape=(30, 6))(rid_series_m)
    rid_cnn_output = tf.keras.layers.Conv1D(128, 3, strides=1, activation="relu", use_bias=True, padding='same')(
        rid_cnn_output)
    rid_cnn_output = tf.keras.layers.Conv1D(64, 3, strides=1, activation="relu", use_bias=True, padding='same')(
        rid_cnn_output)
    rid_cnn_output = keras.layers.MaxPooling1D(pool_size=2, strides=3, padding="valid")(rid_cnn_output)
    rid_cnn_output = tf.keras.layers.Flatten()(rid_cnn_output)

    rid_day_query = tf.expand_dims(rid_day_query, axis=2)
    rid_day_att = tf.nn.softmax(tf.matmul(rid_day_key, rid_day_query))
    rid_day_rep = tf.reduce_sum(tf.multiply(rid_day_seq, rid_day_att), axis=1)
    rid_week_query = tf.expand_dims(rid_week_query, axis=2)
    rid_week_att = tf.nn.softmax(tf.matmul(rid_week_key, rid_week_query))
    rid_week_rep = tf.reduce_sum(tf.multiply(rid_week_seq, rid_week_att), axis=1)
    rid_month_query = tf.expand_dims(rid_month_query, axis=2)
    rid_month_att = tf.nn.softmax(tf.matmul(rid_month_key, rid_month_query))
    rid_month_rep = tf.reduce_sum(tf.multiply(rid_month_seq, rid_month_att), axis=1)

    rid_prm_rep = tf.concat([rid_cnn_output, rid_day_rep, rid_week_rep, rid_month_rep], axis=1)

    fz_med_compt_price_seq = tf.reduce_sum(tf.multiply(fz_med_compt_price_seq, day_att_weight), axis=1)  # batch*8

    group_rid_cnn_output = tf.keras.layers.Conv1D(64, 3, strides=1, activation="relu", use_bias=True, padding='same',
                                                  input_shape=(30, 6))(group_rid_series)
    group_rid_cnn_output = tf.keras.layers.Conv1D(128, 3, strides=1, activation="relu", use_bias=True, padding='same')(
        group_rid_cnn_output)
    group_rid_cnn_output = tf.keras.layers.Conv1D(64, 3, strides=1, activation="relu", use_bias=True, padding='same')(
        group_rid_cnn_output)
    group_rid_cnn_output = tf.keras.layers.Flatten()(group_rid_cnn_output)

    group_day_query = tf.expand_dims(group_day_query, axis=2)
    group_day_att = tf.nn.softmax(tf.matmul(group_day_key, group_day_query))
    group_day_rep = tf.reduce_sum(tf.multiply(group_day_seq, group_day_att), axis=1)
    group_week_query = tf.expand_dims(group_week_query, axis=2)
    group_week_att = tf.nn.softmax(tf.matmul(group_week_key, group_week_query))
    group_week_rep = tf.reduce_sum(tf.multiply(group_week_seq, group_week_att), axis=1)
    group_month_query = tf.expand_dims(group_month_query, axis=2)
    group_month_att = tf.nn.softmax(tf.matmul(group_month_key, group_month_query))
    group_month_rep = tf.reduce_sum(tf.multiply(group_month_seq, group_month_att), axis=1)

    group_prm_rep = tf.concat([group_rid_cnn_output, group_day_rep, group_week_rep, group_month_rep], axis=1)

    ### group-level occupancy
    group_compete_input = tf.concat([group_static_fea, group_ext_feats], axis=1)
    group_demand_input = tf.concat([group_static_fea, group_prm_rep], axis=1)
    group_compete_outputs = tf.keras.layers.Dropout(0.2)(group_compete_input)
    group_compete_outputs = layers.batch_norm(group_compete_outputs, is_training=is_training, activation_fn=None,
                                              variables_collections=[dnn_parent_scope])
    group_compete_outputs = tf.keras.layers.Dense(512, name='group_compete_Dense1', use_bias=False)(
        group_compete_outputs)

    group_compete_outputs = layers.batch_norm(group_compete_outputs, is_training=is_training,
                                              activation_fn=activation_fn,
                                              variables_collections=[dnn_parent_scope])
    group_compete_outputs = tf.keras.layers.Dense(256, name='group_compete_Dense2', use_bias=False)(
        group_compete_outputs)
    group_compete_outputs = layers.batch_norm(group_compete_outputs, is_training=is_training,
                                              activation_fn=activation_fn,
                                              variables_collections=[dnn_parent_scope])
    group_demand_outputs = tf.keras.layers.Dropout(0.2)(group_demand_input)
    group_demand_outputs = layers.batch_norm(group_demand_outputs, is_training=is_training, activation_fn=None,
                                             variables_collections=[dnn_parent_scope])
    group_demand_outputs = tf.keras.layers.Dense(512, name='group_demand_Dense1', use_bias=False)(group_demand_outputs)

    group_demand_outputs = layers.batch_norm(group_demand_outputs, is_training=is_training, activation_fn=activation_fn,
                                             variables_collections=[dnn_parent_scope])
    group_demand_outputs = tf.keras.layers.Dense(256, name='group_demand_Dense2', use_bias=False)(group_demand_outputs)
    group_demand_outputs = layers.batch_norm(group_demand_outputs, is_training=is_training, activation_fn=activation_fn,
                                             variables_collections=[dnn_parent_scope])
    group_demand_vec = group_demand_outputs
    group_demand_rep = select_block(group_demand_outputs, 'group_demand_sb')
    group_compete_input_new = tf.concat([group_compete_outputs, group_demand_rep], axis=1)
    group_compete_outputs = tf.keras.layers.Dense(128, name='group_compete_Dense3', use_bias=False)(
        group_compete_input_new)
    group_compete_outputs = layers.batch_norm(group_compete_outputs, is_training=is_training,
                                              activation_fn=activation_fn,
                                              variables_collections=[dnn_parent_scope])
    group_logit = tf.keras.layers.Dense(1, name='group_compete_Dense4', use_bias=False)(group_compete_outputs)
    group_compete_rep = select_block(group_compete_outputs, 'group_compete_sb')
    group_demand_input_new = tf.concat([group_demand_outputs, group_compete_rep], axis=1)
    group_demand_outputs = tf.keras.layers.Dense(128, name='group_demand_Dense3', use_bias=False)(
        group_compete_input_new)
    group_demand_outputs = layers.batch_norm(group_demand_outputs, is_training=is_training,
                                             activation_fn=activation_fn,
                                             variables_collections=[dnn_parent_scope])
    group_bias_logit = tf.keras.layers.Dense(1, name='group_demand_Dense4', use_bias=False)(group_demand_outputs)


    ### room-level occupancy
    rid_compete_input = tf.concat([rid_static_fea, rid_internal_pre, rid_price_internal_pre, rid_ext_repre], axis=1)
    rid_demand_input = tf.concat([rid_static_fea, rid_prm_rep], axis=1)
    rid_compete_outputs = tf.keras.layers.Dropout(0.2)(rid_compete_input)
    rid_compete_outputs = layers.batch_norm(rid_compete_outputs, is_training=is_training, activation_fn=None,
                                            variables_collections=[dnn_parent_scope])
    rid_compete_outputs = tf.keras.layers.Dense(512, name='rid_compete_Dense1', use_bias=False)(rid_compete_outputs)

    rid_compete_outputs = layers.batch_norm(rid_compete_outputs, is_training=is_training, activation_fn=activation_fn,
                                            variables_collections=[dnn_parent_scope])
    rid_compete_outputs = tf.keras.layers.Dense(256, name='rid_compete_Dense2', use_bias=False)(rid_compete_outputs)
    rid_compete_outputs = layers.batch_norm(rid_compete_outputs, is_training=is_training, activation_fn=activation_fn,
                                            variables_collections=[dnn_parent_scope])
    rid_demand_outputs = tf.keras.layers.Dropout(0.2)(rid_demand_input)
    rid_demand_outputs = layers.batch_norm(rid_demand_outputs, is_training=is_training, activation_fn=None,
                                           variables_collections=[dnn_parent_scope])
    rid_demand_outputs = tf.keras.layers.Dense(512, name='rid_demand_Dense1', use_bias=False)(rid_demand_outputs)

    rid_demand_outputs = layers.batch_norm(rid_demand_outputs, is_training=is_training, activation_fn=activation_fn,
                                           variables_collections=[dnn_parent_scope])
    rid_demand_outputs = tf.keras.layers.Dense(256, name='rid_demand_Dense2', use_bias=False)(rid_demand_outputs)
    rid_demand_outputs = layers.batch_norm(rid_demand_outputs, is_training=is_training, activation_fn=activation_fn,
                                           variables_collections=[dnn_parent_scope])
    rid_demand_rep = select_block(rid_demand_outputs, 'rid_demand_sb')
    rid_compete_input_new = tf.concat([rid_compete_outputs, rid_demand_rep], axis=1)
    rid_compete_outputs = tf.keras.layers.Dense(128, name='rid_compete_Dense3', use_bias=False)(rid_compete_input_new)
    rid_compete_outputs = layers.batch_norm(rid_compete_outputs, is_training=is_training, activation_fn=activation_fn,
                                    variables_collections=[dnn_parent_scope])
    rid_logit = tf.keras.layers.Dense(1, name='rid_compete_Dense4', use_bias=False)(rid_compete_outputs)
    rid_compete_rep = select_block(rid_compete_outputs, 'rid_compete_sb')
    prm_simi_weight = select_block2(tf.concat([rid_prm_rep, group_prm_rep], axis=1), 'prm_simi_sb', 256)
    group_demand_vec = prm_simi_weight*group_demand_vec
    rid_demand_input_new = tf.concat([rid_demand_outputs, rid_compete_rep, group_demand_vec], axis=1)
    rid_demand_outputs = tf.keras.layers.Dense(128, name='rid_demand_Dense3', use_bias=False)(rid_demand_input_new)
    rid_demand_outputs = layers.batch_norm(rid_demand_outputs, is_training=is_training, activation_fn=activation_fn,
                                            variables_collections=[dnn_parent_scope])
    bias_logit = tf.keras.layers.Dense(1, name='rid_demand_Dense4', use_bias=False)(rid_demand_outputs)

    add_variables_from_scope('Dense', [ops.GraphKeys.GLOBAL_VARIABLES, ops.GraphKeys.MODEL_VARIABLES])

    return rid_s, rid_logit, group_logit, group_ela, jy_label, rid_ela, p_0, q_0, sale_p, group_p, group_q, group_avg_price, group_avg_jy, group_price, bias_logit, group_bias_logit