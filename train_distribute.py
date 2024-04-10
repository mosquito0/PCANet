# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import optimizers
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops
from utils import *
from model_fn import model_block
from fg_worker import FeatureColumnGenerator
import os
import time
import rtp_fg
import json
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)
tf.flags.DEFINE_string('tables', '', '')
tf.flags.DEFINE_string('train_tables', '', '')
tf.flags.DEFINE_string('dev_tables', '', '')

# fg
tf.flags.DEFINE_string("fg_config_java", "", "feature generator file for java")
tf.flags.DEFINE_string("fg_config_tf", "", "feature generator file for tf")

tf.flags.DEFINE_string('model_dir', 'zy_model/lr/', 'model dir')
tf.flags.DEFINE_string('model_dir_restore', 'zy_model/lr/', 'model dir')
tf.flags.DEFINE_bool('is_sequence_train', False, 'flag for train mode')

tf.flags.DEFINE_integer('batch_size', 512, 'batch size')
tf.flags.DEFINE_integer('max_train_step', 1000000, 'max_train_step')
tf.flags.DEFINE_integer('dev_total', 100000, 'dev_total')

tf.flags.DEFINE_bool('is_training', True, '')
tf.flags.DEFINE_float('keep_prob', 1.0, "")
tf.flags.DEFINE_string('optimizer', 'Adam', "")
tf.flags.DEFINE_string('pad_str', 'pad', "")

tf.flags.DEFINE_string('buckets', "", 'buckets')
tf.flags.DEFINE_string('checkpointDir', "", 'checkpointDir')

# distribute
tf.flags.DEFINE_integer("task_index", None, "Worker task index")
tf.flags.DEFINE_string("ps_hosts", "", "ps hosts")
tf.flags.DEFINE_string("worker_hosts", "", "worker hosts")
tf.flags.DEFINE_string("job_name", None, "job name: worker or ps")
tf.flags.DEFINE_integer('aggregate', 100, 'aggregate batch number')
tf.flags.DEFINE_integer("save_time", 600, 'train epoch')

FLAGS = tf.flags.FLAGS

table_col_num = 14
# -----------------------------------------------------------------------------------------------------------

linear_parent_scope = "linear"
dnn_parent_scope = "dnn"


def parser(batch, feature_configs):
    columns = batch.get_next()
    rid, shid, ela_label, rid_ela, jy_label, feature, avg_price, avg_jy, group_p, group_q, price, group_avg_price, group_avg_jy, group_price = columns
    # shape must be rank 1
    feature = tf.reshape(feature, [-1, 1])
    feature = tf.squeeze(feature, axis=1)
    features = rtp_fg.parse_genreated_fg(feature_configs, feature)
    return rid, features, ela_label, jy_label, rid_ela, avg_price, avg_jy, group_p, group_q, price, group_avg_price, group_avg_jy, group_price


def input_fn(files, batch_size, mode, slice_id, slice_count):
    tf.logging.info("slice_count:{}, slice_id:{}".format(slice_count, slice_id))
    if mode == 'train':
        if FLAGS.is_sequence_train:
            # sequence train
            dataset = tf.data.TableRecordDataset(files, [[' ']] * table_col_num, slice_id=slice_id,
                                                 slice_count=slice_count).batch(batch_size)
        else:
            # global train
            dataset = tf.data.TableRecordDataset(files, [[' ']] * table_col_num, slice_id=slice_id,
                                                 slice_count=slice_count).shuffle(
                buffer_size=200 * batch_size).repeat().batch(batch_size)
    elif mode == 'dev':
        dataset = tf.data.TableRecordDataset(files, [[' ']] * table_col_num, slice_id=slice_id,
                                             slice_count=slice_count).repeat().batch(batch_size)
    return dataset


# -----------------------------------------------------------------------------------------------------------
def elapse_time(start_time):
    return round((time.time() - start_time) / 60)


def get_remaining_time(delta, now_step, max_step):
    return int(delta * (max_step - now_step) / 60)


def contrastive_loss(y_true, y_pred):
    y_neg = tf.cast((y_true < 1), tf.float32)
    loss2 = y_true * tf.keras.backend.square(y_pred) + y_neg * tf.keras.backend.square(1 - y_pred)
    return loss2


def focal_loss(y_true, y_pred, alpha=0.25, gamma=2):
    ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred, name='sigmoid_cross_entropy')
    y_pred = tf.sigmoid(y_pred)
    p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
    if alpha:
        alpha = tf.convert_to_tensor(alpha, dtype=tf.keras.backend.floatx())
        alpha_factor = (y_true * alpha + (1 - y_true) * (1 - alpha))
    if gamma:
        gamma = tf.convert_to_tensor(gamma, dtype=tf.keras.backend.floatx())
        modulating_factor = tf.pow((1.0 - p_t), gamma)

    return (alpha_factor * modulating_factor * ce)


def auc_loss(_y_true, y_pred):
    y_pred = tf.sigmoid(y_pred)
    y_pred_clipped = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    y_true = (_y_true > 0.5)
    y_pos = tf.boolean_mask(y_pred_clipped, y_true)
    y_neg = tf.boolean_mask(y_pred_clipped, ~y_true)
    num_pos = tf.shape(y_pos)[0]
    num_neg = tf.shape(y_neg)[0]
    y_pos = tf.reshape(y_pos, [num_pos, 1])
    y_neg = tf.reshape(y_neg, [num_neg, 1])
    y_pos_matrix = tf.tile(y_pos, [1, num_neg])
    y_neg_matrix = tf.tile(y_neg, [1, num_pos])
    y_pos_matrix = tf.transpose(y_pos_matrix, [1, 0])
    auc_matrix = y_pos_matrix - y_neg_matrix
    auc_matrix = tf.reshape(auc_matrix, [num_pos * num_neg])
    auc_index = (auc_matrix < 0.05)
    auc_loss = tf.boolean_mask(auc_matrix, auc_index)
    auc_loss = tf.keras.backend.square(auc_loss)
    auc_loss = tf.reduce_sum(auc_loss) / ((tf.keras.backend.sum(
        tf.cast((_y_true < 1), tf.float32)) * tf.keras.backend.sum(tf.cast((_y_true > 0), tf.float32))) + 1)
    return auc_loss


def model_fn(data_batch, global_step, is_chief):
    # construct the model structure
    # feature preprocess
    tf.logging.info("loading json config...")

    with open('./xx_fliggy_pricing_dnn.json', 'r') as f:
        feature_configs_java = json.load(f)

    with open('./xx_fliggy_pricing_dnn_tf.json', 'r') as f:
        feature_configs_tf = json.load(f)

    tf.logging.info("java fg parsing...")
    rid, features, label, jy_label, rid_ela, avg_price, avg_jy, group_p, group_q, price, group_avg_price, group_avg_jy, group_price = parser(data_batch, feature_configs_java)
    tf.logging.info("finished java fg parsing...")

    tf.logging.info("tf fg...")
    fc_generator = FeatureColumnGenerator(feature_configs_tf)
    tf.logging.info("finished tf fg...")
    ########################################################
    params = {name: value for name, value in FLAGS.__flags.items()}
    print("print params...")
    for key in params:
        print("params {}: {}".format(key, params[key]))
    # model
    rid_s, rid_logit, group_logit, label, jy_label, rid_ela, P_0, Q_0, group_p, group_q, sale_P, group_p, group_q, group_avg_price, group_avg_jy, group_price, bias_logit, group_bias_logit = model_block(
        rid, features, label, jy_label, rid_ela, avg_price, avg_jy, group_p, group_q, price, group_avg_price, group_avg_jy, group_price, fc_generator, FLAGS.is_training,
        FLAGS.keep_prob, params)

    with tf.variable_scope('predict'):
        rid_logit = 5.0 * (-1.0 / 12.0 * tf.sigmoid(rid_logit) - 1.0 / 6.0)
        group_logit = 5.0 * (-1.0 / 12.0 * tf.sigmoid(group_logit) - 1.0 / 6.0)

        rid_p_logit = Q_0 * tf.exp(rid_logit) + bias_logit * P_0 / sale_P
        group_p_logit = group_q * tf.exp(group_logit) + group_bias_logit * group_p / group_price
        predict_score = tf.identity(rid_p_logit, name="rank_predict")

    with tf.name_scope("train_metrics"):
        rid_mae = tf.reduce_mean(tf.abs(rid_p_logit - rid_ela))
        rid_mse = tf.reduce_mean((rid_p_logit - rid_ela) * (rid_p_logit - rid_ela))
        group_mae = tf.reduce_mean(tf.abs(group_logit - label))
        group_mse = tf.reduce_mean((group_logit - label) * (group_logit - label))
        train_metrics = [rid_mae, rid_mse, group_mae, group_mse]

    with tf.name_scope('loss'):
        rid_ela_loss = tf.losses.huber_loss(rid_ela, rid_p_logit)
        group_loss = tf.losses.huber_loss(label, group_p_logit)
        loss = rid_ela_loss + 0.3 * group_loss

        loss = tf.reduce_sum(loss)
        tf.add_to_collection("losses", loss)
        losses = tf.get_collection('losses')
        tf.logging.info("train losses: {}".format(losses))
        loss_total = tf.add_n(losses)

    train_op = make_training_op(loss_total, global_step)
    return loss_total, train_op, train_metrics


def make_training_op(training_loss, global_step):
    _DNN_LEARNING_RATE = 0.001
    _LINEAR_LEARNING_RATE = 0.005
    _GRADIENT_CLIP_NORM = 100.0
    OPTIMIZER_CLS_NAMES = [
        "Adagrad",
        "Adam",
        "Ftrl",
        "Momentum",
        "RMSProp",
        "SGD"
    ]
    warm_up_learning_rate = 0.0001
    warm_up_step = 50000
    init_learning_rate = 0.001
    decay_steps = 10000
    decay_rate = 0.96
    learning_rate2 = tf.train.exponential_decay(_DNN_LEARNING_RATE,
                                                global_step,
                                                20000,
                                                0.9,
                                                staircase=True
                                                )
    learning_rate = tf.train.smooth_exponential_decay(warm_up_learning_rate,
                                                      warm_up_step,
                                                      init_learning_rate,
                                                      global_step,
                                                      decay_steps,
                                                      decay_rate)
    # bn
    with ops.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_ops = []

        if FLAGS.is_sequence_train:
            dnn_optimizer = tf.train.AdamAsyncOptimizer(learning_rate2)
        else:
            dnn_optimizer = tf.train.AdamOptimizer(learning_rate)
        train_ops.append(
            optimizers.optimize_loss(
                loss=training_loss,
                global_step=global_step,
                learning_rate=_DNN_LEARNING_RATE,
                optimizer=dnn_optimizer,
                variables=ops.get_collection("trainable_variables"),
                name=dnn_parent_scope,
                clip_gradients=None,
                increment_global_step=None))
        tf.logging.info(
            "optimizer scope {} variables: {}".format("trainable_variables", ops.get_collection("trainable_variables")))

        train_op = control_flow_ops.group(*train_ops)
        with ops.control_dependencies([train_op]):
            with ops.colocate_with(global_step):
                return state_ops.assign_add(global_step, 1).op


def train(worker_count, task_index, cluster, is_chief, target):
    worker_device = "/job:worker/task:%d/cpu:%d" % (task_index, 0)
    tf.logging.info("worker_deivce = %s" % worker_device)

    model_dir_restore = FLAGS.model_dir_restore
    model_dir = FLAGS.model_dir
    batch_size = FLAGS.batch_size
    checkpointDir = FLAGS.checkpointDir
    buckets = FLAGS.buckets
    model_dir = os.path.join(checkpointDir, model_dir)
    model_dir_restore = os.path.join(checkpointDir, model_dir_restore)
    tf.logging.info(
        "buckets:{} checkpointDir:{} checkpointDir_restore:{}".format(buckets, model_dir, model_dir_restore))
    # -----------------------------------------------------------------------------------------------
    tf.logging.info("loading input...")
    train_file = FLAGS.train_tables.split(',')
    dev_file = FLAGS.dev_tables.split(',')
    # assign io related variables and ops to local worker device
    with tf.device(worker_device):
        train_dataset = input_fn(train_file, batch_size, 'train', slice_count=worker_count, slice_id=task_index)
        train_iterator = train_dataset.make_one_shot_iterator()
    tf.logging.info("finished loading input...")
    # assign global variables to ps nodes
    available_worker_device = "/job:worker/task:%d" % (task_index)
    with tf.device(tf.train.replica_device_setter(worker_device=available_worker_device, cluster=cluster)):
        global_step = tf.Variable(0, name="global_step", trainable=False)
        # construct the model structure`
        loss, train_op, train_metrics = model_fn(train_iterator, global_step, is_chief)
    rid_mae, rid_mse, group_mae, group_mse = train_metrics
    tf.logging.info("start training")

    sess_config = tf.ConfigProto(allow_soft_placement=True,
                                 log_device_placement=False)

    sess_config.gpu_options.allow_growth = True
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]

    for var in bn_moving_vars:
        tf.add_to_collection(tf.GraphKeys.TRAINABLE_VARIABLES, var)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, var)

    hooks = [tf.train.StopAtStepHook(last_step=FLAGS.max_train_step)]
    saver = tf.train.Saver(name='restore_saver')
    step = 0

    with tf.train.MonitoredTrainingSession(checkpoint_dir=model_dir,
                                           master=target,
                                           is_chief=is_chief,
                                           config=sess_config,
                                           save_checkpoint_secs=FLAGS.save_time,
                                           hooks=hooks) as sess:
        ckpt_state = tf.train.get_checkpoint_state(model_dir_restore)
        if ckpt_state is not None:
            tf.logging.info('restore checkpoint: {}...'.format(ckpt_state.model_checkpoint_path))
            saver.restore(sess, ckpt_state.model_checkpoint_path)
        else:
            tf.logging.info('nothing to restore, train init...')

        chief_is_end = False
        sess_is_end = False
        while (not sess_is_end) and (not sess.should_stop()):
            if not chief_is_end:
                try:
                    step += 1
                    _, loss_val, global_step_val, rid_mae_, rid_mse_, group_mae_, group_mse_ = sess.run(
                        [train_op, loss, global_step, rid_mae, rid_mse, group_mae, group_mse])

                    rid_rmse = np.sqrt(rid_mse_)
                    group_rmse = np.sqrt(group_mse_)

                    print('losses:', loss_val)

                    tf.logging.info(
                        'rid mae: {} group mae: {} rid rmse: {} group rmse: {} loss at step {} global_step {} max_step {}: {}'.format(
                            rid_mae_, group_mae_, rid_rmse, group_rmse, step, global_step_val,
                            FLAGS.max_train_step,
                            loss_val))
                except tf.errors.OutOfRangeError as e:
                    # for sequence train
                    if is_chief:
                        tf.logging.info("chief node end...")
                        chief_is_end = True
                        tf.logging.info("waiting all worker nodes to be end")
                        last_step = global_step_val
                    else:
                        tf.logging.info("worker node end...")
                        break
            else:
                while 1:
                    # check all workers nodes
                    time.sleep(60)
                    global_step_val = sess.run(global_step)
                    if global_step_val > last_step:
                        last_step = global_step_val
                    else:
                        tf.logging.info("all worker nodes end. chief node is finished")
                        sess_is_end = True
                        break
    tf.logging.info("%d steps finished." % step)


def main(_):
    tf.logging.info("job name = %s" % FLAGS.job_name)
    tf.logging.info("task index = %d" % FLAGS.task_index)
    is_chief = FLAGS.task_index == 0
    # construct the servers
    ps_spec = FLAGS.ps_hosts.split(",")
    worker_spec = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"ps": ps_spec, "worker": worker_spec})
    worker_count = len(worker_spec)
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    # join the ps server
    if FLAGS.job_name == "ps":
        server.join()
    # start the training
    train(worker_count=worker_count, task_index=FLAGS.task_index, cluster=cluster, is_chief=is_chief,
          target=server.target)


if __name__ == '__main__':
    tf.app.run()
