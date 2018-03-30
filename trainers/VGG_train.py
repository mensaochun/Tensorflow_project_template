import os

import numpy as np
import tensorflow as tf

from utils.logger import Summary


class VGGTrainer(object):
    def __init__(self, model, data, config):
        self.model = model
        self.data = data
        # read params from configs
        self.config = config
        self.logdir = self.config["logdir"]
        self.max_iter = self.config["max_iter"]
        self.summary_iter = self.config["summary_iter"]
        self.checkpoint_dir = self.config["checkpoint_dir"]
        self.max_to_keep = self.config["max_to_keep"]
        self.save_iter = self.config["save_iter"]
        self.snapshot_prefix = self.config["snapshot_prefix"]
        self.is_learning_rate_decay = self.config["learning_rate_decay"]["is_learning_rate_decay"]
        # set learning rate
        if (self.is_learning_rate_decay):
            self.global_step, self.learning_rate = self._init_learning_rate()
        else:
            self.global_step = None
            self.learning_rate = self.config["learning_rate"]

        # to choose optimizer.
        self.optimizer = self._set_optimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.model.loss, global_step=self.global_step)  # TODO

        # init saver
        self.saver = tf.train.Saver(max_to_keep=5)

        # set up session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _init_learning_rate(self):
        decay_steps = self.config["learning_rate_decay"]["decay_steps"]
        decay_rate = self.config["learning_rate_decay"]["decay_rate"]
        staircase = self.config["learning_rate_decay"]["staircase"]
        initial_learning_rate = self.config["learning_rate"]
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps, decay_rate,
                                                   staircase, name='learning_rate')
        return global_step, learning_rate

    def _load_pretrain_model(self, pretrain_model):
        data_dict = np.load(pretrain_model).item()
        for key in data_dict:
            with tf.variable_scope(key, reuse=True):
                for subkey in data_dict[key]:
                    try:
                        var = tf.get_variable(subkey)
                        self.sess.run(var.assign(data_dict[key][subkey]))
                        print "Assign pretrain model " + subkey + " to " + key
                    except ValueError:
                        print "Note: ignore " + key

    def _set_optimizer(self, learning_rate):
        if self.config["optimizer"] == "Adagrad":
            return tf.train.AdagradOptimizer(learning_rate)
        elif self.config["optimizer"] == "GradientDescent":
            return tf.train.GradientDescentOptimizer(learning_rate)
        elif self.config["optimizer"] == "Adam":
            return tf.train.AdamOptimizer(learning_rate)
        elif self.config["optimizer"] == "RMSProp":
            return tf.train.RMSPropOptimizer(learning_rate)
        else:
            raise Exception("Invalid optimizer!")

    def train(self, pretrain_model=None, is_restore=False):

        # summary
        tf.summary.scalar("Loss", self.model.loss)
        tf.summary.scalar("Accuracy", self.model.accuracy)
        Summary.add_all_vars()
        Summary.add_all_grads(self.optimizer, self.model.loss)
        Summary.add_image_summary(self.model.X[0:10, :, :, :])
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(logdir=self.logdir, graph=self.sess.graph)

        # load pretrained-model or restore training
        if pretrain_model is not None and not is_restore:
            print "Loading pretrained model weights from ", pretrain_model
            self._load_pretrain_model(pretrain_model)

        if is_restore:
            try:
                ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
                print 'Restoring from %s' % self.checkpoint_dir
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                stem = os.path.splitext(os.path.basename(ckpt.model_checkpoint_path))[0]
                restore_iter = int(stem.split('_')[-1])
                if self.is_learning_rate_decay:
                    self.sess.run(self.global_step.assign(restore_iter))
            except:
                raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)

        print "Start to train..."
        for step in range(self.max_iter):
            # get batch training data
            self.batch_X, self.batch_Y = self.data.next_batch()
            feed_dict = {self.model.X: self.batch_X, self.model.Y: self.batch_Y}

            # summary information
            if step % self.summary_iter == 0:
                _, summary, loss, acc, p = self.sess.run(
                    [self.train_op, merged, self.model.loss, self.model.accuracy, self.model.probability],
                    feed_dict=feed_dict)
                writer.add_summary(summary, step)
                if self.is_learning_rate_decay:
                    learning_rate = self.sess.run(self.learning_rate)
                else:
                    learning_rate=self.learning_rate
                print "Epoch:%d, Step:%d/%d, Learning rate:%f, Loss:%.4f, Accuracy:%.4f" % (self.data.epochs_completed,
                                                                                            step,
                                                                                            self.max_iter,
                                                                                            learning_rate,
                                                                                            loss,
                                                                                            acc)
            else:
                self.sess.run(self.optimizer, feed_dict=feed_dict)

            # save checkpoint
            if (step + 1) % self.save_iter == 0:
                filename = self.checkpoint_dir + self.snapshot_prefix + "_iter_" + str(step) + ".ckpt"
                self.saver.save(self.sess, filename, global_step=self.global_step, )
                print "Save checkpoint to ", self.checkpoint_dir
