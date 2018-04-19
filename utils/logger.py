
import os
import tensorflow as tf
class Logger:
    def __init__(self, sess, config):
        self.sess = sess
        self.config = config
        self.summary_placeholders = {}
        self.summary_ops = {}
        self.train_summary_writer = tf.summary.FileWriter(os.path.join(self.config.summary_dir, "train"),
                                                          self.sess.graph)
        self.test_summary_writer = tf.summary.FileWriter(os.path.join(self.config.summary_dir, "exp1"))

    # it can summarize scalars and images.
    def summarize(self, step, summarizer="train", scope="", summaries_dict=None):
        """
        :param step: the step of the summary
        :param summarizer: use the train summary writer or the exp1 one
        :param scope: variable scope
        :param summaries_dict: the dict of the summaries values (tag,value)
        :return:
        """
        summary_writer = self.train_summary_writer if summarizer == "train" else self.test_summary_writer
        with tf.variable_scope(scope):

            if summaries_dict is not None:
                summary_list = []
                for tag, value in summaries_dict.items():
                    if tag not in self.summary_ops:
                        if len(value.shape) <= 1:
                            self.summary_placeholders[tag] = tf.placeholder('float32', value.shape, name=tag)
                        else:
                            self.summary_placeholders[tag] = tf.placeholder('float32', [None] + list(value.shape[1:]),
                                                                            name=tag)
                        if len(value.shape) <= 1:
                            self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])
                        else:
                            self.summary_ops[tag] = tf.summary.image(tag, self.summary_placeholders[tag])

                    summary_list.append(self.sess.run(self.summary_ops[tag], {self.summary_placeholders[tag]: value}))

                for summary in summary_list:
                    summary_writer.add_summary(summary, step)
                summary_writer.flush()


class Summary(object):
    def __init__(self):
        pass

    @staticmethod
    def add_all_vars():
        trainable_var = tf.trainable_variables()
        for var in trainable_var:
            Summary.add_weights_or_biases_summary(var)

    @staticmethod
    def add_all_grads(optimizer, loss):
        trainable_vars = tf.trainable_variables()
        grads = optimizer.compute_gradients(loss, var_list=trainable_vars)
        # print(len(var_list))
        for grad, var in grads:
            Summary.add_gradient_summary(grad, var)
        optimizer.apply_gradients(grads)

    @staticmethod
    def add_weights_or_biases_summary(var):
        if var is not None:
            tf.summary.histogram(var.op.name, var)

    @staticmethod
    def add_activation_summary(var):
        if var is not None:
            tf.summary.histogram(var.op.name + "/Activation", var)
            tf.summary.scalar(var.op.name + "/Sparsity", tf.nn.zero_fraction(var))

    @staticmethod
    def add_gradient_summary(grad, var):
        if grad is not None:
            tf.summary.histogram(var.op.name + "/Gradient", grad)

    @staticmethod
    def add_image_summary(X):
        if X is not None:
            tf.summary.image("Input_image",X, 10)
