import argparse
import os

import cv2
import tensorflow as tf
import yaml

from get_model import model

os.environ["CUDA_VISIBLE_DEVICES"] = ""

def restore_graph(ckpt_dir):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    restorer = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    print 'Restoring from %s' % ckpt_dir
    restorer.restore(sess, ckpt.model_checkpoint_path)
    return sess

# def get_args():
#     argparser = argparse.ArgumentParser(description="Test a VGG net")
#     argparser.add_argument("--ckpt", dest="ckpt",
#                            default=None,
#                            type=str)
#     argparser.add_argument("--config", dest="config",
#                            default=None, type=str)
#     args = argparser.parse_args()
#     return args


def get_args():
    argparser = argparse.ArgumentParser(description="Test")
    argparser.add_argument("--model", dest="model", default="VGG", type=str)
    argparser.add_argument("--ckpt", dest="ckpt",
                           default="/home/pi/PycharmProjects/tensorflow_template/experiments/exp1/checkpoint",
                           type=str)
    argparser.add_argument("--config", dest="config",
                           default="/home/pi/PycharmProjects/tensorflow_template/configs/config.yaml", type=str)
    args = argparser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    vgg_model = model(args.model,config)
    sess = restore_graph(args.ckpt)
    img = cv2.imread("/home/pi/PycharmProjects/tensorflow_template/data/cat_dog_resize/cat.1.jpg")
    img_ = img.reshape(1, 224, 224, 3)
    result = sess.run(vgg_model.reference, feed_dict={vgg_model.X: img_})
    # cat:0, dog:1
    print result
