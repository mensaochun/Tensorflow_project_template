import argparse
import os

import cv2
import numpy as np
import tensorflow as tf
import yaml

from data_loader.data_generator import MemoryDataGenerator
from models import VGGModel

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def load_weight(sess, weight_file):
    data_dict = np.load(weight_file).item()
    for key in data_dict:
        with tf.variable_scope(key, reuse=True):
            for subkey in data_dict[key]:
                try:
                    var = tf.get_variable(subkey)
                    sess.run(var.assign(data_dict[key][subkey]))
                    print "Assign pretrain model " + subkey + " to " + key
                except ValueError:
                    print "Note: ignore " + key


def predict(X, model):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    load_weight(sess, args.weights)
    result = sess.run(model.reference, feed_dict={model.X: X})
    return result


def plot_images(images):
    for i in range(images.shape[0]):
        img = images[i, :, :, :]
        # np.transpose(img,[2,1,0])
        img += np.array([103.939, 116.779, 123.68]).reshape(1, 1, 3)
        cv2.imshow("%s" % i, img.astype(np.uint8))
        cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_args():
    argparser = argparse.ArgumentParser(description="Test a VGG net")
    argparser.add_argument("--weights", dest="weights",
                           default="/home/pi/PycharmProjects/TFFRCNN/data/pretrain_model/VGG_imagenet.npy", type=str)
    argparser.add_argument("--config", dest="config",
                           default="/home/pi/PycharmProjects/VGG16_tensorflow/configs/config.yaml", type=str)
    args = argparser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    vgg_model = VGGModel(config)
    dog_cat_data = MemoryDataGenerator(config)
    images = dog_cat_data.images[0:50, :, :, :]
    # plot_images(images)
    result = predict(images, vgg_model)
    print result
