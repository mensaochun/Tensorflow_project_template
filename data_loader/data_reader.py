import cPickle
import os
import random

import cv2
import numpy as np


def read_cifar10_data(file):
    with open(file, 'rb') as fo:
        data_dict = cPickle.load(fo)
    return data_dict


def read_data_and_sub_mean(data_dir, img_mean, num_per_class=1000, image_size=[224, 224, 3]):
    print "Begin to read dog and cat data..."

    file_names = os.listdir(data_dir)
    random.shuffle(file_names)
    total_num = len(file_names)
    if (2 * num_per_class > total_num):
        raise Exception("The number of images to use is more than total num.")
    images = np.zeros([2 * num_per_class, image_size[0], image_size[1], 3], dtype=np.float32)
    labels = np.zeros([2 * num_per_class, 2])
    total_count, cat_count, dog_count = 0, 0, 0
    for name in file_names:
        if total_count % 100 == 0:
            print "Read image", total_count
        if name[:3] == "cat":
            if cat_count < num_per_class:
                file_name = os.path.join(data_dir, name)
                img = cv2.imread(file_name)
                img = img - np.array(img_mean).reshape(1, 1, 3)
                labels[total_count, 0] = 1
                images[total_count, :, :, :] = img
                cat_count += 1
                total_count += 1
            else:
                continue
        elif name[:3] == "dog":
            if dog_count < num_per_class:
                file_name = os.path.join(data_dir, name)
                img = cv2.imread(file_name)
                img = img - np.array(img_mean)
                labels[total_count, 1] = 1
                images[total_count, :, :, :] = img
                dog_count += 1
                total_count += 1
            else:
                continue

        if cat_count >= num_per_class and dog_count >= num_per_class:
            break

    if total_count != 2 * num_per_class:
        raise Exception("The total number of cat or dog is less than the number per-class to use.")
    data_dict = {}
    data_dict["images"] = images
    data_dict["labels"] = labels

    print "Total number of the samples is", total_count
    print "Dog:", dog_count, " Cat:", cat_count

    return data_dict


def resize_image(data_dir, save_dir, resize_shape=(224, 224)):
    print "Begin to resize dog and cat data..."
    file_names = os.listdir(data_dir)
    total_count, cat_count, dog_count = 0, 0, 0
    for name in file_names:
        if name[:3] == "cat":
            cat_count += 1
        else:
            dog_count += 1
        file_name = os.path.join(data_dir, name)
        img = cv2.imread(file_name)
        img_resize = cv2.resize(img, resize_shape)
        cv2.imwrite(os.path.join(save_dir, name), img_resize)
        total_count += 1
        if (total_count % 100 == 0):
            print "Resize image", total_count

    print "Total number of the samples is", total_count
    print "Dog:", dog_count, " Cat:", cat_count


def compute_mean(data_dir, image_size=(224, 224, 3)):
    print "Begin to compute images mean..."
    file_names = os.listdir(data_dir)
    image_sum = np.zeros([224, 224, 3], dtype=np.float32)
    total_count = 0
    for name in file_names:
        file_name = os.path.join(data_dir, name)
        image = cv2.imread(file_name)
        if image.shape != image_size:
            raise Exception("Dimension not match!")
        image_sum += image
        total_count += 1
    image_mean = image_sum / total_count
    mean = np.mean(image_mean, axis=(0, 1))  # BGR
    print mean


if __name__ == '__main__':
    data_dir = "/home/pi/PycharmProjects/VGG16_tensorflow/data/dog_cat/train"
    save_dir = "/home/pi/PycharmProjects/VGG16_tensorflow/data/dog_cat/train_resize"
    # resize_image(data_dir, save_dir)
    compute_mean(save_dir)
