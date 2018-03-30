import argparse
import os

import yaml

from data_loader.data_generator import MemoryDataGenerator
from models.VGG_model import VGGModel
from trainers.VGG_train import VGGTrainer


def set_environ(is_use_gpu):
    if is_use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


def make_exp_dir(exp_name):
    exp_dir = os.path.join(os.path.dirname(os.getcwd()), "experiments", exp_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        chkpts_dir = os.path.join(exp_dir, "checkpoint")
        summary_dir = os.path.join(exp_dir, "summary")
        os.makedirs(chkpts_dir)
        os.makedirs(summary_dir)

def get_args():
    argparser = argparse.ArgumentParser(description="Train a VGG net")
    argparser.add_argument("--pretrain_model", dest="pretrain_model",
                           default="/home/pi/PycharmProjects/TFFRCNN/data/pretrain_model/VGG_imagenet.npy", type=str)
    argparser.add_argument("--config", dest="config",
                           default="/home/pi/PycharmProjects/VGG16_tensorflow/configs/config.yaml", type=str)
    argparser.add_argument("--is_restore", dest="is_restore", default=False, type=bool)
    argparser.add_argument("--exp_name", dest="exp_name", default="exp1", type=str)
    argparser.add_argument("--is_use_gpu", dest="is_use_gpu", default=False, type=bool)
    args = argparser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    set_environ(args.is_use_gpu)
    make_exp_dir(args.exp_name)
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    vgg_model = VGGModel(config)
    dog_cat_data = MemoryDataGenerator(config)
    vgg_trainer = VGGTrainer(vgg_model, dog_cat_data, config)
    vgg_trainer.train(args.pretrain_model, args.is_restore)
