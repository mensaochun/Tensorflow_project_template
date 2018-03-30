# VGG16 implemented by tensorflow

This is a Tensorflow implemention of VGG 16.

## Data

As example, here we use the Kaggle data: cats vs. dogs,  you can download from: [cats vs. dogs](https://www.kaggle.com/c/dogs-vs-cats)

After downloading, put the `train.zip` into dir `data`, and extract it.

python files in `data_loader` will resize the images, compute the images mean, and generate mini-batch data, for more details, please go to have a look.

## Model

VGG16 model is defined in dir `models`

## Train

In dir `trainers`, the solver is defined to train the model.

If you want to train your data, please go to dir `mains`, the command in terminal

~~~shell
python train.py --pretrain_model="path/to/vgg_pretrained_model.npy"  --config="path/to/config.yaml" --is_restore=False --exp_name="your_exp_name" --is_use_gpu=True
~~~

you can download vgg16 pretrained model here:[VGG16 trained on ImageNet](https://drive.google.com/open?id=0ByuDEGFYmWsbNVF5eExySUtMZmM)

Enjoy your training!



## TODO list

- [x] Data process 
- [x] Data loader
- [x] VGG16 model
- [x] Train
- [ ] Test
- [ ] Deploy



