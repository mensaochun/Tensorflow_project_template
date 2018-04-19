# A tensorflow project template 

This is a Tensorflow project template based on VGG16 model.
## Config

All parameters are in config file, please see `project_root/configs/config.yaml`.

## Data 

### 1. Download data

As example, here we use the Kaggle data: cats vs. dogs,  you can download data here: [cats vs. dogs](https://www.kaggle.com/c/dogs-vs-cats)

After downloading, put the `train.zip` into folder `data`, and extract it, rename it to `cat_dog`.


### 2. Data preprocessing

- Resize image

Note images in cat_dog have different sizes, so first we should resize the images. Save the resized images in `project_root/data/cat_dog_resize`. 

- Compute mean

Compute the images mean, and write it to config file.

To do above operations, please go to `project_root/data_loader/data_reader.py` .

### 3. Generate data batch

Generally, you can store data in three ways.

1. store data in cpu/gpu memory.
2. store data using tfrecord.
3. store data in disk.

The io speed is 1>2>3. But cpu/gpu memory size is often limited, in this situation, way 2 or 3 may be a better choice.

Here, for convenience, we only store data in way 1.

Note, when training, data batch is generated continuously. We implement this in `/data_loader/data_generator.py` 

We define the Parent class :

~~~python
class DataGenerator(object):
    def __init__(self, config=None):
        self.config = config

    def next_batch(self):
        raise NotImplementedError
~~~

you can implement anyone of the 3 child classes Inherited from this parent, depending on your choice.

~~~python
# store data in cpu memory.
class MemoryDataGenerator(DataGenerator):
    pass

# store all data in disk, and read batch data from disk to cpu memory.
class DiskDataGenerator(DataGenerator):
    pass

# store data in TFrecord file, and read batch data from TFRecord file.
class TFRecordDataGenerator(DataGenerator):
    pass
~~~

For more details, please go to have a look.


## Models

All models are defined in folder `project_root/models`.

**Note:** when you define your model, you must define the model `loss`, `accuray`, `reference`.

## Trainer

### 1. Download pretrained model.

you can download vgg16 pretrained model here:[VGG16 trained on ImageNet](https://drive.google.com/open?id=0ByuDEGFYmWsbNVF5eExySUtMZmM)。And put it in the `project_root/data/pretrained_model`

### 2. Trainer

The solver is defined to train the model in folder `project_root/trainers`

## Mains

Train and test are defined in `project_root/mains`.

For either training or testing, you should modify the config file and command-line arguments correspondingly.

Enjoy your training time!

## Tensorboard

Tensorboard is a powerful tool  for debugging tensorflow program, please don't ignore it.

class `Summary` defined in `project_root/utils/logger.py` is used to get all kinds of visualization.

For more details, please go to have a look.




## TODO list

- [x] Data process 

- [x] Data loader

- [x] VGG16 model

- [x] Train

- [x] Test

- [ ] validation

- [ ] Use factory pattern to call specified model.

      ​



