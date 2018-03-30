import numpy as np

from data_reader import read_data_and_sub_mean


class DataGenerator(object):
    def __init__(self, config=None):
        self.config = config

    def next_batch(self):
        raise NotImplementedError


# store data in cpu memory.
class MemoryDataGenerator(DataGenerator):
    def __init__(self, config):
        super(MemoryDataGenerator, self).__init__(config)
        # read params form config
        self.data_dir = self.config["data_dir"]
        self.images_mean = self.config["images_mean"]
        self.num_per_calss = self.config["num_per_class"]
        self.batch_size = self.config["batch_size"]
        self.is_shuffle = self.config["is_shuffle"]

        # read resized images and labels, and the resized images substract mean
        data_dict = read_data_and_sub_mean(self.data_dir, self.images_mean, self.num_per_calss)
        self._images = data_dict["images"]
        self._labels = data_dict["labels"]
        self._num_samples = self._images.shape[0]
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def next_batch(self):

        """
        Return the next `batch_size` examples from this data set.

        """
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and self.is_shuffle:
            perm0 = np.arange(self._num_samples)
            np.random.shuffle(perm0)
            self._images = self._images[perm0]
            self._labels = self._labels[perm0]
        # Go to the next epoch
        if start + self.batch_size > self._num_samples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_samples = self._num_samples - start
            images_rest_part = self._images[start:self._num_samples]
            labels_rest_part = self._labels[start:self._num_samples]
            # Shuffle the data
            if self.is_shuffle:
                perm = np.arange(self._num_samples)
                np.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = self.batch_size - rest_num_samples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += self.batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_samples

    @property
    def epochs_completed(self):
        return self._epochs_completed


# store all data in disk, and read batch data from disk to cpu memory.
class DiskDataGenerator(DataGenerator):
    pass


# store data in TFrecord file, and read batch data from TFRecord file.
class TFRecordDataGenerator(DataGenerator):
    pass
