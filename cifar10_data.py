import os
import sys
import tarfile
from six.moves import urllib
import pickle
import tensorflow as tf
import numpy as np

DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 24

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

def maybe_download_and_extract(destination):
    """Download and extract the tarball from Alex's website."""
    dest_directory = destination
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
            float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
    
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-py')
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def load_batch(batch):
    with open(batch, "rb") as fo:
        datadict = pickle.load(fo,encoding="latin1")
        data = datadict["data"]
        labels = datadict["labels"]
        data = np.array(data, dtype=np.float32)
        labels = np.array(labels)
        return data, labels

def load_dataset(folder):
    datasets = {"images_train":[], 
    "labels_train":[],
    "images_test":[], 
    "labels_test":[]}
    for b in range(1, 6):
        data, labels = load_batch(os.path.join(folder,"data_batch_{0}".format(b)))
        datasets["images_train"].append(data)
        datasets["labels_train"].append(labels)
    datasets["images_train"] = np.concatenate(datasets["images_train"])
    datasets["labels_train"] = np.concatenate(datasets["labels_train"])
    datasets["images_test"], datasets["labels_test"]  = load_batch(os.path.join(folder,"test_batch"))
    return datasets