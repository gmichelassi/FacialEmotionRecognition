from typing import Tuple

import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory

DATASET_PATH = './dataset/'
IMAGE_SIZE = 48


label_to_text = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}


def load() -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    train_set = image_dataset_from_directory(
        directory=DATASET_PATH + 'train',
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        labels='inferred',
        label_mode='categorical',
        batch_size=64,
        seed=9000
    )

    test_set = image_dataset_from_directory(
        directory=DATASET_PATH + 'test',
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        labels='inferred',
        label_mode='categorical',
        batch_size=64,
        seed=9000
    )

    return train_set, test_set
