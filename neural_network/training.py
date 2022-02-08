import tensorflow as tf

from neural_network.utils.load_dataset import load as load_dataset, IMAGE_SIZE
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocessing
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy, Precision, Recall, AUC
from tensorflow.keras.optimizers import Adam


def build_model(train, test):
    tf.keras.backend.clear_session()

    base_network = MobileNetV2(
        include_top=False,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        weights='imagenet',
    )
    base_network.summary()
    base_network.trainable = False

    inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = mobilenet_v2_preprocessing(inputs)
    x = base_network(x, training=False)
    x = Flatten(name='flatten')(x)
    outputs = Dense(7, activation='softmax', name='custom_fc')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        loss=CategoricalCrossentropy(),
        optimizer=Adam(learning_rate=1),
        metrics=[
            Accuracy(),
            Precision(),
            Recall(),
            AUC()
        ],
    )

    model.summary()

    model.fit(
        train,
        validation_data=test,
        batch_size=64,
        epochs=3,
    )

    model.save('model.h5')


def train_network():
    train_set, test_set = load_dataset()
    build_model(train_set, test_set)
