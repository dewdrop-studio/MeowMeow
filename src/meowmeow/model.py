from .dataset import load_dataset


import tensorflow as tf
import keras
import os
from sklearn.model_selection import train_test_split

MODEL_SAVE_PATH = os.path.join(
    "/".join(os.path.abspath(__file__).split("/")[:-3]), "models"
)
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_PATH, "meowmeow.keras")


class Model:
    def __init__(self, **kwargs):

        self.__model = keras.Sequential()
        self.__dataset = load_dataset()
        self.__input_shape = self.__image_shape()

        if "load" in kwargs and kwargs["load"] == True:
            self.__load_pretrained()
        else:
            self.__build_model()

        if "batch_size" in kwargs:
            self.__batch_size = kwargs["batch_size"]
        else:
            self.__batch_size = 32

        x_train, x_test, y_train, y_test = train_test_split(
            self.__dataset["image"].values,
            self.__dataset["isSilly"],
            test_size=0.2,
            random_state=79420,
        )

        self.__train_dataset = self.__create_dataset(x_train, y_train, is_training=True)
        self.__test_dataset = self.__create_dataset(x_test, y_test, is_training=False)

    def __build_model(self):
        self.__model.add(keras.layers.Input(shape=self.__input_shape))

        ## --- Augmentation layers --- ##
        self.__model.add(keras.layers.RandomFlip("horizontal"))
        self.__model.add(keras.layers.RandomFlip("vertical"))
        self.__model.add(keras.layers.RandomRotation(0.1))

        ## --- Convolutional layers --- ##
        self.__model.add(keras.layers.Conv2D(32, (3, 3), activation="relu"))
        self.__model.add(keras.layers.MaxPooling2D((2, 2)))

        self.__model.add(keras.layers.Conv2D(32, (3, 3), activation="relu"))
        self.__model.add(keras.layers.MaxPooling2D((2, 2)))

        ## --- Fully connected layers --- ##
        self.__model.add(keras.layers.Flatten())
        self.__model.add(keras.layers.Dense(64, activation="relu"))
        self.__model.add(keras.layers.Dense(64, activation="relu"))
        self.__model.add(keras.layers.Dense(1, activation="sigmoid"))

        self.__model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

    def __load_image(self, image_path, label=None):
        if label is None:
            label = 0

        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [self.__input_shape[0], self.__input_shape[1]])
        image = tf.cast(image, tf.float32) / 255.0  # normalize between 0 and 1
        return image, label

    def __image_shape(self):
        random_image = self.__dataset["image"].iloc[0]
        image = tf.io.read_file(random_image)
        image = tf.image.decode_jpeg(image, channels=3)
        return image.shape

    def __load_pretrained(self):
        self.__model = keras.models.load_model(MODEL_SAVE_PATH)
        self.__input_shape = self.__model.input_shape[1:]

    def __create_dataset(self, x, y, is_training=True):
        """Creates a tf.data.Dataset pipeline from filepaths and labels."""
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        if is_training:
            dataset = dataset.shuffle(buffer_size=len(x))

        dataset = dataset.map(self.__load_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.__batch_size)

        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    def save(self):
        self.__model.save(MODEL_SAVE_PATH)

    def train(self, epochs=10):
        if self.__train_dataset is None:
            raise ValueError("Training dataset not available. Load a pretrained model.")

        self.__model.fit(
            self.__train_dataset,
            validation_data=self.__test_dataset,
            epochs=epochs,
            validation_split=0.2,
        )

    def evaluate(self):
        if self.__x_test is None or self.__y_test is None:
            raise ValueError("Test data not available. Load a pretrained model.")
        return self.__model.evaluate(self.__x_test, self.__y_test)

    def predict(self, images):
        if not isinstance(images, list):
            images = [images]

        images_tensor = tf.stack([self.__load_image(img) for img in images])
        predictions = self.__model.predict(images_tensor)
        return predictions

    def summary(self):
        return self.__model.summary()

    def inner_model(self):
        return self.__model
