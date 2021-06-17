import tensorflow as tf


class LoaderParameters:
    def __init__(self, batch_size: int, img_height: int, img_width: int):
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width


def load_data(train_data_directory, validation_data_directory, parameters: LoaderParameters):
    train_data = tf.keras.preprocessing.image_dataset_from_directory(
        train_data_directory,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(parameters.img_height, parameters.img_width),
        batch_size=parameters.batch_size)

    validation_data = tf.keras.preprocessing.image_dataset_from_directory(
        validation_data_directory,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(parameters.img_height, parameters.img_width),
        batch_size=parameters.batch_size)

    return train_data, validation_data
