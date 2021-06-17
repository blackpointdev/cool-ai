from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from data import LoaderParameters
import tensorflow as tf


def create_model(number_of_classes, train_data, validation_data, parameters: LoaderParameters):
    model = Sequential([
        layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(parameters.img_height,
                                                                           parameters.img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(number_of_classes)
    ])

    compile_model(model)

    model.summary()

    number_of_epochs = 10
    history = train_model(model, train_data, validation_data, number_of_epochs)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(number_of_epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    print("Saving model to file...")
    model.save_weights('model_backup/model_weights')

    return model


def load_model(number_of_classes, parameters: LoaderParameters):
    model = Sequential([
        layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(parameters.img_height,
                                                                           parameters.img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(number_of_classes)
    ])
    model.load_weights('model_backup/model_weights')

    return model


def compile_model(model):
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])


def train_model(model, train_data, validation_data, epochs: int):
    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=epochs
    )
    return history
