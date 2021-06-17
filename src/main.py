import pathlib

import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from model import create_model, load_model
from data import load_data, LoaderParameters

if __name__ == '__main__':
    # Prepare paths
    train_data_dir = pathlib.Path("fruits_dataset/train_zip/train/imgs")
    test_data_dir = pathlib.Path("fruits_dataset/test_zip/test/imgs")
    number_of_train_imgs = len(list(train_data_dir.glob('*/*.jpg')))
    number_of_test_imgs = len(list(test_data_dir.glob('*/*.jpg')))

    print(f"Loaded {number_of_train_imgs} train images.")
    print(f"Loaded {number_of_test_imgs} test images.")

    # Loader parameters
    parameters = LoaderParameters(batch_size=32, img_height=180, img_width=180)

    train_ds, val_ds = load_data(train_data_dir, test_data_dir, parameters)
    class_names = train_ds.class_names
    print(f"Detected classes: {class_names}")

    # Setup data cache
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Data normalization
    normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    # Notice the pixels values are now in `[0,1]`.
    print(np.min(first_image), np.max(first_image))

    print("""
    Select option:
        1. Create and train new model.
        2. Load existing model (./model_weights).
    """)
    while True:
        input_value = input("-> ")
        if input_value == '1':
            model = create_model(len(class_names), train_ds, val_ds, parameters)
            break
        elif input_value == '2':
            model = load_model(len(class_names), parameters)
            break
        else:
            print("Incorrect input, try again.")
            continue

    ### TEST ###

    final_test_data_dir = pathlib.Path("./final_tests")
    paths = list(final_test_data_dir.glob("*.jpg"))

    for path in paths:
        str_path = str(path)
        print(f"Analysing {str_path}...")
        img = keras.preprocessing.image.load_img(
            str_path, target_size=(parameters.img_height, parameters.img_width)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
        )
