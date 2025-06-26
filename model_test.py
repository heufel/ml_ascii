import tensorflow as tf
from tensorflow import keras
import time
import numpy as np

if __name__ == "__main__":
    start = time.time()
    test_ds = tf.keras.utils.image_dataset_from_directory("curated/", validation_split = 0.2, seed=np.random.random_integers(0, 1000), label_mode='categorical', color_mode='grayscale', batch_size=32, image_size=(64,64), subset="validation")
    normalization = keras.layers.Rescaling(1.0/255.0)
    norm_test_ds = test_ds.map(lambda x, y: (normalization(x), y))
    model = keras.models.load_model("ascii_reader.keras")
    loss, acc = model.evaluate(norm_test_ds)
    print(f'Test loss = {loss:.4}')
    print(f'Test accuracy = {acc:.2%}')
    print(f"Test took {time.time()-start:.3} seconds.")