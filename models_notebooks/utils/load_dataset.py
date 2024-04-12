import pathlib
import tensorflow as tf

data_dir = pathlib.Path("/home/danil/Documents/курсовая/image_classification/"
                        "categories/all_cleaned/")

batch_size = 32
img_height = 256
img_width = 256


def load_dataset():
    train_ds = tf.keras.utils.image_dataset_from_directory(
      data_dir,
      validation_split=0.30,
      subset="training",
      label_mode='categorical',
      seed=2,
      image_size=(img_height, img_width),
      pad_to_aspect_ratio=True,
      batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
      data_dir,
      validation_split=0.30,
      subset="validation",
      label_mode='categorical',
      seed=2,
      image_size=(img_height, img_width),
      pad_to_aspect_ratio=True,
      batch_size=batch_size)

    val_batches = tf.data.experimental.cardinality(val_ds)
    test_ds = val_ds.take(val_batches // 2)
    val_ds = val_ds.skip(val_batches // 2)

    return train_ds, val_ds, test_ds
