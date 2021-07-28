import tensorflow as tf

tf_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "formatted_data",
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="nearest",
    follow_links=False,
)
