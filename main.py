import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model


def main():
    tf_training_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "formatted_data",
        labels="inferred",
        label_mode="categorical",
        class_names=None,
        color_mode="rgb",
        batch_size=32,
        image_size=(256, 256),
        shuffle=True,
        seed=42,
        validation_split=0.1,
        subset="training",
        interpolation="nearest",
        follow_links=False,
        smart_resize=True
    )

    tf_validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "formatted_data",
        labels="inferred",
        label_mode="categorical",
        class_names=None,
        color_mode="rgb",
        batch_size=32,
        image_size=(256, 256),
        shuffle=True,
        seed=42,
        validation_split=0.1,
        subset="validation",
        interpolation="nearest",
        follow_links=False,
        smart_resize=True
    )

    base_model = InceptionV3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- we have 4 classes
    predictions = Dense(4, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                  metrics=[tf.keras.metrics.CategoricalCrossentropy()])

    model.fit(tf_training_dataset, epochs=3)


if __name__ == "__main__":
    main()
