import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
# import os

import matplotlib.pyplot as plt


def plot_metric(history, metric, save_name):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_' + metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics)
    plt.plot(epochs, val_metrics)
    plt.title('Training and validation ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_" + metric, 'val_' + metric])
    plt.savefig(save_name)


def get_data_gen():
    train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=None,
        shear_range=0.0,
        zoom_range=0.1,
        channel_shift_range=0.0,
        fill_mode="nearest",
        cval=0.0,
        horizontal_flip=True,
        validation_split=0.1,
    )

    return train_data_gen


def inspect_data():
    train_data_gen = get_data_gen()
    train_generator = train_data_gen.flow_from_directory("formatted_data", subset="training")
    b = train_generator.next()
    print(b)
    # numpy_data = b.as_numpy_iterator()
    # images = list(numpy_data)[0]
    # single_img = images[0][0, 1, :, :, :]
    # img = Image.fromarray(single_img, "RGB")
    # img.show()


def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    train_data_gen = get_data_gen()

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
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])

    history = model.fit(train_data_gen.flow_from_directory("formatted_data", subset="training"), epochs=30,
                        validation_data=train_data_gen.flow_from_directory("formatted_data", subset="validation"))

    plot_metric(history, "categorical_accuracy", "dev/pre_train.png")

    # at this point, the top layers are well trained and we can start fine-tuning
    # convolutional layers from inception V3. We will freeze the bottom N layers
    # and train the remaining top layers.

    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    # for i, layer in enumerate(base_model.layers):
    #     print(i, layer.name)

    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True

    model.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.9), loss='categorical_crossentropy',
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])

    history = model.fit(train_data_gen.flow_from_directory("formatted_data", subset="training"), epochs=60,
                        validation_data=train_data_gen.flow_from_directory("formatted_data", subset="validation"))

    plot_metric(history, "categorical_accuracy", "dev/fine_tune.png")

    # model.save("model/river_classifier_model.tf")


if __name__ == "__main__":
    main()
    # inspect_data()
