from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import smart_resize
import tensorflow as tf
import os
from PIL import Image
import csv


def main():
    # Let's see the accuracy on validation set.
    model = load_model("model/river_classifier_model.tf")
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

    # print("Evaluate on test data")
    # results = model.evaluate(tf_validation_dataset, batch_size=128)
    # print("test loss, test acc:", results)

    # Ok, now we load data from test set and label them.
    images = os.listdir("data/test_data/")
    dc = {}
    for i in images:
        with Image.open("data/test_data/" + i) as im:
            resized_im = smart_resize(tf.keras.preprocessing.image.img_to_array(im), (256, 256))
            infer_input = resized_im[tf.newaxis, :, :, :3]
            predict = tf.argmax(model.predict(infer_input), axis=1)[0]
            print(i + ": " + str(predict.numpy()))
            dc[i] = str(predict.numpy())

    with open('dev/test_data.csv', 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['image_name', 'label'])
        for k in dc:
            csv_writer.writerow([k, dc[k]])


if __name__ == "__main__":
    main()
