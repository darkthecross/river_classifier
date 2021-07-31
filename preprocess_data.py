""" This script takes data/train_data as input, and re-arrange files as desired file tree.

Usage:
1. Make sure you have data/train_data as the correct path.
2. Run `python3 preprocess_data.py`
"""

import csv
import os
import shutil


def main():
    dc = {}
    cat = []
    with open("data/train_data/train_label.csv") as csv_file:
        rd = csv.reader(csv_file)
        # we construct a dict, keyed by file name and value is its category.
        for row in rd:
            # skip the first row
            if row[1] != 'label':
                dc[row[0]] = row[1]
                if not row[1] in cat:
                    cat.append(row[1])

    # There are some images which are not recognizable in tf.
    bad_list = ["cELTM.jpg", "ChxbM.jpg", "CnGVg.jpg", "Ksebf.jpg", "LjFyd.jpg", "Lxulv.jpg", "MqotO.jpg", "NhvrR.jpg",
                "qmzFQ.jpg", "THovD.jpg", "TuPQk.jpg", "uHjEZ.jpg", "TMkCF.jpg", 'zYogd.jpg']

    # Clean formatted_data/
    dirs = os.scandir()
    for d in dirs:
        if d.name == "formatted_data":
            shutil.rmtree("formatted_data/")

    os.mkdir("formatted_data/")
    for c in cat:
        os.mkdir("formatted_data/" + c + "/")

    images = os.listdir("data/train_data/train_image/")
    for i in images:
        if i not in dc:
            raise Exception("Image " + i + " unlabelled!")
        if i in bad_list:
            print("Excluded " + i)
        else:
            image_category = dc[i]
            shutil.copy("data/train_data/train_image/" + i, "formatted_data/" + image_category + "/" + i)


if __name__ == "__main__":
    main()
