import csv
import os
import shutil


def main():
    dc_old = {}
    dc_new = {}
    with open("dev/test_data.csv") as csv_file:
        rd = csv.reader(csv_file)
        # we construct a dict, keyed by file name and value is its category.
        for row in rd:
            # skip the first row
            if row[1] == 'label':
                continue
            dc_old[row[0]] = row[1]
    with open("dev/test_data_new.csv") as csv_file:
        rd = csv.reader(csv_file)
        # we construct a dict, keyed by file name and value is its category.
        for row in rd:
            # skip the first row
            if row[1] == 'label':
                continue
            dc_new[row[0]] = row[1]

    count_diff = 0
    for k in dc_old:
        if dc_old[k] != dc_new[k]:
            print(f"{k}, old: {dc_old[k]}, new: {dc_new[k]}")
            count_diff += 1
    print(f"diff: {count_diff}")


if __name__ == "__main__":
    main()
