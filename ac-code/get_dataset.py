#!/usr/bin/python
"""Apparel Attribute Classification: get_dataset"""

# import the necessary packages
import os
import sys
import urllib
import pandas as pd

# doc string
__author__ = "Sanchit Aggarwal"
__email__ = "sanchitagarwal108@gmail.com"


"""
# Function to download dataset
"""
def get_dataset(datafile, label_column, url_column, dataset_path):
    data_df = pd.read_csv(datafile, header=0, delimiter="\t", quoting=3, error_bad_lines=False)
    print "data shape:", data_df.shape
    print "data columns:", data_df.columns.values
    print data_df

    # to reverse dataset
    # reversed_df = data_df.iloc[::-1]

    print "creating dataset directory..."
    dataset_directory = os.path.abspath(dataset_path)
    if not os.path.exists(dataset_directory):
        os.mkdir(dataset_directory)

    target_labels =  list(set(data_df.ix[:,label_column]))
    print "target labels: ", target_labels

    print "creating label directories..."
    for label in target_labels:
        label_directory = os.path.join(dataset_directory,label)
        if not os.path.exists(label_directory):
            os.mkdir(label_directory)

    for index, row in data_df.iterrows():
        print "%s downloading image.. %s - %s " %(index, row[label_column],row[url_column])
        label_directory = os.path.join(dataset_directory,row[label_column])
        image_name = os.path.join(label_directory, row[url_column].split('/')[-1])
        if not os.path.exists(image_name):
            urllib.urlretrieve(row[url_column], image_name)


if __name__ == '__main__':
    # read the data file (training or testing)
    print "reading data file..."
    datafile = sys.argv[1]
    label_column = int(sys.argv[2])
    url_column = int(sys.argv[3])
    dataset_path = sys.argv[4]

    print "label_column: ", label_column
    print "url_column: ", url_column
    print "data file: ", datafile
    print "dataset path: ", dataset_path

    # download dataset
    get_dataset(datafile, label_column, url_column, dataset_path)
