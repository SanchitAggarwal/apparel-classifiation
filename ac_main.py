#!/usr/bin/python
"""Apparel Attribute Classification"""

# import the necessary packages
import os
import sys
import urllib
import pandas as pd
import cv2
import argparse
import skimage
import numpy as np
import scipy.cluster.vq as vq
from scipy.cluster.vq import whiten
from imutils.object_detection import non_max_suppression
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from numpy import array, zeros, resize, sqrt, histogram, hstack, vstack, savetxt, zeros_like, concatenate
import time
from datetime import datetime
import cPickle

# doc string
__author__ = "Sanchit Aggarwal"
__email__ = "sanchitagarwal108@gmail.com"


# Initiate SIFT detector
print "initializing sift detector"
sift = cv2.SIFT()  #obsolete in  opencv 3.1.0
# sift = cv2.xfeatures2d.SIFT_create()

frontalface_cascade_path = './haarcascades/haarcascade_frontalface_default.xml'
upperbody_cascade_path = './haarcascades/haarcascade_mcs_upperbody.xml'
ml_model_path = os.path.join(os.getcwd(),'ml_model/')
output_path = os.path.join(os.getcwd(),'output/')
preprocess_path = os.path.join(os.getcwd(),'preprocess')
size = (150,150)
k_thresh = 1 # early stopping threshold for kmeans originally at 1e-5, increased for speedup

if not os.path.exists(ml_model_path):
    os.mkdir(ml_model_path)

if not os.path.exists(output_path):
    os.mkdir(output_path)

if not os.path.exists(preprocess_path):
    os.mkdir(preprocess_path)


"""
parse the command line arguments
"""
def parseArguments():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    # this is the root folder for different creatives.
    ap.add_argument("-t", "--training_dataset_folder",required=False, help="Folder for Apparel Atribute Training Dataset.")
    ap.add_argument("-p", "--prediction_dataset_folder",required=False, help="Folder for the images to be classified.")
    ap.add_argument("-m", "--model_path",required=False, help="Path of saved model.")
    ap.add_argument("-b", "--codebook_path",required=False, help="Path of saved codebook.")
    ap.add_argument("-v", "--validation",required=False, help="1/0 validation during training")
    ap.add_argument("-s", "--split",required=False, help="split ratio, range 0-1")

    # parsing the arguments
    args = vars(ap.parse_args())
    # printing arguments
    for keys,values in args.items():
        print("Parsing Arguments: {} : {}".format(keys,values))
    if args["training_dataset_folder"] is None and args["model_path"] is None:
        ap.error('-- either of one (training_dataset_folder/model_path) is required')
    if args["model_path"] is not None and args["codebook_path"] is None:
        ap.error('-- both model_path and codebook_path are required')
    if args["model_path"] is not None and args["test_dataset_folder"] is None:
        ap.error('-- both model_path and test_dataset_folder are required')
    return args

"""
return the file list for given extension in given directory
"""
def getFileList(directory, extension = "jpg"):
    print "reading directory for %s images" % (extension)
    labels = []
    files = []
    directory =  directory.strip()
    print os.path.abspath(directory)
    for path,dirs,file_list in os.walk(os.path.abspath(directory)):
        print file_list
        print path
        print dirs
        for fi in file_list:
            if fi.endswith(extension):
                files.append(os.path.join(path,fi))
                labels.append(path.split('/')[-1])
    print files
    print labels
    return files,labels

"""
To predict and crop upper body from image.
"""
def getBodyPart(image, cascade, min_size = (30,30)):
    # load cascade
    bodypart_cascade = cv2.CascadeClassifier(cascade)
    bodyparts = bodypart_cascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=min_size,
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    # apply non-maxima suppression to the bounding boxes
    # Threshold is used to pick boxes which are fairly overlapping
    overlapping_bodyparts = np.array([[x, y, x + w, y + h] for (x, y, w, h) in bodyparts])
    final_bodyparts = non_max_suppression(overlapping_bodyparts, probs=None, overlapThresh=0.65)
    final_bodyparts = np.array([[x, y, w - x, h - y] for (x, y, w, h) in final_bodyparts])
    print ("Number of Body Parts Detected: {}".format(len(final_bodyparts)))
    return final_bodyparts

'''
Function to extract SIFT descriptors
'''
def getSIFT(image):
    # extracting sift keypoints and descriptors
    print "extracting sift keypoints and descriptors..."
    # find the keypoints and descriptors with SIFT
    keypoints, descriptors = sift.detectAndCompute(training_image, None)

'''
Function to extract HoG descriptors
'''
def getHoG(image):
    winSize = (64,64)
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    winStride = (8,8)
    padding = (8,8)
    locations = ((10,20),)
    hist = hog.compute(image,winStride,padding,locations)
    # skimage.feature.hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), block_norm='L1', visualise=False, transform_sqrt=False, feature_vector=True, normalise=None)
    # print len(hist.flatten())
    return hist.flatten()

'''
Function to preprocess image detecting upperbody parts
'''
def preprocess_image(image, filename):
    cropped_upperbody_image = np.empty((0))
    upperbody = getBodyPart(image,upperbody_cascade_path, (30,30))
    for body in upperbody:
        cropped_upperbody_image = cropImage(image, body)
        face = getBodyPart(cropped_upperbody_image, frontalface_cascade_path, (30,30))
        if len(face)>0:
            cropped_upperbody_image = cv2.resize(cropped_upperbody_image, size)
            cropped_image_file_name = os.path.join(preprocess_path, 'cropped_'+ filename)
            cv2.imwrite(cropped_image_file_name,cropped_upperbody_image)
            # upperbody_image = drawRectangle(image, body, 0, 0, 255)
            # upperbody_image = putText(upperbody_image, "Upperbody", body[0], body[1],0,255,0)
            # image_file_name = os.path.join(preprocess_path, filename)
            # cv2.imwrite(image_file_name,upperbody_image)
            break
    return cropped_upperbody_image

"""
# Function to extract features
"""
def extractFeatures(imagefiles,labels):
    print "extracting features..."
    feature_label_list = []
    for index in range(len(imagefiles)):
        print "%s calculating features for %s" %(index, imagefiles[index])
        training_image = preprocess_image(cv2.imread(imagefiles[index]),imagefiles[index].split('/')[-1])
        if training_image.size:
            feature_label_list.append([imagefiles[index].split('/')[-1],labels[index], getHoG(training_image),[]])

    # get the data frame for training data
    print "----Creating Data Frame----"
    dataframe = pd.DataFrame(feature_label_list ,columns=['files','labels','features','histograms'])
    return dataframe

"""
# Function to compute codebook
"""
def computeCodebook(dataframe):
    total_features = len(dataframe)
    print "total keypoints", total_features
    nclusters = int(sqrt(total_features))
    print "total clusters", nclusters
    features = array(dataframe['features'])
    features = np.concatenate(features).astype(None)
    print features
    codebook, distortion = vq.kmeans(features,nclusters,thresh=k_thresh)
    return codebook

"""
# Function to compute histogram
"""
def computeHistogram(dataframe,codebook):
    for index, row in dataframe.iterrows():
        feature = row["features"]
        code, dist = vq.vq(feature, codebook)
        word_histogram, bin_edges = histogram(code, bins=range(codebook.shape[0] + 1), normed=True)
        dataframe.loc[index,"histograms"] = word_histogram
    return dataframe

"""
# Function to train the model for given pipeline
"""
def trainModel(training_set, pipeline):
    # Learning Model
    print "learning model"
    f = vstack(training_set["histograms"].values)
    print f
    model = pipeline.fit(f,  training_set["labels"])
    return model

"""
# Function to predict from creatives from learned model
"""
def predictModel(test_set,model):
    f = vstack(test_set["histograms"].values)
    print f
    predicted = model.predict(f)
    return predicted

"""
# Function to save the model
"""
def saveModel(model, filename):
    # save the classifier
    with open(filename, 'wb') as fid:
        cPickle.dump(model, fid)

"""
# Function to read the model
"""
def readModel(filename):
    model = None
    # load model
    if os.path.isfile(filename):
        with open(filename, 'rb') as fid:
            model = cPickle.load(fid)
    return model

"""
To crop the given region in the image.
"""
def cropImage( image, region):
    crop = image[region[1]:region[1] + region[3], region[0]:region[0]+region[2]] #img[y: y + h, x: x + w]
    return crop

"""
To draw rectangle in the image.
"""
def drawRectangle(image, region, r,g,b):
    # Draw a rectangle around the faces
    cv2.rectangle(image, (region[0], region[1]), (region[0]+region[2], region[1]+region[3]), (r, g, b), 2)
    return image

"""
To Put text on the given location of the image.
"""
def putText(image,text_string,x,y,r,g,b):
	cv2.putText(image,text_string, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (r,g,b))
	return image

'''
Function for prediction and saving results
'''
def prediction(data_set, model, codebook, target_labels):
    print "----prediction----"
    # computing the visual word histogram for each image using the codebook
    print "----Computing the histograms---"
    data_set = computeHistogram(data_set,codebook)
    print "first 5 histogram"
    print data_set.head(5)

    data_set["predicted"] = predictModel(data_set,model)
    print data_set["predicted"]

    print(classification_report(data_set["labels"], data_set["predicted"], target_names = target_labels))

     # Save results to output object
    print "----Saving Results----"
    output = pd.DataFrame( data={"image":data_set['files'], "actual_label":data_set['labels'], "predicted_label":data_set['predicted']} )
    filename = output_path + '%s_output.csv'%datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    output.to_csv(filename, index=False, sep='\t', quoting=3 )

'''
Function to train model and codebook from given labelled images
'''
def training(training_image_folder, split):
    ################## Dataset ############################################
    print "----Reading Image List----"
    imagefiles,labels = getFileList(training_image_folder)

    ################## Feature Extraction #################################
    # extract descriptors for the images (compute features)
    print "----Extracting Features----"
    feature_dataframe = extractFeatures(imagefiles,labels)
    print "data shape:", feature_dataframe.shape
    print "data columns:", feature_dataframe.columns.values
    print "first 5 descriptors"
    print feature_dataframe.head(5)

    ################## Training and Validation Split #################################
    if split > 0:
        # split into training and validation set
        print "splitting data into training and validation set"
        training_set, validation_set = train_test_split(feature_dataframe, test_size = split)
        print training_set.shape
        print validation_set.shape
    else:
        training_set =  feature_dataframe


    # ################## Coding and Pooling #################################
    # computing the codebook for visual bag-of-words
    print "----Computing the codebook---"
    codebook = computeCodebook(training_set)

    # save codebook
    print "----Saving Codebook----"
    filename = ml_model_path + '%s_coodebook.pkl'%datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    saveModel(codebook,filename)

    # computing the visual word histogram for each image using the codebook
    print "----Computing the histograms---"
    training_set = computeHistogram(training_set,codebook)
    print "first 5 histogram"
    print training_set.head(5)

    # define the pipeline
    pipeline = Pipeline([
    ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))])

    # get training model
    model = trainModel(training_set, pipeline)
    print model
    # save model
    print "----Saving Model----"
    filename = ml_model_path + '%s_model.pkl'%datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    saveModel(model,filename)


    # predict for validation set
    if split > 0:
        print "----Validation and Classification Report----"
        target_labels =  list(set(training_set["labels"]))
        prediction(validation_set, model, codebook, target_labels)

'''
Function to predict the class labels from test images
'''
def testing(query_image_folder, model, codebook):
    print "----Reading Image List----"
    imagefiles,labels = getFileList(query_image_folder)

    ################## Feature Extraction #################################
    # extract descriptors for the images (compute features)
    print "----Extracting Features----"
    feature_dataframe = extractFeatures(imagefiles,labels)
    print "data shape:", feature_dataframe.shape
    print "data columns:", feature_dataframe.columns.values
    print "first 5 descriptors"
    print feature_dataframe.head(5)

    target_labels =  list(set(labels))
    prediction(feature_dataframe, model, codebook, target_labels)

if __name__ == '__main__':
    # parsing arguments
    args = parseArguments()
    # checking for training data folder
    if args["training_dataset_folder"] is not None:
        split = 0
        if args["validation"] is not None and int(args["validation"]) == 1:
            if args["split"] is not None:
                split = float(args["split"])
            else:
                split = 0.3 #default

        training(args["training_dataset_folder"],split)

    if args["model_path"] is not None and args["codebook_path"] is not None and args["test_dataset_folder"] is not None:
        model = readModel(args["model_path"])
        codebook = readModel(args["codebook_path"])
        query_image_folder = args["test_dataset_folder"]
        testing(test_dataset_folder, model, codebook)
