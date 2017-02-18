## **Apparel Attribute Classification** ##

To identify the attribute of the Apparel in the Image.

### Datasets ###
Attribute Annotated Dataset. Find the sample sheet [here](https://github.com/SanchitAggarwal/apparel-classifiation/tree/master/data).


### Requirements ###

 - python 2.7
 - opencv
 - python modules:
	 - pandas

### The Code ###
Clone the repository.

``` sh
git clone https://github.com/SanchitAggarwal/apparel-classifiation
cd apparel-classifiation
```

For training the model, run:
``` sh
python  ac_main.py -t training_dataset_folder -v 1 -s 0.3
```

This will save a `model.pkl` file at the root folder.

For testing the model, run:
``` sh
python  ac_main.py -m model.pkl -p prediction_dataset_folder
```

For simultaneously training and validation and testing, run:
``` sh
python ac_main.py -t training_dataset_folder -m model.pkl -b codebook_path -p prediction_dataset_folder -v 1 -s 0.3
```

### Approach ###
The problem is to classify apparel according to the curve of the neck. The very first thing is to represent the curve/shape of the neck using efficient feature representation. Since Histogram of Oriented Gradient ( HoG ) is known for shape representation. We can try formulating the problem using HoG image of the Neck region in each image.

To segment the neck area from each image, we need to first determine the person upper body. This can be achieved by applying Upper body detector o locate humans in the natural scenes. Once we have region of interest segmented from each image we can represent them using Bag of Words over HoG features, which can then be fed to relevant classifiers.

Since the robust classification of clothing in natural scene is a non-trivial and complex problem, we choose Support Vector Machine for classification purpose.

#### Pre-processing ####
Performed pre-processing of images to extract upper body parts from the images. Non-maxima Suppression is used to get the best hypothesis for the upper body part. The bounding region is then used to extract features like HoG, SURF.

To remove the false positives of Upper Body detection we find a frontal face inside the upper body. The hypothesis without frontal faces are false positives and can be removed.

``` python
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
```

#### Feature Extraction and Bag of Words Representation: ####
Features: We use HoG of the detected region.

Coding: A codebook is learned by K-Means, further all features are vector quantized using this codebook.

Pooling: Finally, the quantized vectors are then spatially pooled with spatial pyramids and max-pooling is applied to the histograms.

``` python
"""
# Function to compute codebook
"""
def computeCodebook(dataframe):
    total_keypoints = dataframe["keypoints"].sum()
    print "total keypoints", total_keypoints
    nclusters = int(sqrt(total_keypoints))
    print "total clusters", nclusters
    feature = zeros((1,dataframe['descriptor_width'].iloc[0]))
    for index, row in dataframe.iterrows():
        discriptor = row["descriptors"].reshape(row["keypoints"],row["descriptor_width"])
        feature = concatenate((feature,discriptor),axis=0)
    feature =  feature[1:]
    print "feature generated!"
    print "===============================FEATURE=========================================="
    print feature
    codebook, distortion = vq.kmeans(feature,nclusters,thresh=k_thresh)
    return codebook
```

#### Improvements:####
Can detect better upper body hypothesis by combining a well known Calvin upper body detector [here] (http://groups.inf.ed.ac.uk/calvin/calvin_upperbody_detector/) other than Haar based upper body detector.

Further we can add features for B-Spline detection which are extensively used for curve matching.
We can also use other features like SURF, Self Similarity (SSD), Local Binary Patterns (LBP) and use the combination of all to learn the initial codebook.
We can try Random Forest with SVM for node splitting for further improvement in classification.


### Experiments ###
Performed different experiments for feature selection and classifier selection. For all the experiments we divided the training data into training set and validation set with a validation set size of 0.3

``` python
    if split > 0:
          # split into training and validation set
          print "splitting data into training and validation set"
          training_set, validation_set = train_test_split(feature_dataframe, test_size = split)
          print training_set.shape
          print validation_set.shape
      else:
          training_set =  feature_dataframe

```

#### **Experiment 1:** *HoG Features + SVM - 30% Validation* ####
For a quick evaluation purpose, We just used two classes U neck and V neck with 20, 20 images in each set.

*Results*:

|               |precision|  recall|  f1-score|  support|
| ------------- |:-------:| ------:| --------:|--------:|
|**Boat Neck**   |0.94      |0.88      |0.91       |455|
|**chinese collar neck**       |0.95      |0.72      |0.82       |327|
|**low cut neck**     |0.87      |0.81      |0.84       |349|
|**Round Collar**  |0.97      |0.85      |0.91       |153|
|**Round Neck**     |0.98      |0.94      |0.96       |568|
|**Square Collar**     |0.85      |0.98      |0.91      |1295|
|**Square Neck**    |0.84      |0.61      |0.71       |132|
|**U Neck**   |0.91      |0.93      |0.92       |902|
|**V Neck**   |0.91      |0.93      |0.92       |902|
|**avg / total**|**0.91**      |**0.90**      |**0.90**      |**4181**|


precision    recall  f1-score   support

U neck       1.00      0.50      0.67         2
V Neck       0.80      1.00      0.89         4

avg / total       0.87      0.83      0.81         6


#### **Conclusion** ###
A very basic pipeline for Apparel Attribute classification in Natural scenes is presented here. The algorithm first determine relevant region for Neck type classification by utilizing state of the art upper body detector. A HoG representation of the region is then used to learn a Bag of Words over Support Vector Machines.
