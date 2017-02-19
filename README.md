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
    total_features = len(dataframe)
    print "total keypoints", total_features
    nclusters = int(sqrt(total_features))
    print "total clusters", nclusters
    features = array(dataframe['features'])
    features = np.concatenate(features).astype(None)
    print features
    codebook, distortion = vq.kmeans(features,nclusters,thresh=k_thresh)
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

#### **Experiment 1:** *Bag of Words on HoG Features + SVM - 30% Validation* ####
For a quick evaluation purpose and checking the pipeline, We just used two classes U neck and V neck with 20, 20 images in each set.

*Results*:

|               |precision|  recall|  f1-score|  support|
| ------------- |:-------:| ------:| --------:|--------:|
|**U Neck**   |  0.67        |0.80        | 0.73         |5|
|**V Neck**   |0.83     | 0.71         | 0.77      |7|
|**avg / total**|**0.76**      |**0.75 **      |**0.75 **      |**12**|


#### **Experiment 2:** *Bag of Words on HoG Features + SVM - 30% Validation* ####
Finally, We run the pipeline on 5K images for all the classes. The model is performing really bad. Hence We tried some improvements.

*Results*:

|                      |precision|  recall|  f1-score|  support|
| -------------        |:-------:| ------:| --------:|--------:|
|**Round Neck**        |  0.00     |0.00    | 0.00     |113        |
|**U Neck**            |  0.00     |0.00    | 0.00     |24        |
|**Boat neck**         |  0.00     |0.00    | 0.00     |75        |
|**chinese collarneck**|  0.00     |0.00    | 0.00     |253        |
|**Low cut neck**      |  0.00     |0.00    | 0.00     |78        |
|**V Neck**            |  0.00       |0.00    | 0.00     |45        |
|**Square Neck**       |  0.00     |0.00    | 0.00     |2        |
|**Square collar**     |  0.00     |0.00    | 0.00     |340        |
|**Round collar**      |  0.00     |0.00    | 0.00     |234        |
|**Collar neck **      |  0.23     |1.00    | 0.37     |339        |
|**avg / total**       |**0.05**   |**0.23**|**0.08**  |**1503**   |

#### **Conclusion** ###
A very basic pipeline for Apparel Attribute classification in Natural scenes is presented here. The algorithm first determine relevant region for Neck type classification by utilizing state of the art upper body detector. A HoG representation of the region is then used to learn a Bag of Words over Support Vector Machines.
