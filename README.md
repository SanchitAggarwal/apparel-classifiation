## **Apparel Attribute Classification** ##

To identify the attribute of the Apparel in the Image.

### Datasets ###
Attribute Annotated Dataset.


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
python  ac_main.py -m model.pkl -b codebook.pkl -p prediction_dataset_folder
```

For simultaneously training and validation and testing, run:
``` sh
python ac_main.py -t training_dataset_folder -m model.pkl -b codebook_path -p prediction_dataset_folder -v 1 -s 0.3
```

For Training Dataset, have a folder with following structure:
- training_dataset_folder
 - Class_1
 - Class_2
 - ...
 - Class N

For Prediction Dataset, have a folder with above structure or:
- prediction_dataset_folder
 - Unknown

### Approach ###
The problem is to classify apparel according to the curve of the neck. The very first thing is to represent the curve/shape of the neck using efficient feature representation. Since Histogram of Oriented Gradient ( HoG ) is known for shape representation. We can try formulating the problem using HoG image of the Neck region in each image.

To segment the neck area from each image, we need to first determine the person upper body. This can be achieved by applying Upper body detector to locate humans in the natural scenes. Once we have region of interest segmented from each image we can represent them using Bag of Words over HoG features, which can then be fed to relevant classifiers.

Since the robust classification of clothing in natural scene is a non-trivial and complex problem, we choose Support Vector Machine for classification purpose.

The code is divided into five major blocks.
- Dataset formation/downloading
- Pre-processing
- Feature Extraction
- Model Learning and validation
- Testing and Report Generation

#### Pre-processing ####
Performed pre-processing of images to extract upper body parts from the images. Non-maxima Suppression is used to get the best hypothesis for the upper body part. The bounding region is then used to extract features like HoG, SURF.

To remove the false positives of Upper Body detection we find a frontal face inside the upper body. The hypothesis without frontal faces are false positives and can be removed.

``` python
'''
Function to preprocess image detecting upperbody parts
'''
def preprocessImage(image, filename):
    face_detected_upperbody_image = np.empty((0))
    upperbody = getBodyPart(image,upperbody_cascade_path, (30,30))
    for body in upperbody:
        cropped_upperbody_image = cropImage(image, body)
        face = getBodyPart(cropped_upperbody_image, frontalface_cascade_path, (30,30))
        if len(face)>0:
            face_detected_upperbody_image = cv2.resize(cropped_upperbody_image, size)
            cropped_image_file_name = os.path.join(preprocess_path, 'cropped_'+ filename)
            cv2.imwrite(cropped_image_file_name,face_detected_upperbody_image)
            upperbody_image = drawRectangle(image, body, 0, 0, 255)
            upperbody_image = putText(upperbody_image, "Upperbody", body[0], body[1],0,255,0)
            image_file_name = os.path.join(preprocess_path, filename)
            cv2.imwrite(image_file_name,upperbody_image)
            break
    return face_detected_upperbody_image
```

#### Feature Extraction and Bag of Words Representation: ####
Features: We use HoG of the detected region. Cell Size: 2 x 2, Block Size: 4 x 4

Coding: A codebook is learned by K-Means, further all features are vector quantized using this codebook.

Pooling: Finally, the quantized vectors are then spatially pooled with spatial pyramids and max-pooling is applied to the histograms.

``` python
"""
# Function to compute codebook
"""
def computeCodebook(dataframe):
    total_features = len(dataframe)
    print "total keypoints", total_features
    nclusters = int(sqrt(no_of_clusters))
    print "total clusters", nclusters
    features = dataframe['features'].tolist()
    # print features
    features = np.vstack(features)
    print features
    print len(features)
    features = whiten(features)
    codebook, distortion = vq.kmeans(features,nclusters,thresh=k_thresh)
    return codebook
```


### Experiments ###
Performed different experiments for feature selection and classifier selection. For all the experiments we divided the training data into training set and validation set with a validation set size of 0.3

``` python
if split > 0:
      # split into training and validation set
      print "splitting data into training and validation set"
      training_set, validation_set = train_test_split(feature_dataframe.copy(), test_size = split)
      print training_set.shape
      print validation_set.shape
  else:
      training_set =  feature_dataframe.copy()


```

#### **Experiment 1:** *HoG Features + SVM - Binary Class - 30% Validation* ####
For a quick evaluation purpose and checking the pipeline, We just used two classes U neck and V neck with 20, 20 images in each set. HoG features are extracted and directly fed to SVM classifier.

*Results*:

|               |precision|  recall|  f1-score|  support|
| ------------- |:-------:| ------:| --------:|--------:|
|**U Neck**   |  1.00         |0.33        | 0.50        |3|
|**V Neck**   |0.71      | 1.00         | 0.83      |5|
|**avg / total**|**0.82**      |**0.75**      |**0.71**      |**8**|


#### **Experiment 2:** *HoG Features + SVM - Binary Class - 30% Validation* ####
Then we evaluated on the whole dataset containing 2K images of U Neck and V Neck. HoG features are extracted and directly fed to SVM classifier.

*Results*:

|               |precision|  recall|  f1-score|  support|
| ------------- |:-------:| ------:| --------:|--------:|
|**U Neck**   |  0.67        |0.97        | 0.80         |302|
|**V Neck**   |0.85     | 0.24        |0.38      |188|
|**avg / total**|**0.74**      |**0.64**      |**0.75**      |**490**|

#### **Experiment 3:** *HoG Features + Random forest - Binary Class - 30% Validation* ####
We also experimented with Random Forest classifier. We evaluated on dataset containing 2K images of U Neck and V Neck with HoG features.

*Results*:

|               |precision|  recall|  f1-score|  support|
| ------------- |:-------:| ------:| --------:|--------:|
|**U Neck**   |  0.70        |0.81        | 0.75         |308|
|**V Neck**   |0.56     | 0.41        |0.47      |182|
|**avg / total**|**0.65**      |**0.66**      |**0.65**      |**490**|


#### **Experiment 4:** *HoG Features + SVM - Binary Class - 30% Validation* ####
Then we evaluated on the whole dataset containing 5K images with all attribute classes. HoG features are extracted and directly fed to SVM classifier.

*Results*:

|                      |precision|  recall|  f1-score|  support|
| -------------        |:-------:| ------:| --------:|--------:|
|**Round Neck**        |  0.37     |0.23    | 0.28     |100        |
|**U Neck**            |  0.00     |0.00    | 0.00     |17        |
|**Boat neck**         |  0.26     |0.15    | 0.19     |82        |
|**chinese collarneck**|  0.41     |0.28   | 0.33    |231        |
|**Low cut neck**      |  0.40     |0.27    | 0.32     |64        |
|**V Neck**            |  0.11     |0.02    | 0.04     |47        |
|**Square Neck**       |  0.00     |0.00     | 0.00     |1        |
|**Square collar**     |   0.40     |0.64    | 0.49     |312        |
|**Round collar**      |  0.24     |0.40    | 0.31     |178        |
|**Collar neck**      |  0.71     |0.49    | 0.58     |297        |
|**avg / total**       |**0.42**   |**0.40**|**0.39**  |**1329**   |


#### **Experiment 5:** *Bag of Words on HoG Features  + SVM - 30% Validation* ####
To further improve the pipeline, we experimented with Bag of Words over Hog Features. We run the pipeline on 4 classes. We used 256 cluster for quick computation, The model is performing really bad and requires cluster tuning as BoW model is sensitive to number of clusters.

|                      |precision|  recall|  f1-score|  support|
| -------------        |:-------:| ------:| --------:|--------:|
|**U Neck**            |  0.00     |0.00    | 0.00     |107        |
|**V Neck**            |  0.00     |0.00    | 0.00     |44        |
|**Boat neck**         |  0.48     |1.00    | 0.65     |344        |
|**Square Neck**       |  0.23     |1.00    | 0.37     |216        |
|**avg / total**       |**0.23**   |**0.48**|**0.32**  |**711**   |


#### **Experiment 6:** *Bag of Words on HoG Features + SVM - 30% Validation* ####
Finally, We run the pipeline on 5K images for all the classes. The model is performing really bad. It has overfitted the Collar Neck Attribute. One way to improve this is to group the similar curve collars and then learn a hierarchical classifiers for same group collars with features like B-splines. In addition Parameter Tuning at different sections of the pipeline such as HoG extraction, BoW dictionary Learning, SVM training can further help improving the performance.
The results can be further improve by including other features and using different models such as Random Forests.

*Results*:

|                      |precision|  recall|  f1-score|  support|
| -------------        |:-------:| ------:| --------:|--------:|
|**Round Neck**        |  0.00     |0.00    | 0.00     |113        |
|**U Neck**            |  0.00     |0.00    | 0.00     |24        |
|**Boat neck**         |  0.00     |0.00    | 0.00     |75        |
|**chinese collarneck**|  0.00     |0.00    | 0.00     |253        |
|**Low cut neck**      |  0.00     |0.00    | 0.00     |78        |
|**V Neck**            |  0.00     |0.00    | 0.00     |45        |
|**Square Neck**       |  0.00     |0.00    | 0.00     |2        |
|**Square collar**     |  0.00     |0.00    | 0.00     |340        |
|**Round collar**      |  0.00     |0.00    | 0.00     |234        |
|**Collar neck**      |  0.23     |1.00    | 0.37     |339        |
|**avg / total**       |**0.05**   |**0.23**|**0.08**  |**1503**   |

#### **Experiment 7:** *HoG Features + SVM - 30% Validation (with HoG on Skin Region )* ####
We added a Skin Region detector and extracted HoG features only for Region with Skin in the image.
We fed these features to SVM Classifier. We evaluated the new pipeline on dataset containing 2K images of U Neck and V Neck.

*Results*:

|               |precision|  recall|  f1-score|  support|
| ------------- |:-------:| ------:| --------:|--------:|
|**U Neck**     |  0.77   |  0.75  | 0.76     |        298|
|**V Neck**     |0.63     | 0.66   | 0.64     |        192|
|**avg / total**|**0.71** |**0.71**|**0.71**  |    **490**|


#### **Conclusion** ###
- A very basic pipeline for Apparel Attribute classification in Natural scenes is presented here. The algorithm first determine relevant region for Neck type classification by utilizing state of the art upper body detector. A HoG representation of the region is then used to learn a Bag of Words with a layer of Support Vector Machines for final classification.

- It can be observed that the basic pipeline works well for two class problem to classify U Neck and V Neck but it drastically fails for a more complex attribute classification.

- The approach has a potential of doing better with higher resolution images and more sophisticated features and classifiers.

#### Improvements:####
- Can detect better upper body hypothesis by combining a well known Calvin upper body detector [here] (http://groups.inf.ed.ac.uk/calvin/calvin_upperbody_detector/) other than Haar based upper body detector.

- Further we can add features for **B-Spline** detection which are extensively used for curve matching.
We can also use other features like **SURF**, **Self Similarity (SSD)**, **Local Binary Patterns (LBP)** and use the combination of all to learn the initial codebook.

- To further enhance we can also find the percentage of skin Color (Histogram of colors) for different neck types as added features and do experiments over different moments of color tones.

- We can try Random Forest with SVM for node splitting for further improvement in classification.

- We can also try Convolution Neural Networks or can use CNN features in the pipeline.

#### **References** ###
[1) ECCV2012_ClothingAttributes.pdf] (http://chenlab.ece.cornell.edu/people/Andy/publications/ECCV2012_ClothingAttributes.pdf)

[2) Apparel Classification using CNNs] (http://cs231n.stanford.edu/reports2016/286_Report.pdf)

[3) Style Finder: Fine-Grained Clothing Style Recognition and Retrieval] (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.370.6283&rep=rep1&type=pdf)
