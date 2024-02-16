# Identification of Smartphone Activities Based on Smartphone Movement Sensors

## Overview

This project aims at identifying smartphone activities (typing, scrolling, watching) solely from inertial smartphone sensors
(accelerometer, gyroscope, and magnetometer). The model reaches a balanced accuracy of 78.5% which is a significant improvement over the baseline accuracy of 16.7%.

This project is inspired by the work of Sijie Zhuo et al. in their paper 'Real-time Smartphone Activity Classification Using Inertial Sensors—Recognition of Scrolling, Typing, and Watching Videos While Sitting or Walking.'"

This project attempts to replicate and expand upon the findings of the paper, using the dataset provided by Zhuo et al. as a foundation. The dataset comprises raw sensor data captured from smartphone activities, which has been processed to classify various states of user interaction, whether the user is sitting or walking.

The dataset used for this project includes sensor data that is labeled for the following activities: 'Sitting_Idle', 'Sitting_Scroll', 'Sitting_Type', 'Sitting_Watch', 'Walking_Idle', 'Walking_Scroll', 'Walking_Type', and 'Walking_Watch'. These labels are used to train a model to predict user activity based on new sensor data inputs.

For further reading and to gain a comprehensive understanding of the research this project is based upon, you can access the paper and dataset through the following links:

[Paper](https://www.mdpi.com/1424-8220/20/3/655)

[Dataset](https://www.mdpi.com/1424-8220/20/3/655/s1)

## Results 
### Model Performance

The model's performance can be assessed using the provided confusion matrices, which detail the classification results 
for the six activities: Sitting_Idle, Sitting_Type, Sitting_Watch, Sitting_Scroll, Walking_Idle, and Walking_Scroll.

The overall balanced accuracy of the model is 0.785, indicating a strong ability to distinguish between the activities. 
This performance is especially noteworthy given the baseline performance of 0.167, which represents the accuracy of a random
classifier. 

However, there are areas where the model can confuse certain activities with others, such as Sitting_Idle with Sitting_Scroll and Walking_Idle with Walking_Scroll, where the latter activities in both pairs were mistaken for the former ones in a noticeable number of instances.

![alt text](https://github.com/benediktjordan/Smartphone-usage-identification-based-on-smartphone-movement-sensors/blob/34050217db157a692c798a466181e107f297442f/img/ConfusionMatrix.png)

### Important Features

The SHAP values highlight the most influential features in the model's predictions. The top features that impact the model's output include:
- Acceleration Y Change Quantiles with aggregation function "mean"
- Acceleration Z Benford Correlation
- Acceleration X Range Count
- Rotation X FFT Aggregated Variance

These features appear to have the most substantial impact on the model's decisions, with the Acceleration Y Change Quantiles being particularly prominent. The detailed SHAP values graph indicates that the model relies on a combination of sensor data capturing movement dynamics and statistical features derived from the sensor signals to make its predictions.

![alt text](https://github.com/benediktjordan/Smartphone-usage-identification-based-on-smartphone-movement-sensors/blob/34050217db157a692c798a466181e107f297442f/img/SHAPValues.png)

In conclusion, the model exhibits a promising ability to differentiate between various smartphone usage states with a set of key features providing the most predictive power. Continued refinement and training could further improve its performance, especially in distinguishing closely related activities.


## Data Structure

The dataset is derived from a comprehensive [study](https://www.mdpi.com/1424-8220/20/3/655) aimed at classifying smartphone user activities into active and passive states based on inertial sensor data. The study used an Android application to collect motion data during various activities such as scrolling through feeds, typing, and watching videos. These tasks were performed both while seated and walking, including a baseline of smartphone non-use for comparison.

The study engaged 21 participants, predominantly software engineering and computer science students from the University of Auckland. Participants ranged in age from 21 to 42, with an average age of 25.1 years. The diversity of the participant pool is critical for ensuring the generalizability of the study's findings.

Data was collected using the smartphone's inertial measurement unit (IMU) sensors, which include:
- Triaxial Accelerometers: To measure linear acceleration and detect motion. 
- Gyroscopes: For capturing orientation and rotation. 
- Magnetometers: To determine the magnetic field orientation, aiding in directional bearing.


The sensor data were sampled at two frequencies, 5Hz and 50Hz, which provided a detailed capture of movements for the diverse activities ranging from subtle to significant.

For an in-depth examination of the dataset and methodology, please refer to the original study by Zhuo et al. (2020) 
available [here](https://www.mdpi.com/1424-8220/20/3/655).

## Data Preprocessing 
In this project, we undergo several preprocessing steps to prepare the sensor data for feature extraction and model training. Below we detail each step taken during the preprocessing phase.

### Dropping Duplicates and Time Transformation

Initially, we handle the raw data by transforming the time column to a consistent datetime format and resetting the index for easier manipulation. We then proceed to remove duplicate entries based on all columns except the time-related ones. This ensures that our dataset is free from redundancies and reflects unique sensor readings. The duplicate removal process is conducted separately for each user and activity to maintain the granularity of the data.

### Feature Extraction

We employ the tsfresh library to extract relevant features from the sensor data. The extraction process is chunked to manage computational resources effectively, and NaN values are dropped to maintain data quality. We segment the data into various time intervals and extract features accordingly. In this step, we also ensure that a sufficient percentage of data is available for each segment; otherwise, the segment is excluded from further analysis.

### Feature Selection

Post feature extraction, we proceed with feature selection to refine our dataset to the most informative variables. This is performed using tsfresh's feature selection algorithm, which filters out irrelevant or redundant features based on their significance to the activity labels. We serialize the selected features for subsequent use in model training.

Each preprocessing step is critical for ensuring that the machine learning model is trained on high-quality, representative data. The removal of duplicates and transformation of the time column ensure temporal consistency. Feature extraction and selection help in distilling the sensor data down to the most relevant information for activity recognition.


## Modeling Process
The modeling process involves training a Decision Forest (DF) model to classify smartphone activities based on sensor data. This process can be subdivided into three parts: model training, parameter tuning and model deployment.

### Model Training 
A decision forest, which is an ensemble of decision trees, is utilized to improve predictive accuracy and control over-fitting. The decision forest model is known for its ability to handle a large dataset with higher dimensionality and provides a more generalized model by combining the predictions from multiple decision trees.

Leave-One-Subject-Out Cross-Validation (LOSOCV) is employed as the cross-validation technique. In this approach, the dataset from each participant is used once as the test set while the data from all other participants serve as the training set. This process is iterated through all participants, ensuring that the model is tested on unseen data every time, which simulates a more realistic scenario where the model is applied to new users.

### Parameter Tuning

Parameter tuning is conducted to find the optimal model parameters that yield the best classification performance. The grid search space for parameter tuning includes various hyperparameters such as the number of estimators, maximum depth, minimum samples split, minimum samples leaf, maximum features, out-of-bag score, class weight, criterion, maximum samples, and bootstrap option​
. The Python code performs a grid search to explore different combinations of these parameters.

The process involves running the Decision Forest algorithm with different parameter sets. The results, including the performance metrics and predictions, are compiled and analyzed to select the optimal parameter set. This step is crucial for enhancing the model's accuracy and ensuring that it generalizes well to unseen data​

### Model Deployment

The deployment model is trained on the entire dataset without any cross-validation, parameter tuning, feature importance computation, or validation. This model is intended for real-world application, where it can classify activities based on real-time sensor data.

The deployment model is configured with a specific set of parameters, including the number of estimators, criterion, maximum depth, minimum samples split, and class weight. The model uses a combination of sensors (accelerometer, rotation, gravity) and is trained on data sampled at a 50Hz frequency. The data is segmented into 15-second intervals for feature extraction​
.

This final model is saved and can be deployed for real-time activity classification. The modeling process ensures that the model is trained with optimal parameters and is capable of making accurate predictions on new, unseen data.

## Installation

To run the project, you need to have Python installed on your machine. Additionally, certain libraries and dependencies are required, which can be installed via the following command:

```
pip install -r requirements.txt
```

## Usage

To use this project, first clone the repository to your local machine:

```
git clone [repository-link]
```

Navigate to the project directory and execute the main script:

```
cd [project-directory]
python main.py
```



## License
This project is licensed under the MIT License.
