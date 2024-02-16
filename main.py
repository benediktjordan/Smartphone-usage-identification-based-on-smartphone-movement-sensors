# This repo is about a project which tries to identify if phone users use their
# smartphone actively (i.e. typing) or passively (i.e. just holding it in their
# hand) based on sensor data.

#test

# This project is based on the paper and dataset Zhuo et al. (2019): Real-time Smartphone Activity Classification Using Inertial Sensors—Recognition of Scrolling, Typing, and Watching Videos While Sitting or Walking
## Paper: https://www.mdpi.com/1424-8220/20/3/655
## Dataset: https://www.mdpi.com/1424-8220/20/3/655/s1

# If you have any questions don´t hesitate to contact me through benedikt.jordan@posteo.de
#region import
import pandas as pd
import os
from tqdm import tqdm
import datetime
import time
import pickle





# visualization
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

# CoreML conversion
import coremltools
from coremltools.converters import sklearn


#endregion

#region load data
#region transform all 5 Hz and 50 Hz data into two dataframes
path_storage = "/Users/benediktjordan/Documents/WellspentWork/SmartphoneActivityIdentification/datasets/zhuo_paper/"
path_to_store = "/Users/benediktjordan/Documents/WellspentWork/SmartphoneActivityIdentification/data_analysis/dataset/"
frequencies = ["5Hz", "50Hz"]
for frequency in frequencies:
    print("started with frequency " + frequency + " data...")
    df_all = pd.DataFrame()
    # iterate through all folders within the frequency folder
    for folder in os.listdir(path_storage + frequency + "_data"):
        print("started with participant " + folder + "...")
        if folder == ".DS_Store":
            continue
        # iterate through all files within the folder
        for file in os.listdir(path_storage + frequency + "_data" + "/" + folder):
            print("started with file " + file + "...")
            # if there is "questions" in filename or filename doesnt end with .csv, skip
            if "questions" in file or file[-4:] != ".csv":
                continue
            # load data into dataframe
            df = pd.read_csv(path_storage + frequency + "_data" + "/" + folder + "/" + file)
            # add user id
            df["user_id"] = folder
            # add frequency
            df["sampling_rate"] = frequency
            # concatenate to df_all
            df_all = pd.concat([df_all, df], axis=0)
    # store df_all
    df_all.to_csv(path_to_store + frequency + "_data.csv", index=False)
#endregion

#endregion

#region data exploration
path_storage = "/Users/benediktjordan/Documents/WellspentWork/SmartphoneActivityIdentification/data_analysis/dataset/"
df_5Hz = pd.read_csv(path_storage + "5Hz_data.csv")
df_50Hz = pd.read_csv(path_storage + "50Hz_data.csv")

#TODO create summary stats visualizations

#temporary
# convert time column to datetime: format is "hh:mm:ss.mmm"
df_5Hz["datetime"] = pd.to_datetime(df_5Hz["time"], format="%H:%M:%S.%f")
#endregion

#region data preparation
#region preprocessing: drop duplicates and transform time column
frequencies = ["5Hz", "50Hz"]
path_storage = "/Users/benediktjordan/Documents/WellspentWork/SmartphoneActivityIdentification/data_analysis/dataset/"
path_to_store = "/Users/benediktjordan/Documents/WellspentWork/SmartphoneActivityIdentification/data_analysis/data_preparation/preprocessed_data/"
for frequency in frequencies:
    print("started with frequency " + frequency + " data...")
    df = pd.read_csv(path_storage + frequency + "_data.csv")
    # if entry in "time" has format "mm:ss.m", add "00:" to the beginning and "000" to the end
    df["time"] = df["time"].apply(lambda x: "00:" + x + "00" if len(x) == 7 else x)

    # convert time column to datetime: format is "hh:mm:ss.mmm"
    df["datetime"] = pd.to_datetime(df["time"], format="%H:%M:%S.%f")

    # reset index
    df.reset_index(inplace=True, drop=True)

    print("Size of dataframe before dropping duplicates: " + str(df.shape))
    for user_id in df["user_id"].unique():
        print("started with user " + str(user_id) + "...")
        # iterate through activities
        for activity in df["activity"].unique():
            #print("started with activity " + activity + "...")
            df_user_activity = df[(df["user_id"] == user_id) & (df["activity"] == activity)]
            if len(df_user_activity) == 0:
                continue
            # get indices to drop based on all columns except "time" and "datetime" and keep first
            indices_to_keep = df_user_activity.drop_duplicates(subset=df_user_activity.columns.drop(["time", "datetime"]), keep="first").index
            indices_to_drop = df_user_activity.drop(indices_to_keep).index

            print("duplicates dropped for user " + str(user_id) + " and activity " + activity + ": " + str(len(indices_to_drop)/len(df_user_activity)) + " duplicates")
            # drop duplicates
            df.drop(indices_to_drop, inplace=True)
    print("Size of dataframe after dropping duplicates: " + str(df.shape))

    # store df
    df.to_csv(path_to_store + frequency + "_time-transformed_duplicates-dropped.csv", index=False)

#endregion

#testarea
df_5Hz = pd.read_csv(path_to_store + "5Hz_time-transformed_duplicates-dropped.csv")
df_50Hz = pd.read_csv(path_to_store + "50Hz_time-transformed_duplicates-dropped.csv")

#region create tsfresh features
print(platform.processor())
frequencies = [50, 5]
feature_segments = [60, 30, 15, 10, 5, 2, 1]  # in seconds
feature_segments = [1]  # in seconds
exclusion_threshold_feature_segments = 0.6  # if less than xx% of the data is available for a feature segment, exclude it
label_column_name = "activity"

sensor_sets_to_extract = [["accelerometer", "rotation", "gravity"]]
path_storage = "/Users/benediktjordan/Documents/WellspentWork/SmartphoneActivityIdentification/data_analysis/data_preparation/preprocessed_data/"
path_to_store = "/Users/benediktjordan/Documents/WellspentWork/SmartphoneActivityIdentification/data_analysis/data_preparation/features/"

for sensor_frequency in frequencies:
    ## run feature extraction in chunks
    for sensor_set in sensor_sets_to_extract:
        path_sensorfile = path_storage + str(sensor_frequency) + "Hz_time-transformed_duplicates-dropped.csv"

        # create sensor columns
        sensor_column_names = []
        for sensor in sensor_set:
            sensor_column_names.append(database_sensor_columns[sensor])
        sensor_column_names = [item for sublist in sensor_column_names for item in sublist]

        for feature_segment in feature_segments:
            print("Start with Feature Segment: ", feature_segment)

            # iterate through sensor dataframe in steps of 500000
            chunksize_counter = 1
            if path_sensorfile.endswith(".csv"):
                print("path_sensorfile is .csv")
                for df_sensor in tqdm(pd.read_csv(path_sensorfile, chunksize=200000)):
                    # print the current time
                    print(f"date: {datetime.datetime.now()}")
                    # check if chunk was already computed
                    path_features = path_to_store + str(sensor_set) + "_sensor-frequency-" + str(sensor_frequency) + "Hz_timeperiod-" + str(
                        feature_segment) + " s_chunknumber-" + str(
                        chunksize_counter) + ".pkl"
                    if os.path.exists(path_features):
                        print("Jump over chunksize ", chunksize_counter, " in time_period ", feature_segment,
                              " and sensor ", str(sensor_set), " , was already computed, continuing with next chunk")
                        chunksize_counter += 1
                        continue

                    print("Start with chunk numer ", chunksize_counter)
                    time.start = time.time()

                    # drop rows which have at least one NaN value in any of the sensor_column_names columns
                    number_of_rows_before = df_sensor.shape[0]
                    df_sensor.dropna(subset=sensor_column_names, inplace=True)
                    percentage_of_rows_dropped = (number_of_rows_before - df_sensor.shape[0]) / number_of_rows_before
                    print("percentage of rows dropped due to NaN: ", str(percentage_of_rows_dropped * 100))
                    if percentage_of_rows_dropped == 1:
                        print("all rows dropped, continuing with next chunk")
                        continue
                    df_features = create_features.feature_extraction(df_sensor, sensor_column_names, feature_segment,
                                                                     sensor_frequency, exclusion_threshold_feature_segments,
                                                                     label_column_name, time_column_name="datetime")
                    # check if df_features is an empty DataFrame; if so, continues with next chunksize
                    if df_features.empty:
                        print("df_features is empty for chunksize ", chunksize_counter, " in time_period ",
                              feature_segment,
                              ", was removed, continuing with next chunk")
                        chunksize_counter += 1
                        continue

                    print(f"date: {datetime.datetime.now()}")
                    print(
                        "Time for " + str(feature_segment) + " seconds: " + str(
                            (time.time() - time.start) / 60) + " - without saving")

                    # save features with pickle
                    with open(path_features, 'wb') as f:
                        pickle.dump(df_features, f, pickle.HIGHEST_PROTOCOL)

                    # df_features.to_csv(dir_sensorfiles + "features/" + str(seconds) + " seconds_high_frequency_sensors.csv")
                    print("Time for " + str(feature_segment) + " seconds and chunknumber: " + str(
                        chunksize_counter) + ":" + str(
                        (time.time() - time.start) / 60) + " with saving")
                    # print shape of df_features
                    print(df_features.shape)
                    # increase chunksize_counter
                    chunksize_counter += 1

            elif path_sensorfile.endswith(".pkl"):
                print("path_sensorfile is .pkl")
                df_sensor_complete = pd.read_pickle(path_sensorfile)
                print("Size of df_sensor_complete: ", df_sensor_complete.shape)
                chunksize_total = int(df_sensor_complete.shape[0] / 100000)

                # iterate through df_sensor_complete in steps of 100000
                for i in tqdm(range(0, df_sensor_complete.shape[0], 100000)):
                    # check if chunk was already computed
                    dir_storage = path_storage + sensor + "/"
                    # create directory if it does not exist
                    if not os.path.exists(dir_storage):
                        os.makedirs(dir_storage)
                    path_features = dir_storage + sensor + "_timeperiod-" + str(
                        feature_segment) + "_only-active-smartphone-sessions-" + only_active_smartphone_sessions + " s_chunknumber-" + str(
                        chunksize_counter) + ".pkl"

                    if os.path.exists(path_features):
                        print("Jump over chunksize ", chunksize_counter, " in time_period ", feature_segment,
                              " and sensor ", sensor, " , was already computed, continuing with next chunk")
                        chunksize_counter += 1
                        continue

                    df_sensor = df_sensor_complete[i:i + 100000]
                    print("The shape of this chunk is ", df_sensor.shape)
                    print("The chunksize_counter is ", chunksize_counter)
                    print("i is ", i)

                    # print the current time
                    print(f"date: {datetime.datetime.now()}")
                    time.start = time.time()
                    df_features = computeFeatures.feature_extraction(df_sensor, sensor_column_names, feature_segment,
                                                                     sensor_frequency,
                                                                     time_column_name="timestamp",
                                                                     ESM_event_column_name="ESM_timestamp")
                    # check if df_features is an empty DataFrame; if so, continues with next chunksize
                    if df_features.empty:
                        print("df_features is empty for chunksize ", chunksize_counter, " in time_period ", seconds,
                              ", was removed, continuing with next chunk")
                        chunksize_counter += 1
                        continue

                    print(f"date: {datetime.datetime.now()}")
                    print(
                        "Time for " + str(feature_segment) + " seconds: " + str(
                            (time.time() - time.start) / 60) + " - without saving")

                    # save features with pickle
                    with open(path_features, 'wb') as f:
                        pickle.dump(df_features, f, pickle.HIGHEST_PROTOCOL)

                    # df_features.to_csv(dir_sensorfiles + "features/" + str(seconds) + " seconds_high_frequency_sensors.csv")
                    print("Time for " + str(feature_segment) + " seconds and chunknumber: " + str(
                        chunksize_counter) + "/" + str(chunksize_total) + ":" + str(
                        (time.time() - time.start) / 60) + " with saving")

                    # print shape of df_features
                    print("Size of the features: ", df_features.shape)
                    # increase chunksize_counter
                    chunksize_counter += 1

#testarea
df_test = pd.read_pickle("/Users/benediktjordan/Documents/WellspentWork/SmartphoneActivityIdentification/data_analysis/data_preparation/features/['accelerometer', 'rotation', 'gravity']_sensor-frequency-5Hz_timeperiod-60 s_chunknumber-1.pkl")
df_test["device_id"].unique()
df_test["activity"].unique()


## combining the chunks
## Note: as this repetitively crahed: can set for which chunknumber the concatenated file is saved in for
## loop below; if crash appears: just delete all other chunk files before that one and start with
## that one again (have to manually rename the intermediate file)
frequencies = [50, 5]
feature_segments = [1, 2, 5, 10, 15, 30, 60]  # in seconds
frequencies = [5]
feature_segments = [1]  # in seconds

path_storage = "/Users/benediktjordan/Documents/WellspentWork/SmartphoneActivityIdentification/data_analysis/data_preparation/features/"
path_to_store = "/Users/benediktjordan/Documents/WellspentWork/SmartphoneActivityIdentification/data_analysis/data_preparation/features/"
chunksize_counter = 8
sensor_sets_to_extract = [["accelerometer", "rotation", "gravity"]]

for sensor_frequency in frequencies:
    for feature_segment in feature_segments:
        for sensor_set in sensor_sets_to_extract:
            # create empty dataframe
            df_features = pd.DataFrame()
            # iterate through all chunks
            for chunknumber in range(1, chunksize_counter):

                path_features = path_to_store + str(sensor_set) + "_sensor-frequency-" + str(
                    sensor_frequency) + "Hz_timeperiod-" + str(
                    feature_segment) + " s_chunknumber-" + str(
                    chunknumber) + ".pkl"

                # try to load chunk; if doesn´t exist - continue
                try:
                    with open(path_features, 'rb') as f:
                        df_features_chunk = pickle.load(f)
                except:
                    print("chunknumber ", chunknumber, " in time_period ", feature_segment, " does not exist, continuing with next chunk")
                    continue

                # load chunk which exists
                with open(path_features, 'rb') as f:
                    df_features_chunk = pickle.load(f)

                # print size of chunk and df_features
                print("chunknumber ", chunknumber, " in time_period ", feature_segment, " has size ", df_features_chunk.shape)

                # concatenate chunk to df_features
                df_features = pd.concat([df_features, df_features_chunk], axis=0)
                print("chunknumber ", chunknumber, " in time_period ", feature_segment, " loaded and concatenated")
                print("df_features has size ", df_features.shape)


            # reset index
            df_features.reset_index(drop=True, inplace=True)

            # save df_features
            path_features = path_to_store + str(sensor_set) + "_sensor-frequency-" + str(
                sensor_frequency) + "Hz_timeperiod-" + str(
                feature_segment) + "_FeaturesExtracted.pkl"
            with open(path_features, 'wb') as f:
                pickle.dump(df_features, f, pickle.HIGHEST_PROTOCOL)

#endregion

#region feature selection with tsfresh algorithm
path_storage = "/Users/benediktjordan/Documents/WellspentWork/SmartphoneActivityIdentification/data_analysis/data_preparation/features/"
path_to_store = "/Users/benediktjordan/Documents/WellspentWork/SmartphoneActivityIdentification/data_analysis/data_preparation/features/"
label_column_name = "activity"
feature_segments = [60,30,15, 10, 5,2,1] # second 1 is excluded as my system always crushes
sensor_set = ["accelerometer", "rotation", "gravity"]
frequencies = [5, 50]
frequencies = [5]
feature_segments = [5]
apply_tsfresh_feature_selection = "yes"
time_column_name = "datetime"
drop_columns = []


for sensor_frequency in frequencies:
    for feature_segment in feature_segments:
        print("feature segment started: ", feature_segment)

        # load df_features
        #path_features = dir_sensorfiles + "data_preparation/features/highfrequencysensors-" + str(sensors_included) + "_timeperiod-" + str(seconds) + " s.pkl"
        path_features = path_storage + str(sensor_set) + "_sensor-frequency-" + str(
                    sensor_frequency) + "Hz_timeperiod-" + str(
                    feature_segment) + "_FeaturesExtracted.pkl"
        df_features = pd.read_pickle(path_features)

        #drop columns
        #    drop_columns = ["GPS_timestamp_merged", "Unnamed: 0.21", "Unnamed: 0.20", "Unnamed: 0.19", "Unnamed: 0.18", "Unnamed: 0.17", "Unnamed: 0.16", "Unnamed: 0.15", "Unnamed: 0.14", "Unnamed: 0.13", "Unnamed: 0.12", "Unnamed: 0.11", "Unnamed: 0.10", "Unnamed: 0.9", "Unnamed: 0.8", "Unnamed: 0.7", "Unnamed: 0.6", "Unnamed: 0.5", "Unnamed: 0.4", "Unnamed: 0.3", "Unnamed: 0.2", "Unnamed: 0.1", ]
        df_features.drop(drop_columns, axis=1, inplace=True)

        #temporary: only select part of rows
        #df_features = df_features.iloc[0:10000, :].copy()
        #print("df_features loaded")
        #temporary set first row device_id == 1
        #df_features.at[0, "device_id"] = 1

        #temporary: drop column "timestamp_beginning_of_feature_segment"
        #df_features.drop(columns=["timestamp_beginning_of_feature_segment"], inplace=True)

        #temporary: drop column "timestamp_beginning_of_feature_segment"

        features_filtered, df_analytics = create_features.feature_selection(df_features, label_column_name, apply_tsfresh_feature_selection)

        # save df_features
        path_features = path_storage + str(sensor_set) + "_sensor-frequency-" + str(
            sensor_frequency) + "Hz_timeperiod-" + str(
            feature_segment) + "_FeaturesExtracted_Selected.pkl"

        with open(path_features, 'wb') as f:
            pickle.dump(features_filtered, f, pickle.HIGHEST_PROTOCOL)
        print("df_features saved")


#temporary: convert pickle to csv (since in other environments pickle is not working)
feature_segments = [10, 5, 2, 1]
for seconds in feature_segments:
    print("seconds started: ", seconds)
    path_features = "/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_preparation/features/data_driven_approach/label_human motion_['linear_accelerometer', 'gyroscope', 'magnetometer', 'rotation', 'accelerometer']-merged-to-GPSFeatures_feature-segment-" + str(seconds) + "_min-gps-accuracy-35_only-active-smartphone-sessions-yes_FeaturesExtracted_Merged_Selected.pkl"
    df_features = pd.read_pickle(path_features)
    df_features.to_csv("/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_preparation/features/data_driven_approach/label_human motion_['linear_accelerometer', 'gyroscope', 'magnetometer', 'rotation', 'accelerometer']-merged-to-GPSFeatures_feature-segment-" + str(seconds) + "_min-gps-accuracy-35_only-active-smartphone-sessions-yes_FeaturesExtracted_Merged_Selected.csv")

#endregion



#endregion

#region modeling
##region Decision Forest for comparing different feature segments and different frequencies
#region training DF
feature_segments = [60, 30, 15, 10, 5,2,1] #in seconds; define length of segments of parameters (over which duration they have been created)

combination_sensors = ["accelerometer", "rotation", "gravity"]
frequencies = [5, 50]
label_column_name = "activity"

path_storage = "/Users/benediktjordan/Documents/WellspentWork/SmartphoneActivityIdentification/data_analysis/data_preparation/features/"
path_to_store = "/Users/benediktjordan/Documents/WellspentWork/SmartphoneActivityIdentification/data_analysis/modeling/FeatureSegmentComparison/"

n_permutations = 0 # define number of permutations; better 1000
# if label classes should be joint -> define in label mapping
label_mapping = {
    'Sitting_Read': 'Sitting_Scroll',
    "Walking_Read": "Walking_Scroll"
}
#label_mapping = None

label_classes = ['Multitasking_Scroll', 'Multitasking_Type', 'Multitasking_Watch',
       'Multitasking_Read', 'Sitting_Idle', 'Sitting_Type',
       'Sitting_Watch', 'Sitting_Scroll', 'Sitting_Read', 'Walking_Idle',
       'Walking_Read', 'Walking_Type', 'Walking_Watch', 'Walking_Scroll' ] # which label classes should be considered
label_classes = ['Sitting_Idle', 'Sitting_Type',
                 'Sitting_Watch', 'Sitting_Scroll', 'Walking_Idle',
                 'Walking_Type', 'Walking_Watch', 'Walking_Scroll']

parameter_tuning = "no" # if True: parameter tuning is performed; if False: default parameters are used
drop_cols = [] # columns that should be dropped
feature_importance = "shap"

parameter_set = {
    "n_estimators": 100, #default
    "criterion": "gini", #default
    "max_depth": None, #default
    "min_samples_split": 2,#default
    "min_samples_leaf": 1,#default
    "min_weight_fraction_leaf": 0.,#default
    "max_features": None, # THIS IS CHANGED
    "max_leaf_nodes": None,#default
    "min_impurity_decrease": 0.0,#default
    "bootstrap": True,#default
    "oob_score": False,#default
    "n_jobs": None,#default
    "verbose": 0,#default
    "warm_start": False,#default
    "class_weight": "balanced", # THIS IS CHANGED
    "n_jobs": -1,#default
    "random_state": 11#default
}

# select only data which are in the label_segment around ESM_event & drop columns
df_decisionforest_results_all = pd.DataFrame(columns=["Label", "Seconds around Event", "Balanced Accuracy", "Accuracy", "F1",
                                             "Precision", "Recall", "Sensor Combination", "Feature Segments"])
df_analytics = pd.DataFrame(columns=["Sensor Combination", "Feature Segment", "Event Description", "Number Participants", "Number Samples", "Number Features"])
for frequency in frequencies:
    for feature_segment in feature_segments:

        # create path to data based on sensor combination and feature segment
        path_dataset = path_storage + str(combination_sensors) + "_sensor-frequency-" + str(frequency) + "Hz_timeperiod-" +\
                       str(feature_segment) + "_FeaturesExtracted_Selected.pkl"

        t0 = time.time()
        print("start of frequency: " + str(frequency) + " and parameter_segment: " + str(feature_segment))
        path_to_store_current = path_to_store + str(frequency) + "Hz_" + str(combination_sensors) + "_feature-segment-" + str(feature_segment) + "/"
        if not os.path.exists(path_to_store_current):
            os.makedirs(path_to_store_current)
        if path_dataset.endswith(".csv"):
            df = pd.read_csv(path_dataset)
        elif path_dataset.endswith(".pkl"):
            with open(path_dataset, "rb") as f:
                df = pickle.load(f)

        #update analytics
        df_analytics = df_analytics.append({"Sensor Combination": str(combination_sensors),
                                            "Feature Segment": str(feature_segment),
                                            "Event Description": "Dataset loaded",
                                            "Number Participants": len(df["device_id"].unique()),
                                            "Number Samples": len(df),
                                            "Number Features": len(df.columns) - 1}, ignore_index=True)

        df = df.drop(columns=drop_cols)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        #drop NaN in device_id:
        #TODO: probably error that there are still NaN in device_id -> probably from Merging step -> resolve if time
        df = df.dropna(subset=["device_id"])
        #update analytics
        df_analytics = df_analytics.append({"Sensor Combination": str(combination_sensors),
                                            "Feature Segment": str(feature_segment),
                                            "Event Description": "NaN in device_id dropped",
                                            "Number Participants": len(df["device_id"].unique()),
                                            "Number Samples": len(df),
                                            "Number Features": len(df.columns) - 1}, ignore_index=True)

        #drop NaN in label_column_name
        df = df.dropna(subset=[label_column_name])
        # update analytics
        df_analytics = df_analytics.append({"Sensor Combination": str(combination_sensors),
                                            "Feature Segment": str(feature_segment),
                                            "Event Description": "NaN in label_column_name dropped",
                                            "Number Participants": len(df["device_id"].unique()),
                                            "Number Samples": len(df),
                                            "Number Features": len(df.columns) - 1}, ignore_index=True)

        # drop rows with NaN: will be many, as after merging of high-freq with GPS, NaN have not be dropped yet
        df = df.dropna()
        # update analytics
        df_analytics = df_analytics.append({"Sensor Combination": str(combination_sensors),
                                            "Feature Segment": str(feature_segment),
                                            "Event Description": "NaN in any column dropped",
                                            "Number Participants": len(df["device_id"].unique()),
                                            "Number Samples": len(df),
                                            "Number Features": len(df.columns) - 1}, ignore_index=True)

        #combine label classes if necessary
        if label_mapping != None:
            df = df.reset_index(drop=True)
            for mapping in label_mapping:
                df.loc[df[label_column_name] == mapping[0], label_column_name] = mapping[1]
        df = df[df[label_column_name].isin(label_classes)]  # drop rows which don´t contain labels which are in label_classes

        # update analytics
        df_analytics = df_analytics.append({"Sensor Combination": str(combination_sensors),
                                            "Feature Segment": str(feature_segment),
                                            "Event Description": "Label Classes which are not included are dropped",
                                            "Number Participants": len(df["device_id"].unique()),
                                            "Number Samples": len(df),
                                            "Number Features": len(df.columns) - 1}, ignore_index=True)

        df_decisionforest = df.reset_index(drop=True)

        print("Size of dataset is " + str(df_decisionforest.shape))
        # create dataframe which counts values of label_column_name grouped by "device_id" and save it to csv
        df_label_counts = df_decisionforest.groupby("device_id")[label_column_name].value_counts().unstack().fillna(0)
        df_label_counts["total"] = df_label_counts.sum(axis=1)
        df_label_counts.loc["total"] = df_label_counts.sum(axis=0)
        df_label_counts.to_csv(path_to_store_current + label_column_name +  "_label_counts.csv")

        # check if df_decisionforest is empty
        if df_decisionforest.empty:
            print("df_decisionforest is empty")
            continue

        #run DF
        df_decisionforest_results, df_labels_predictions = DecisionForest.DF_sklearn(df_decisionforest, label_column_name, n_permutations, path_to_store_current,
                                                              feature_importance = feature_importance, confusion_matrix_order = label_classes,
                                                              title_confusion_matrix = "Confusion Matrix", title_feature_importance_grid =
                                                              "Feature Importance Values", parameter_tuning = parameter_tuning,
                                                              parameter_set = parameter_set)
        df_decisionforest_results["Sensor Combination"] = str(combination_sensors)
        df_decisionforest_results["Feature Segments"] = str(feature_segment)
        df_decisionforest_results["Sampling Rate"] = str(frequency)

        df_decisionforest_results_all = pd.concat([df_decisionforest_results_all, df_decisionforest_results], axis=0)
        df_decisionforest_results_all.to_csv(path_to_store_current + label_column_name + "s_parameter_tuning-" + parameter_tuning + "_results.csv")
        df_labels_predictions.to_csv(path_to_store_current + label_column_name + "s_parameter_tuning-" + parameter_tuning + "_labels_predictions.csv")
        print("Finished Combination: " + str(combination_sensors) + " in " + str((time.time() - t0)/60) + " minutes")

df_analytics.to_csv(path_to_store + "parameter_tuning-" + parameter_tuning + "_analytics.csv")
df_decisionforest_results_all.to_csv(path_to_store + "parameter_tuning-" + parameter_tuning + "_results_overall.csv")

#endregion

#region visualize results
datasets = ["Data-Driven", "Theory-Driven"]
label_segments = [45, 90]
parameter_tuning = "no"
for dataset in datasets:

    fig, ax = plt.subplots(figsize=(10, 5))
    for label_segment in label_segments:
        if dataset == "Data-Driven":
            # load csv with "," as separator
            df_results = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_analysis/decision_forest/timeperiod_around_event-" + str(label_segment) + "-GPS_parameter_tuning-" + parameter_tuning + "_results_data-driven-dataset.csv", sep = ";")
        elif dataset == "Theory-Driven":
            df_results = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_analysis/decision_forest/timeperiod_around_event-" + str(label_segment) + "-GPS_parameter_tuning-" + parameter_tuning + "_results_theory-driven-dataset.csv", sep = ";")

        # reorder the rows: first hypothesis-driven, then data-driven; and with those, ascending feature segments
        df_results = df_results.sort_values(by=["Feature Segments"])
        label = str(label_segment) + "s Label Segment"
        if label_segment == 90:
            label = "95s Label Segment"
        # visualize Balanced Accuracy and F1 Score for "Feature Set and Feature Segment" as two lines in a SNS lineplot
        #sns.set_style("whitegrid")
        #sns.set_context("paper", font_scale=1.5)
        #sns.set_palette("colorblind")
        #create sns lineplot
        sns.lineplot(x="Feature Segments", y="Balanced Accuracy", data=df_results, label=label, ax=ax)
    plt.title("Model Performance for " + dataset + " Dataset")
    plt.xlabel("Feature Segment")
    plt.ylabel("Balanced Accuracy")
    plt.legend()
    #plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    #set y-axis limits
    plt.ylim(0, 1)
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    #plt.show()
    #save with bbox_inches="tight" to avoid cutting off x-axis labels
    fig.savefig("/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_analysis/decision_forest/timeperiod_around_event-" + str(label_segments) + "-GPS_parameter_tuning-" + parameter_tuning + "_" + str(dataset) + "_results_overall_visualization.png",
                bbox_inches="tight", dpi= 600)

#region OUTDATED VISUALIZE IN ONE PLOT: visualizing performances for 2 label segments x 11 feature segment-dataset combinations for balanced accuracy
label_segments = [45, 90]
parameter_tuning = "no"
fig, ax = plt.subplots(figsize=(10, 5))
for label_segment in label_segments:
    df_results = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_analysis/decision_forest/timeperiod_around_event-" + str(label_segment) + "-GPS_parameter_tuning-" + parameter_tuning + "_results_overall.csv")
    # add dataset description depending on value in "Sensor Combination" and "Feature Segments"
    df_results["Feature Set and Feature Segment"] = ""
    for index, row in df_results.iterrows():
        if row["Sensor Combination"] == "['linear_accelerometer', 'gyroscope', 'magnetometer', 'rotation', 'accelerometer']":
            if row["Feature Segments"] == 60:
                df_results.loc[index, "Feature Set and Feature Segment"] = "Data-Driven (60s)"
            elif row["Feature Segments"] == 30:
                df_results.loc[index, "Feature Set and Feature Segment"] = "Data-Driven (30s)"
            elif row["Feature Segments"] == 10:
                df_results.loc[index, "Feature Set and Feature Segment"] = "Data-Driven (10s)"
            elif row["Feature Segments"] == 5:
                df_results.loc[index, "Feature Set and Feature Segment"] = "Data-Driven (5s)"
            elif row["Feature Segments"] == 2:
                df_results.loc[index, "Feature Set and Feature Segment"] = "Data-Driven (2s)"
            elif row["Feature Segments"] == 1:
                df_results.loc[index, "Feature Set and Feature Segment"] = "Data-Driven (1s)"
        elif row["Sensor Combination"] == "['linear_accelerometer', 'rotation']":
            if row["Feature Segments"] == 60:
                df_results.loc[index, "Feature Set and Feature Segment"] = "Hypothesis-Driven (60s)"
            elif row["Feature Segments"] == 30:
                df_results.loc[index, "Feature Set and Feature Segment"] = "Hypothesis-Driven (30s)"
            elif row["Feature Segments"] == 10:
                df_results.loc[index, "Feature Set and Feature Segment"] = "Hypothesis-Driven (10s)"
            elif row["Feature Segments"] == 5:
                df_results.loc[index, "Feature Set and Feature Segment"] = "Hypothesis-Driven (5s)"
            elif row["Feature Segments"] == 2:
                df_results.loc[index, "Feature Set and Feature Segment"] = "Hypothesis-Driven (2s)"
            elif row["Feature Segments"] == 1:
                df_results.loc[index, "Feature Set and Feature Segment"] = "Hypothesis-Driven (1s)"
    # reorder the rows: first hypothesis-driven, then data-driven; and with those, ascending feature segments
    df_results = df_results.sort_values(by=["Feature Segments"])
    df_results = df_results.sort_values(by=["Sensor Combination"], ascending=False)
    label = str(label_segment) + "s Label Segment"
    if label_segment == 90:
        label = "95s Label Segment"
    # visualize Balanced Accuracy and F1 Score for "Feature Set and Feature Segment" as two lines in a SNS lineplot
    #sns.set_style("whitegrid")
    #sns.set_context("paper", font_scale=1.5)
    #sns.set_palette("colorblind")
    #create sns lineplot
    sns.lineplot(x="Feature Set and Feature Segment", y="Balanced Accuracy", data=df_results, label=label, ax=ax)
plt.title("Model Performance of Different Datasets, Feature Segments, and Label Segments")
plt.xlabel("Dataset and Feature-Segment")
plt.ylabel("Balanced Accuracy")
plt.legend()
plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
#set y-axis limits
plt.ylim(0, 1)
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
#plt.show()
#save with bbox_inches="tight" to avoid cutting off x-axis labels
fig.savefig("/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_analysis/decision_forest/timeperiod_around_event-" + str(label_segments) + "-GPS_parameter_tuning-" + parameter_tuning + "_results_overall_visualization.png",
            bbox_inches="tight", dpi= 600)
#endregion



# visualizing performances of the different DF (balanced accuracy & F1 score)
label_segment = 90
parameter_tuning = "no"
df_results = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_analysis/decision_forest/timeperiod_around_event-" + str(label_segment) + "-GPS_parameter_tuning-" + parameter_tuning + "_results_overall.csv")
# add dataset description depending on value in "Sensor Combination" and "Feature Segments"
df_results["Feature Set and Feature Segment"] = ""
for index, row in df_results.iterrows():
    if row["Sensor Combination"] == "['linear_accelerometer', 'gyroscope', 'magnetometer', 'rotation', 'accelerometer']":
        if row["Feature Segments"] == 60:
            df_results.loc[index, "Feature Set and Feature Segment"] = "Data-Driven (60s)"
        elif row["Feature Segments"] == 30:
            df_results.loc[index, "Feature Set and Feature Segment"] = "Data-Driven (30s)"
        elif row["Feature Segments"] == 10:
            df_results.loc[index, "Feature Set and Feature Segment"] = "Data-Driven (10s)"
        elif row["Feature Segments"] == 5:
            df_results.loc[index, "Feature Set and Feature Segment"] = "Data-Driven (5s)"
        elif row["Feature Segments"] == 2:
            df_results.loc[index, "Feature Set and Feature Segment"] = "Data-Driven (2s)"
        elif row["Feature Segments"] == 1:
            df_results.loc[index, "Feature Set and Feature Segment"] = "Data-Driven (1s)"
    elif row["Sensor Combination"] == "['linear_accelerometer', 'rotation']":
        if row["Feature Segments"] == 60:
            df_results.loc[index, "Feature Set and Feature Segment"] = "Hypothesis-Driven (60s)"
        elif row["Feature Segments"] == 30:
            df_results.loc[index, "Feature Set and Feature Segment"] = "Hypothesis-Driven (30s)"
        elif row["Feature Segments"] == 10:
            df_results.loc[index, "Feature Set and Feature Segment"] = "Hypothesis-Driven (10s)"
        elif row["Feature Segments"] == 5:
            df_results.loc[index, "Feature Set and Feature Segment"] = "Hypothesis-Driven (5s)"
        elif row["Feature Segments"] == 2:
            df_results.loc[index, "Feature Set and Feature Segment"] = "Hypothesis-Driven (2s)"
        elif row["Feature Segments"] == 1:
            df_results.loc[index, "Feature Set and Feature Segment"] = "Hypothesis-Driven (1s)"
# visualize Balanced Accuracy and F1 Score for "Feature Set and Feature Segment" as two lines in a SNS lineplot
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)
sns.set_palette("colorblind")
fig, ax = plt.subplots(figsize=(10, 5))
#visualize Balanced Accuracy and F1 Score for "Feature Set and Feature Segment" as two lines in a SNS lineplot; use pandas.melt
sns.lineplot(x="Feature Set and Feature Segment", y="value", hue="variable", data=pd.melt(df_results, id_vars=["Feature Set and Feature Segment"], value_vars=["Balanced Accuracy", "F1"]))
plt.title("Model Performance of Different Datasets and Feature Segments")
plt.xlabel("Dataset and Feature-Segment")
plt.ylabel("Score")
plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
#set y-axis limits
plt.ylim(0, 1)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
#plt.show()
#save with bbox_inches="tight" to avoid cutting off x-axis labels
plt.savefig("/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_analysis/decision_forest/timeperiod_around_event-" + str(label_segment) + "-GPS_parameter_tuning-" + parameter_tuning + "_results_overall_visualization.png",
            bbox_inches="tight", dpi= 300)
#endregion
#endregion

##region hyperparameter tune the best model from previous comparison step
#region training DF
feature_segments = [10] #in seconds; define length of segments of parameters (over which duration they have been created)
combination_sensors = ["accelerometer", "rotation", "gravity"]
frequencies = [5]
label_column_name = "activity"

path_storage = "/Users/benediktjordan/Documents/WellspentWork/SmartphoneActivityIdentification/data_analysis/data_preparation/features/"
path_to_store = "/Users/benediktjordan/Documents/WellspentWork/SmartphoneActivityIdentification/data_analysis/modeling/HyperparameterTuning/"
title_confusion_matrix = "Confusion Matrix"
title_feature_importance_grid = "SHAP Feature Importance"

n_permutations = 0 # define number of permutations; better 1000
# if label classes should be joint -> define in label mapping
label_mapping = {
    'Sitting_Read': 'Sitting_Scroll',
    "Walking_Read": "Walking_Scroll"
}
#label_mapping = None

label_classes = ['Multitasking_Scroll', 'Multitasking_Type', 'Multitasking_Watch',
       'Multitasking_Read', 'Sitting_Idle', 'Sitting_Type',
       'Sitting_Watch', 'Sitting_Scroll', 'Sitting_Read', 'Walking_Idle',
       'Walking_Read', 'Walking_Type', 'Walking_Watch', 'Walking_Scroll' ] # which label classes should be considered
label_classes = ['Sitting_Idle', 'Sitting_Type',
                 'Sitting_Watch', 'Sitting_Scroll', 'Walking_Idle',
                 'Walking_Type', 'Walking_Watch', 'Walking_Scroll']

parameter_tuning = "yes" # if True: parameter tuning is performed; if False: default parameters are used
drop_cols = [] # columns that should be dropped
feature_importance = "shap"

parameter_set = {
    "n_estimators": 100, #default
    "criterion": "gini", #default
    "max_depth": None, #default
    "min_samples_split": 2,#default
    "min_samples_leaf": 1,#default
    "min_weight_fraction_leaf": 0.,#default
    "max_features": None, # THIS IS CHANGED
    "max_leaf_nodes": None,#default
    "min_impurity_decrease": 0.0,#default
    "bootstrap": True,#default
    "oob_score": False,#default
    "n_jobs": None,#default
    "verbose": 0,#default
    "warm_start": False,#default
    "class_weight": "balanced", # THIS IS CHANGED
    "n_jobs": -1,#default
    "random_state": 11#default
}


# smaller parameter tuning set in order to have less runtime
n_estimators = [100, 500]  # default 100
max_depth = [15, None] # default None
min_samples_split = [2, 30] # default 2
min_samples_leaf = [1, 3] # default 1
max_features = ["sqrt", None] # default "sqrt"
oob_score = [False, True] # default False;
class_weight = ["balanced", "balanced_subsample"] # default None
criterion =[ "gini", "entropy"] # default "gini"
max_samples = [None, 0.8] # default None which means that 100% of the samples are used for each tree
bootstrap = [True]

grid_search_space = dict(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
             min_samples_leaf=min_samples_leaf, max_features=max_features, oob_score = oob_score, class_weight = class_weight,
                            criterion = criterion, max_samples = max_samples, bootstrap = bootstrap)


# select only data which are in the label_segment around ESM_event & drop columns
df_decisionforest_results_all = pd.DataFrame(columns=["Label", "Seconds around Event", "Balanced Accuracy", "Accuracy", "F1",
                                             "Precision", "Recall", "Sensor Combination", "Feature Segments"])
df_analytics = pd.DataFrame(columns=["Sensor Combination", "Feature Segment", "Event Description", "Number Participants", "Number Samples", "Number Features"])
for frequency in frequencies:
    for feature_segment in feature_segments:

        # create path to data based on sensor combination and feature segment
        path_dataset = path_storage + str(combination_sensors) + "_sensor-frequency-" + str(frequency) + "Hz_timeperiod-" +\
                       str(feature_segment) + "_FeaturesExtracted_Selected.pkl"

        t0 = time.time()
        print("start of frequency: " + str(frequency) + " and parameter_segment: " + str(feature_segment))
        path_to_store_current = path_to_store + str(frequency) + "Hz_" + str(combination_sensors) + "_feature-segment-" + str(feature_segment) + "/"
        if not os.path.exists(path_to_store_current):
            os.makedirs(path_to_store_current)
        if path_dataset.endswith(".csv"):
            df = pd.read_csv(path_dataset)
        elif path_dataset.endswith(".pkl"):
            with open(path_dataset, "rb") as f:
                df = pickle.load(f)

        #update analytics
        df_analytics = df_analytics.append({"Sensor Combination": str(combination_sensors),
                                            "Feature Segment": str(feature_segment),
                                            "Event Description": "Dataset loaded",
                                            "Number Participants": len(df["device_id"].unique()),
                                            "Number Samples": len(df),
                                            "Number Features": len(df.columns) - 1}, ignore_index=True)

        df = df.drop(columns=drop_cols)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        #drop NaN in device_id:
        #TODO: probably error that there are still NaN in device_id -> probably from Merging step -> resolve if time
        df = df.dropna(subset=["device_id"])
        #update analytics
        df_analytics = df_analytics.append({"Sensor Combination": str(combination_sensors),
                                            "Feature Segment": str(feature_segment),
                                            "Event Description": "NaN in device_id dropped",
                                            "Number Participants": len(df["device_id"].unique()),
                                            "Number Samples": len(df),
                                            "Number Features": len(df.columns) - 1}, ignore_index=True)

        #drop NaN in label_column_name
        df = df.dropna(subset=[label_column_name])
        # update analytics
        df_analytics = df_analytics.append({"Sensor Combination": str(combination_sensors),
                                            "Feature Segment": str(feature_segment),
                                            "Event Description": "NaN in label_column_name dropped",
                                            "Number Participants": len(df["device_id"].unique()),
                                            "Number Samples": len(df),
                                            "Number Features": len(df.columns) - 1}, ignore_index=True)

        # drop rows with NaN: will be many, as after merging of high-freq with GPS, NaN have not be dropped yet
        df = df.dropna()
        # update analytics
        df_analytics = df_analytics.append({"Sensor Combination": str(combination_sensors),
                                            "Feature Segment": str(feature_segment),
                                            "Event Description": "NaN in any column dropped",
                                            "Number Participants": len(df["device_id"].unique()),
                                            "Number Samples": len(df),
                                            "Number Features": len(df.columns) - 1}, ignore_index=True)

        #combine label classes if necessary
        if label_mapping != None:
            df = df.reset_index(drop=True)
            for mapping in label_mapping:
                df.loc[df[label_column_name] == mapping[0], label_column_name] = mapping[1]
        df = df[df[label_column_name].isin(label_classes)]  # drop rows which don´t contain labels which are in label_classes

        # update analytics
        df_analytics = df_analytics.append({"Sensor Combination": str(combination_sensors),
                                            "Feature Segment": str(feature_segment),
                                            "Event Description": "Label Classes which are not included are dropped",
                                            "Number Participants": len(df["device_id"].unique()),
                                            "Number Samples": len(df),
                                            "Number Features": len(df.columns) - 1}, ignore_index=True)

        df_decisionforest = df.reset_index(drop=True)

        print("Size of dataset is " + str(df_decisionforest.shape))
        # create dataframe which counts values of label_column_name grouped by "device_id" and save it to csv
        df_label_counts = df_decisionforest.groupby("device_id")[label_column_name].value_counts().unstack().fillna(0)
        df_label_counts["total"] = df_label_counts.sum(axis=1)
        df_label_counts.loc["total"] = df_label_counts.sum(axis=0)
        df_label_counts.to_csv(path_to_store_current + label_column_name +  "_label_counts.csv")

        # check if df_decisionforest is empty
        if df_decisionforest.empty:
            print("df_decisionforest is empty")
            continue

        #run DF
        df_decisionforest_results, df_labels_predictions = DecisionForest.DF_sklearn(df_decisionforest, label_column_name, n_permutations, path_to_store_current,
                                                              feature_importance = feature_importance, confusion_matrix_order = label_classes,
                                                              title_confusion_matrix = title_confusion_matrix, title_feature_importance_grid =
                                                              title_feature_importance_grid, parameter_tuning = parameter_tuning,
                                                              parameter_set = parameter_set, grid_search_space = grid_search_space)
        df_decisionforest_results["Sensor Combination"] = str(combination_sensors)
        df_decisionforest_results["Feature Segments"] = str(feature_segment)
        df_decisionforest_results["Sampling Rate"] = str(frequency)

        df_decisionforest_results_all = pd.concat([df_decisionforest_results_all, df_decisionforest_results], axis=0)
        df_decisionforest_results_all.to_csv(path_to_store_current + label_column_name + "s_parameter_tuning-" + parameter_tuning + "_results.csv")
        df_labels_predictions.to_csv(path_to_store_current + label_column_name + "s_parameter_tuning-" + parameter_tuning + "_labels_predictions.csv")
        print("Finished Combination: " + str(combination_sensors) + " in " + str((time.time() - t0)/60) + " minutes")

df_analytics.to_csv(path_to_store + "parameter_tuning-" + parameter_tuning + "_analytics.csv")
df_decisionforest_results_all.to_csv(path_to_store + "parameter_tuning-" + parameter_tuning + "_results_overall.csv")

#endregion

#region visualize results
datasets = ["Data-Driven", "Theory-Driven"]
label_segments = [45, 90]
parameter_tuning = "no"
for dataset in datasets:

    fig, ax = plt.subplots(figsize=(10, 5))
    for label_segment in label_segments:
        if dataset == "Data-Driven":
            # load csv with "," as separator
            df_results = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_analysis/decision_forest/timeperiod_around_event-" + str(label_segment) + "-GPS_parameter_tuning-" + parameter_tuning + "_results_data-driven-dataset.csv", sep = ";")
        elif dataset == "Theory-Driven":
            df_results = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_analysis/decision_forest/timeperiod_around_event-" + str(label_segment) + "-GPS_parameter_tuning-" + parameter_tuning + "_results_theory-driven-dataset.csv", sep = ";")

        # reorder the rows: first hypothesis-driven, then data-driven; and with those, ascending feature segments
        df_results = df_results.sort_values(by=["Feature Segments"])
        label = str(label_segment) + "s Label Segment"
        if label_segment == 90:
            label = "95s Label Segment"
        # visualize Balanced Accuracy and F1 Score for "Feature Set and Feature Segment" as two lines in a SNS lineplot
        #sns.set_style("whitegrid")
        #sns.set_context("paper", font_scale=1.5)
        #sns.set_palette("colorblind")
        #create sns lineplot
        sns.lineplot(x="Feature Segments", y="Balanced Accuracy", data=df_results, label=label, ax=ax)
    plt.title("Model Performance for " + dataset + " Dataset")
    plt.xlabel("Feature Segment")
    plt.ylabel("Balanced Accuracy")
    plt.legend()
    #plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    #set y-axis limits
    plt.ylim(0, 1)
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    #plt.show()
    #save with bbox_inches="tight" to avoid cutting off x-axis labels
    fig.savefig("/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_analysis/decision_forest/timeperiod_around_event-" + str(label_segments) + "-GPS_parameter_tuning-" + parameter_tuning + "_" + str(dataset) + "_results_overall_visualization.png",
                bbox_inches="tight", dpi= 600)

#region OUTDATED VISUALIZE IN ONE PLOT: visualizing performances for 2 label segments x 11 feature segment-dataset combinations for balanced accuracy
label_segments = [45, 90]
parameter_tuning = "no"
fig, ax = plt.subplots(figsize=(10, 5))
for label_segment in label_segments:
    df_results = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_analysis/decision_forest/timeperiod_around_event-" + str(label_segment) + "-GPS_parameter_tuning-" + parameter_tuning + "_results_overall.csv")
    # add dataset description depending on value in "Sensor Combination" and "Feature Segments"
    df_results["Feature Set and Feature Segment"] = ""
    for index, row in df_results.iterrows():
        if row["Sensor Combination"] == "['linear_accelerometer', 'gyroscope', 'magnetometer', 'rotation', 'accelerometer']":
            if row["Feature Segments"] == 60:
                df_results.loc[index, "Feature Set and Feature Segment"] = "Data-Driven (60s)"
            elif row["Feature Segments"] == 30:
                df_results.loc[index, "Feature Set and Feature Segment"] = "Data-Driven (30s)"
            elif row["Feature Segments"] == 10:
                df_results.loc[index, "Feature Set and Feature Segment"] = "Data-Driven (10s)"
            elif row["Feature Segments"] == 5:
                df_results.loc[index, "Feature Set and Feature Segment"] = "Data-Driven (5s)"
            elif row["Feature Segments"] == 2:
                df_results.loc[index, "Feature Set and Feature Segment"] = "Data-Driven (2s)"
            elif row["Feature Segments"] == 1:
                df_results.loc[index, "Feature Set and Feature Segment"] = "Data-Driven (1s)"
        elif row["Sensor Combination"] == "['linear_accelerometer', 'rotation']":
            if row["Feature Segments"] == 60:
                df_results.loc[index, "Feature Set and Feature Segment"] = "Hypothesis-Driven (60s)"
            elif row["Feature Segments"] == 30:
                df_results.loc[index, "Feature Set and Feature Segment"] = "Hypothesis-Driven (30s)"
            elif row["Feature Segments"] == 10:
                df_results.loc[index, "Feature Set and Feature Segment"] = "Hypothesis-Driven (10s)"
            elif row["Feature Segments"] == 5:
                df_results.loc[index, "Feature Set and Feature Segment"] = "Hypothesis-Driven (5s)"
            elif row["Feature Segments"] == 2:
                df_results.loc[index, "Feature Set and Feature Segment"] = "Hypothesis-Driven (2s)"
            elif row["Feature Segments"] == 1:
                df_results.loc[index, "Feature Set and Feature Segment"] = "Hypothesis-Driven (1s)"
    # reorder the rows: first hypothesis-driven, then data-driven; and with those, ascending feature segments
    df_results = df_results.sort_values(by=["Feature Segments"])
    df_results = df_results.sort_values(by=["Sensor Combination"], ascending=False)
    label = str(label_segment) + "s Label Segment"
    if label_segment == 90:
        label = "95s Label Segment"
    # visualize Balanced Accuracy and F1 Score for "Feature Set and Feature Segment" as two lines in a SNS lineplot
    #sns.set_style("whitegrid")
    #sns.set_context("paper", font_scale=1.5)
    #sns.set_palette("colorblind")
    #create sns lineplot
    sns.lineplot(x="Feature Set and Feature Segment", y="Balanced Accuracy", data=df_results, label=label, ax=ax)
plt.title("Model Performance of Different Datasets, Feature Segments, and Label Segments")
plt.xlabel("Dataset and Feature-Segment")
plt.ylabel("Balanced Accuracy")
plt.legend()
plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
#set y-axis limits
plt.ylim(0, 1)
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
#plt.show()
#save with bbox_inches="tight" to avoid cutting off x-axis labels
fig.savefig("/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_analysis/decision_forest/timeperiod_around_event-" + str(label_segments) + "-GPS_parameter_tuning-" + parameter_tuning + "_results_overall_visualization.png",
            bbox_inches="tight", dpi= 600)
#endregion



# visualizing performances of the different DF (balanced accuracy & F1 score)
label_segment = 90
parameter_tuning = "no"
df_results = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_analysis/decision_forest/timeperiod_around_event-" + str(label_segment) + "-GPS_parameter_tuning-" + parameter_tuning + "_results_overall.csv")
# add dataset description depending on value in "Sensor Combination" and "Feature Segments"
df_results["Feature Set and Feature Segment"] = ""
for index, row in df_results.iterrows():
    if row["Sensor Combination"] == "['linear_accelerometer', 'gyroscope', 'magnetometer', 'rotation', 'accelerometer']":
        if row["Feature Segments"] == 60:
            df_results.loc[index, "Feature Set and Feature Segment"] = "Data-Driven (60s)"
        elif row["Feature Segments"] == 30:
            df_results.loc[index, "Feature Set and Feature Segment"] = "Data-Driven (30s)"
        elif row["Feature Segments"] == 10:
            df_results.loc[index, "Feature Set and Feature Segment"] = "Data-Driven (10s)"
        elif row["Feature Segments"] == 5:
            df_results.loc[index, "Feature Set and Feature Segment"] = "Data-Driven (5s)"
        elif row["Feature Segments"] == 2:
            df_results.loc[index, "Feature Set and Feature Segment"] = "Data-Driven (2s)"
        elif row["Feature Segments"] == 1:
            df_results.loc[index, "Feature Set and Feature Segment"] = "Data-Driven (1s)"
    elif row["Sensor Combination"] == "['linear_accelerometer', 'rotation']":
        if row["Feature Segments"] == 60:
            df_results.loc[index, "Feature Set and Feature Segment"] = "Hypothesis-Driven (60s)"
        elif row["Feature Segments"] == 30:
            df_results.loc[index, "Feature Set and Feature Segment"] = "Hypothesis-Driven (30s)"
        elif row["Feature Segments"] == 10:
            df_results.loc[index, "Feature Set and Feature Segment"] = "Hypothesis-Driven (10s)"
        elif row["Feature Segments"] == 5:
            df_results.loc[index, "Feature Set and Feature Segment"] = "Hypothesis-Driven (5s)"
        elif row["Feature Segments"] == 2:
            df_results.loc[index, "Feature Set and Feature Segment"] = "Hypothesis-Driven (2s)"
        elif row["Feature Segments"] == 1:
            df_results.loc[index, "Feature Set and Feature Segment"] = "Hypothesis-Driven (1s)"
# visualize Balanced Accuracy and F1 Score for "Feature Set and Feature Segment" as two lines in a SNS lineplot
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)
sns.set_palette("colorblind")
fig, ax = plt.subplots(figsize=(10, 5))
#visualize Balanced Accuracy and F1 Score for "Feature Set and Feature Segment" as two lines in a SNS lineplot; use pandas.melt
sns.lineplot(x="Feature Set and Feature Segment", y="value", hue="variable", data=pd.melt(df_results, id_vars=["Feature Set and Feature Segment"], value_vars=["Balanced Accuracy", "F1"]))
plt.title("Model Performance of Different Datasets and Feature Segments")
plt.xlabel("Dataset and Feature-Segment")
plt.ylabel("Score")
plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
#set y-axis limits
plt.ylim(0, 1)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
#plt.show()
#save with bbox_inches="tight" to avoid cutting off x-axis labels
plt.savefig("/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_analysis/decision_forest/timeperiod_around_event-" + str(label_segment) + "-GPS_parameter_tuning-" + parameter_tuning + "_results_overall_visualization.png",
            bbox_inches="tight", dpi= 300)
#endregion
#endregion

#region create deployment model
# Note: this trains a DF on whole dataset without any CV, parameter tuning, feature importance, or validation
# parameter initializations

feature_segment = 15 #in seconds; define length of segments of parameters (over which duration they have been created)
combination_sensors = ["accelerometer", "rotation", "gravity"]
frequency = 50
label_column_name = "activity"

path_storage = "/Users/benediktjordan/Documents/WellspentWork/SmartphoneActivityIdentification/data_analysis/data_preparation/features/"
path_to_store = "/Users/benediktjordan/Documents/WellspentWork/SmartphoneActivityIdentification/data_analysis/modeling/DeploymentModel/"
title_confusion_matrix = "Confusion Matrix"
title_feature_importance_grid = "SHAP Feature Importance"

n_permutations = 0 # define number of permutations; better 1000
# if label classes should be joint -> define in label mapping
label_mapping = {
    'Sitting_Read': 'Sitting_Scroll',
    "Walking_Read": "Walking_Scroll"
}
#label_mapping = None

label_classes = ['Multitasking_Scroll', 'Multitasking_Type', 'Multitasking_Watch',
       'Multitasking_Read', 'Sitting_Idle', 'Sitting_Type',
       'Sitting_Watch', 'Sitting_Scroll', 'Sitting_Read', 'Walking_Idle',
       'Walking_Read', 'Walking_Type', 'Walking_Watch', 'Walking_Scroll' ] # which label classes should be considered
label_classes = ['Sitting_Idle', 'Sitting_Type',
                 'Sitting_Watch', 'Sitting_Scroll', 'Walking_Idle',
                 'Walking_Type', 'Walking_Watch', 'Walking_Scroll']

parameter_tuning = "yes" # if True: parameter tuning is performed; if False: default parameters are used
drop_cols = [] # columns that should be dropped
feature_importance = "shap"

parameter_set = {
    "n_estimators": 100, #default
    "criterion": "gini", #default
    "max_depth": None, #default
    "min_samples_split": 2,#default
    "min_samples_leaf": 1,#default
    "min_weight_fraction_leaf": 0.,#default
    "max_features": None, # THIS IS CHANGED
    "max_leaf_nodes": None,#default
    "min_impurity_decrease": 0.0,#default
    "bootstrap": True,#default
    "oob_score": False,#default
    "n_jobs": None,#default
    "verbose": 0,#default
    "warm_start": False,#default
    "class_weight": "balanced", # THIS IS CHANGED
    "n_jobs": -1,#default
    "random_state": 11#default
}


# initial data transformations
if not os.path.exists(path_to_store):
    os.makedirs(path_to_store)
path_dataset = path_storage + str(combination_sensors) + "_sensor-frequency-" + str(frequency) + "Hz_timeperiod-" +\
                       str(feature_segment) + "_FeaturesExtracted_Selected.pkl"
df = pd.read_pickle(path_dataset)
df = df.drop(columns=drop_cols)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.dropna(subset=["device_id"])
df = df.dropna(subset=[label_column_name])
df = df.dropna()
if label_mapping != None:
    df = df.reset_index(drop=True)
    for mapping in label_mapping:
        df.loc[df[label_column_name] == mapping[0], label_column_name] = mapping[1]
df = df[df[label_column_name].isin(label_classes)]  # drop rows which don´t contain labels which are in label_classes

df_decisionforest = df.reset_index(drop=True)
print("Size of dataset is " + str(df_decisionforest.shape))
# create dataframe which counts values of label_column_name grouped by "device_id" and save it to csv
df_label_counts = df_decisionforest.groupby("device_id")[label_column_name].value_counts().unstack().fillna(0)
df_label_counts["total"] = df_label_counts.sum(axis=1)
df_label_counts.loc["total"] = df_label_counts.sum(axis=0)
df_label_counts.to_csv(
    path_to_store + label_column_name + "_label_counts.csv")

#run and save model
model = DecisionForest.DF_sklearn_deployment(df_decisionforest, label_column_name, parameter_set, path_storage)
pickle.dump(model, open(
    path_to_store + label_column_name + "_sensors-" + str(combination_sensors) + "_sampling-rate-" + str(frequency) + "_feature-segment-" + str(feature_segment) + "_FinalDeploymentModel.sav", "wb"))

#endregion

#region convert into CoreML model
path_storage = "/Users/benediktjordan/Documents/WellspentWork/SmartphoneActivityIdentification/data_analysis/data_preparation/features/"
path_to_store = "/Users/benediktjordan/Documents/WellspentWork/SmartphoneActivityIdentification/data_analysis/modeling/DeploymentModel/"
path_model = "/Users/benediktjordan/Documents/WellspentWork/SmartphoneActivityIdentification/data_analysis/modeling/DeploymentModel/activity_sensors-['accelerometer', 'rotation', 'gravity']_sampling-rate-50_feature-segment-15_FinalDeploymentModel.sav"
label_column_name = "activity"
frequency = 50
feature_segment = 15
combination_sensors = ["accelerometer", "rotation", "gravity"]
label_mapping = {
    'Sitting_Read': 'Sitting_Scroll',
    "Walking_Read": "Walking_Scroll"
}
label_classes = ['Sitting_Idle', 'Sitting_Type',
                 'Sitting_Watch', 'Sitting_Scroll', 'Walking_Idle',
                 'Walking_Type', 'Walking_Watch', 'Walking_Scroll']
path_dataset = path_storage + str(combination_sensors) + "_sensor-frequency-" + str(frequency) + "Hz_timeperiod-" +\
                       str(feature_segment) + "_FeaturesExtracted_Selected.pkl"
df = pd.read_pickle(path_dataset)

if label_mapping != None:
    df = df.reset_index(drop=True)
    for mapping in label_mapping:
        df.loc[df[label_column_name] == mapping[0], label_column_name] = mapping[1]
df = df[df[label_column_name].isin(label_classes)]  # drop rows which don´t contain labels which are in label_classes

# load model
model = pd.read_pickle(path_model)
features_used = list(model.feature_names_in_)

coreml_model = coremltools.converters.sklearn.convert(model)
coreml_model = sklearn.convert(model, features_used, label_column_name)


print(type(coreml_model))
print(coreml_model.input_description)
print(coreml_model.output_description)
model_spec = coreml_model.get_spec()


# check if the predictions of sklearn and coreml are the same
X_test_df = df.drop(columns=[label_column_name, "device_id", "timestamp"])  # add sensor_timestamp later on here
y_test_df = df[label_column_name]  # This is the outcome variable
y_test_df = y_test_df.reset_index(drop=True)

# Convert X_test_df to a list of dictionaries
X_test_dict_list = X_test_df.to_dict(orient='records')
coreml_preds = []
for sample in X_test_dict_list:
    # make predictions
    prediction = coreml_model.predict(sample)
    # add activity class to coreml_pred
    coreml_preds.append(prediction["activity"])
sklearn_preds = model.predict(X_test_df)

# check if both are the same
print("Are the predictions of sklearn and coreml the same? " + str(coreml_preds == sklearn_preds))

# Save the CoreML model to a file
coreml_model.save(path_to_store + label_column_name + "_sensors-" + str(combination_sensors) + "_sampling-rate-" + str(frequency) + "_feature-segment-" + str(feature_segment) + "_FinalDeploymentModel_CoreML.mlmodel")

#endregion

#region create toy CoreML model with limited parameters


#endregion

#region calculate computational complexity
import pyinstrument
from tsfresh import extract_features
from tsfresh.examples.robot_execution_failures import load_robot_execution_failures

# Load the example dataset
df, y = load_robot_execution_failures()

# Add a column to identify the individual time series
df['id'] = df.index // 15

# Define the settings for the feature extraction
settings = {'chunksize': None, 'disable_progressbar': True}

# Define the feature extraction function
def extract_features_with_flops(data, settings):
    # Create a Pyinstrument profiler object
    profiler = pyinstrument.Profiler()

    # Start profiling
    profiler.start()

    # Call the feature extraction function
    X = extract_features(data, column_id='id', **settings)

    # Stop profiling
    profiler.stop()

    # Get the total number of FLOPs required by the function

    return X, profiler

# Run the feature extraction with FLOP measurement
X, profiler = extract_features_with_flops(df, settings)


print("Number of FLOPs:", num_flops)





#endregion

#endregion