#for feature extraction and selection
import platform
import tsfresh
from tsfresh import extract_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute

class create_features:
    ## create feature extraction function (using tsfresh)
    def feature_extraction(df_sensor, sensor_column_names, feature_segment, sensor_frequency, exclusion_threshold_feature_segments,
                           label_column_name, time_column_name="timestamp"):

        df_final = pd.DataFrame()
        counter = 1

        # create list of columns which are used for feature extraction
        feature_extraction_columns = sensor_column_names.copy()
        feature_extraction_columns.append(time_column_name)
        feature_extraction_columns.append(label_column_name)
        if "user_id" in df_sensor.columns: # the device ID in the iteration 01 data
            #rename column "2" to "device_id"
            df_sensor = df_sensor.rename(columns={"user_id": "device_id"})
        feature_extraction_columns.append("device_id")

        # get only sensor columns and also label columns
        df_sensor = df_sensor[feature_extraction_columns]

        # count number of rows with nan value
        nan_rows = df_sensor.isnull().sum(axis=1)
        nan_rows = nan_rows[nan_rows > 0]
        print(f"Number of rows with nan values: {len(nan_rows)}")
        print(f"Percentage of rows with nan values: {len(nan_rows)/df_sensor.shape[0]}")

        # remove rows with NaN values & reset index
        df_sensor = df_sensor.dropna()
        df_sensor = df_sensor.reset_index(drop=True)
        # check: if there are less then segment_size records left, skip this iteration
        if (df_sensor.shape[0] < 10):
            print("Less than 10 records left, skip this iteration")
            return pd.DataFrame()

        # create ID column: used by feature extraction to identify different segments
        # TODO change this ID selection so that it doesn´t rely on counts but on the timestamp (for the
        # cases that the sampling frequency is not all the time 100% correct)
        #iterate through the unique ESM_timestamps
        df_sensor["ID"] = ""
        counter_id = 1

        #convert the time_column_name to datetime objects
        df_sensor[time_column_name] = pd.to_datetime(df_sensor[time_column_name])

        #iterate through users
        for user in df_sensor["device_id"].unique():
            print("start with user: ", user)
            # check if there is no data here
            if (df_sensor[df_sensor["device_id"] == user].shape[0] == 0):
                print("No data for user ", user)
                continue

            # create ID for every segment of size segment_size
            # get the smallest timestamp for the user and ESM_timestamp
            start_timestamp = df_sensor[df_sensor["device_id"] == user][time_column_name].min()
            end_timestamp = start_timestamp + datetime.timedelta(seconds=feature_segment)
            last_esm_timestamp = df_sensor[df_sensor["device_id"] == user][time_column_name].max()
            print("start_timestamp: ", start_timestamp)
            print("end_timestamp: ", last_esm_timestamp)

            while end_timestamp < last_esm_timestamp:
                # create ID colum; the ESM_timestamp has to be equal to ESM_timestamp
                df_sensor.loc[(df_sensor["device_id"] == user) & (
                            df_sensor[time_column_name] >= start_timestamp) & (
                            df_sensor[time_column_name] < end_timestamp), "ID"] = counter_id

                # analytics: count how many rows this ID has
                print("ID: ", counter_id, "number of rows: ", len(df_sensor[df_sensor["ID"] == counter_id]))
                print("start_timestamp: ", start_timestamp, "end_timestamp: ", end_timestamp)

                # increase counter
                counter_id += 1
                # increase start_timestamp and end_timestamp
                start_timestamp = end_timestamp
                end_timestamp = start_timestamp + datetime.timedelta(seconds=feature_segment)

        #outdated version: this version of creating the ID didn´t take the timestamps into account but did it by using only the frequency
        #df_sensor['ID'] = df_sensor.index // segment_size

        # make sure that in one segment, there is only one ESM event (check with ESM_timestamp);
        # if there are more than one ESM events in one segment, check which ESM has the most entries in the segment and delete the other
        #get_indices = df_sensor.groupby(['ID', 'ESM_timestamp'])["ESM_timestamp"].count().reset_index(name="count")
        # delete all rows with segment_size value
        #get_indices = get_indices[get_indices["count"] != segment_size]

        # for every ID, get the ESM_timestamp which has the most entries in the segment
        # and delete all other ESM_timestamps for that ID
        #for ID in get_indices["ID"].unique():
            # check if there is only one ESMTimestamp for that ID (is the case if lenght of the last
            # segment is not segment_size).
        #    if len(get_indices[get_indices["ID"] == ID]["ESM_timestamp"].unique()) == 1:
                # If so, check if the count is >1; if not, delete the row
        #        if int(get_indices[get_indices["ID"] == ID]["count"]) == 1:
                    # drop row with this ID from df_sensor
        #            df_sensor = df_sensor.drop(df_sensor[df_sensor["ID"] == ID].index)
        #        else:
        #            continue
        #        continue
                # print("ID with only one ESM_timestamp: ", ID)
            # get ESM_timestamp with least entries in segment
        #    ESM_timestamp = get_indices[get_indices["ID"] == ID]["count"].sort_values(ascending=False).index[1]
            # get indices of all rows with ID and ESM_timestamp
        #    indices = df_sensor[(df_sensor["ID"] == ID) & (df_sensor["ESM_timestamp"] == ESM_timestamp)].index
            # delete all rows with ID and ESM_timestamp
        #    df_sensor = df_sensor.drop(indices)

        # delete all IDs which have less than 70% of expected rows
        expected_rows = round(feature_segment * sensor_frequency * exclusion_threshold_feature_segments)
        num_rows_before_deletion = df_sensor.shape[0]
        # delete IDs which have less than exclusion_threshold_feature_segments% of expected rows
        df_sensor = df_sensor[df_sensor.groupby("ID")["ID"].transform('size') > expected_rows]
        print("Number of rows deleted because number or records for one ID was below : " + str(expected_rows) + " (70% of expected ID size): " + str(num_rows_before_deletion - df_sensor.shape[0]))

        # drop all rows with ID = ""
        # print: number of rows with ID = "" and percentage of rows with ID = ""
        print("number of rows with ID = \"\": ", len(df_sensor[df_sensor["ID"] == ""]))
        print("percentage of rows with ID = \"\": ", len(df_sensor[df_sensor["ID"] == ""]) / len(df_sensor))
        df_sensor = df_sensor[df_sensor["ID"] != ""]

        # create dataset which contains only device_id, timestamp and  ESM_timestamp (needed for later)
        df_intermediate = df_sensor[["device_id", "ID", time_column_name, label_column_name]].copy()

        # sort by timestamp
        df_sensor = df_sensor.sort_values(by=time_column_name)

        # drop unnecessary columns (i.e. columns which are not used for feature extraction)
        df_sensor = df_sensor.drop(columns=["device_id"])
        df_sensor = df_sensor.drop(columns=[label_column_name])
        if "1" in df_sensor.columns:
            df_sensor = df_sensor.drop(columns=["1"])
        if "2" in df_sensor.columns:
            df_sensor = df_sensor.drop(columns=["2"])

        # extract feature
        extracted_features = extract_features(df_sensor, column_id='ID', column_sort=time_column_name,
                                              n_jobs=8)

        # impute missing values
        # impute(extracted_features)

        # apply feature selection
        # create label pd.Series from df_event[label_column_name].iloc[0] with length of extracted_features
        # features_filtered = select_features(extracted_features, label_series)

        # apply feature extraction, imputing (deleting missing values) and feature selection in one step
        # features_filtered_direct = extract_relevant_features(df_sensor, label_series, column_id='ID', column_sort=time_column_name)

        # add label, device_id and ESM_timestamp to features_filtered
        #sort df_intermediate by timestamp
        df_intermediate = df_intermediate.sort_values(by=time_column_name)
        extracted_features["sensor_timestamp"] = df_intermediate.groupby("ID")[time_column_name].first()
        extracted_features["device_id"] = df_intermediate.groupby("ID")["device_id"].first()
        extracted_features[label_column_name] = df_intermediate.groupby("ID")[label_column_name].first()

        return extracted_features

    ## create function to select features
    def feature_selection(df_features, label_column_name, apply_tsfresh_feature_selection):
        # check how many columns are in df_features
        num_features_before_selection = df_features.shape[1]

        #initialize analytics objects
        analytics_name = []
        analytics_nrows_before = []
        analytics_nrows_after = []
        analytics_nrows_deleted = []
        analytics_ncolumns_before = []
        analytics_ncolumns_after = []
        analytics_ncolumns_deleted = []

        # if there is a column named "sensor_timestamp" rename it to "timestamp"
        if "sensor_timestamp" in df_features.columns:
            df_features.rename(columns={"sensor_timestamp": "timestamp"}, inplace=True)

        # delete columns which contain only one value
        analytics_name.append("delete columns which contain only one value")
        analytics_nrows_before.append(df_features.shape[0])
        analytics_ncolumns_before.append(df_features.shape[1])
        print("shape before deleting columns with only one value: ", df_features.shape)
        #df_features = df_features.loc[:, df_features.nunique() != 1] #computational complexity too high
        #df_features = df_features.loc[:, df_features.apply(pd.Series.nunique) != 1] #computational complexity too high
        drop_cols = df_features.columns[df_features.nunique() == 1]
        df_features.drop(columns=drop_cols, inplace=True)
        print("shape after deleting columns with only one value: ", df_features.shape)
        analytics_nrows_after.append(df_features.shape[0])
        analytics_ncolumns_after.append(df_features.shape[1])

        # delete column if more than 90% of values are missing
        analytics_name.append("delete columns if more than 90% of values are missing")
        analytics_nrows_before.append(df_features.shape[0])
        analytics_ncolumns_before.append(df_features.shape[1])
        print("shape before deleting columns with more than 90% missing values: ", df_features.shape)
        #print(df_features.columns[df_features.isnull().mean() > 0.9 ])
        drop_cols = df_features.columns[df_features.isnull().mean() > 0.9]
        df_features.drop(columns=drop_cols, inplace=True)
        # df_features = df_features.loc[:, df_features.isnull().mean() < .9] #computational complexity too high
        print("shape after deleting columns with more than 90% missing values: ", df_features.shape)
        analytics_nrows_after.append(df_features.shape[0])
        analytics_ncolumns_after.append(df_features.shape[1])

        # delete unnecessary columns
        device_id = df_features["device_id"]
        sensor_timestamp = df_features["timestamp"]
        df_features = df_features.drop(columns = ["timestamp", "device_id"])
        print("unnecessary columns deleted")

        # add label & delete missing values in label
        # add label column
        df_features["device_id"] = device_id
        df_features["timestamp"] = sensor_timestamp

        # count missing values in df_features and groupby ESM_timestamp
        print("missing values in df_features: ", df_features.isnull().sum())
        print("shape of df_features before deleting missing values in label: ", df_features.shape)
        analytics_name.append("delete rows with missing values in label")
        analytics_nrows_before.append(df_features.shape[0])
        analytics_ncolumns_before.append(df_features.shape[1])

        # delete rows which contain nan in label_column_name
        df_features.dropna(subset=[label_column_name], inplace=True)
        print("shape of df_features after deleting missing values in label: ", df_features.shape)
        analytics_nrows_after.append(df_features.shape[0])
        analytics_ncolumns_after.append(df_features.shape[1])

        # apply feature selection
        if apply_tsfresh_feature_selection == "yes":

            # create updated series for label, device_id and ESM_timestamp
            label = pd.Series(df_features[label_column_name])
            sensor_timestamp = df_features["timestamp"]
            device_id = df_features["device_id"]

            # delete label & ESM_timestamp column again
            df_features.drop(columns=[label_column_name, "device_id", "timestamp"], inplace=True)

            # impute missing values
            impute(df_features)
            # iterate through df_features in chunks of 1000 rows and impute missing values and save temporarily
            # for i in tqdm(range(0, df_features.shape[0], 10000)):
            # impute(df_features.iloc[i:i+10000, :])
            #    print("missing values imputed")
            #    df_features.iloc[i:i+10000, :] = df_features.iloc[i:i+10000, :].apply(lambda x: x.fillna(x.mean()), axis=0)
            print("imputation done")

            #apply the tsfresh feature selection
            analytics_name.append("apply tsfresh feature selection")
            analytics_nrows_before.append(df_features.shape[0])
            analytics_ncolumns_before.append(df_features.shape[1])
            t0 = time.time()
            features_filtered = select_features(df_features, label, n_jobs = 8)

            print("feature selection done")
            print("shape of df_features after feature selection : ", features_filtered.shape)
            analytics_nrows_after.append(features_filtered.shape[0])
            analytics_ncolumns_after.append(features_filtered.shape[1])
            #print how many features have been deleted
            num_features_after_selection = features_filtered.shape[1]
            print("number of features deleted: ", num_features_before_selection - num_features_after_selection)

            # add device id, ESM_timestamp and label column again
            features_filtered["device_id"] = device_id
            features_filtered[label_column_name] = label
            features_filtered["timestamp"] = sensor_timestamp
            print("device id and label added")
            print("time needed for feature selection: ", time.time() - t0)

            df_features = features_filtered.copy()

        # create analytics dataframe
        analytics_df = pd.DataFrame({"name": analytics_name, "nrows_before": analytics_nrows_before, "ncolumns_before": analytics_ncolumns_before, "nrows_after": analytics_nrows_after, "ncolumns_after": analytics_ncolumns_after})
        #create column which shows how many rows have been deleted
        analytics_df["nrows_deleted"] = analytics_df["nrows_before"] - analytics_df["nrows_after"]
        #create column which shows how many columns have been deleted
        analytics_df["ncolumns_deleted"] = analytics_df["ncolumns_before"] - analytics_df["ncolumns_after"]

        return df_features, analytics_df

    ## visualization: count columns & get column names
    def count_feature_columns_and_list_columnnames(df_features, sensor_list):
        df_results = pd.DataFrame(columns=["timeperiod", "sensor", "number_features", "number_records"])
        dict_columns = {}
        axes = ["double_values_0", "double_values_1", "double_values_2" ]
        # iterate through sensors, take only first three letters and count how many columns in df_features start with that three letters
        # save the number of columns and rows in a list
        for sensor in sensors:
            for axis in axes:
                # count how many columns start with first three letters of that sensor and the axis
                number_features = len([col for col in df_features.columns if col.startswith(sensor[:3] + "_" + axis)])
                # save number of features and number of records in df_results using concat
                df_results = pd.concat([df_results,
                                        pd.DataFrame([[timeperiod, sensor, axis, number_features, df_features.shape[0]]],
                                                     columns=["timeperiod", "sensor", "axis", "number_features",
                                                              "number_records"])], axis=0)
            # save list of columns in dict_columns
            dict_columns[sensor] = [col for col in df_features.columns if col.startswith(sensor[0:3])]
            print("sensor finished: ", sensor)

        return df_results, dict_columns

    # for battery_charges
    ## calculate difference between "before bed" event/"after bed" event timestamps and "charge battery" event timestamp/"charge
    ## battery double_end_timestamp" timestamp
    def battery_charges_features(self, df):
        # calculate difference between "before bed" event/"after bed" event timestamps and "charge battery" event timestamp/"charge
        # battery double_end_timestamp" timestamp

        return df