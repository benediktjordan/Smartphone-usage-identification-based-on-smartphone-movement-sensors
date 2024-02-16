#region import
#Classification
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

#nested CV
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
# import classification report
from sklearn.metrics import classification_report

#Feature Importance
import shap

#Statistical Testing
from scipy.stats import binom_test
from sklearn.model_selection import permutation_test_score

#endregion

class DecisionForest:

	#DF including LOSOCV, hyperparameter tuning, permutation test, etc
	#Note: since 20.02., the index of the df needs to be reset before using this function
	def DF_sklearn(df, label_column_name, n_permutations, path_storage, parameter_tuning,
				   confusion_matrix_order, title_confusion_matrix, title_feature_importance_grid, grid_search_space = None, feature_importance = None,
				   parameter_set = "default", combine_participants = False):
		"""
		Builds a decision forest using the sklearn library
		:param df:
		:param label_segment: has to be in seconds
		:return:
		"""

		# initialize metrics list for LOSOCV metrics
		timeperiod_list = []
		label_list = []
		acc_final_list = []
		bal_acc_final_list = []
		f1_final_list = []
		precision_final_list = []
		recall_final_list = []

		#reset index
		#df = df.reset_index(drop = True)

		# convert timestamp and ESM_timestamp to datetime
		# check if ESM_timestamp is in the df (since it isnt in the "location" datasets)
		if "ESM_timestamp" in df.columns:
			df["timestamp"] = pd.to_datetime(df["timestamp"])
			df["ESM_timestamp"] = pd.to_datetime(df["ESM_timestamp"])

		# initialize
		test_proband = list()
		outer_results_acc = list()
		outer_results_acc_balanced = list()
		outer_results_f1 = list()
		outer_results_precision = list()
		outer_results_recall = list()
		best_parameters = list()
		outer_results_best_params = list()

		permutation_pvalue = list()
		permutation_modelaccuracy = list()
		pvalues_binomial = list()

		#test_proband_array = np.empty([0])
		#y_test_array = np.empty([0])
		#y_pred_array = np.empty([0])
		df_labels_predictions = pd.DataFrame()

		shap_values_dict = dict()

		# Make list of all participants
		IDlist = set(df["device_id"])
		# make different IDlist in case a simple train-test split should be used for outer iteration (for parameter tuning, LOSOCV will still be used)
		if combine_participants == True:
			IDlist = set(df["device_id_traintest"])


		# for loop to iterate through LOSOCV "rounds"
		counter = 1
		num_participants = len(IDlist)
		for i in tqdm(IDlist):
			print("Start with participant " + str(i) + " as test participant")
			t0_inner = time.time()

			#split data into train and test
			LOOCV_O = i
			df_train = df[df["device_id"] != LOOCV_O]
			df_test = df[df["device_id"] == LOOCV_O]
			if combine_participants == True:
				if i == 1:
					print("This iteration will be skipped since train-set is test-set")
					continue
				df_train = df[df["device_id_traintest"] != LOOCV_O]
				df_test = df[df["device_id_traintest"] == LOOCV_O]

			# define Test data - the person left out of training
			##  check if ESM_timestamp is in the df (since it isnt in the "location" datasets)
			X_test_df = df_test.drop(columns=[label_column_name, "device_id",
												"timestamp"])  # add sensor_timestamp here as soon as available

			X_test = np.array(X_test_df)
			y_test_df = df_test[label_column_name]  # This is the outcome variable
			y_test = np.array(y_test_df)

			# jump over this iteration if y_test contains only one class
			if len(set(y_test)) == 1:
				print("y_test contains only one class")
				continue

			# define Train data - all other people in dataframe
			X_train_df = df_train.copy()

			# define the model
			if parameter_set == "default":
				model = RandomForestClassifier(random_state=11)
			else:
				model = RandomForestClassifier(**parameter_set)


			# define list of indices for inner CV for the GridSearch (use again LOSOCV with the remaining subjects)
			# here a "Leave Two Subejcts Out" CV is used!
			if parameter_tuning == "yes":
				IDlist_inner = list(set(X_train_df["device_id"]))
				inner_idxs = []
				X_train_df = X_train_df.reset_index(drop=True)
				for l in range(0, len(IDlist_inner), 2):
					try:
						IDlist_inner[l + 1]
					except:
						continue
					else:
						train = X_train_df[
							(X_train_df["device_id"] != IDlist_inner[l]) & (X_train_df["device_id"] != IDlist_inner[l + 1])]
						test = X_train_df[
							(X_train_df["device_id"] == IDlist_inner[l]) | (X_train_df["device_id"] == IDlist_inner[l + 1])]
						add = [train.index, test.index]
						inner_idxs.append(add)

			# drop participant column
			df_train = df_train.drop(columns=["device_id"])
			if combine_participants == True:
				df_train = df_train.drop(columns=["device_id_traintest"])

			# drop other columns
			## check if ESM_timestamp is in the df (since it isnt in the "location" datasets)

			X_train_df = X_train_df.drop(columns=[label_column_name, "device_id", "timestamp"])

			#X_train = np.array(X_train_df)
			y_train_df = df_train[label_column_name]  # This is the outcome variable
			y_train_df = y_train_df.reset_index(drop=True)
			#y_train = np.array(y_train_df)  # Outcome variable here

			# parameter tuning: only do, if parameter_tuning is set to True
			if parameter_tuning == "yes":

				# define search
				print("Start parameter tuning")
				search = GridSearchCV(model, grid_search_space, scoring='balanced_accuracy', cv=inner_idxs, refit=True, n_jobs=-1)

				# execute search
				print("Start fitting model with parameter tuning...")
				result = search.fit(X_train_df, y_train_df)
				print("Model fitted.")
				print("Best: %f using %s" % (result.best_score_, result.best_params_))

				# get the best performing model fit on the whole training set
				best_model = result.best_estimator_
				best_params = result.best_params_
				parameter_tuning_active = "yes"

			# if parameter tuning is set to False, use the default parameters
			else:
				print("Start fitting model without parameter tuning...")
				best_model = model.fit(X_train_df, y_train_df)
				parameter_tuning_active = "no"

			# save the model
			with open(path_storage + label_column_name + "_parameter_tuning-"+ parameter_tuning_active +"_test_proband-" + str(
					i) + "_model.sav", 'wb') as f:
				pickle.dump(best_model, f)
			print("Best model: ", best_model)

			# apply permutation test
			print("Start permutation test...")
			## create dataframe which contains all data and delete some stuff
			data_permutation = df.copy()
			data_permutation = data_permutation.reset_index(drop=True)

			## create list which contains indices of train and test samples (differentiate by proband)
			split_permutation = []
			train_permutation = data_permutation[data_permutation["device_id"] != i]
			test_permutation = data_permutation[data_permutation["device_id"] == i]
			if combine_participants == True:
				train_permutation = data_permutation[data_permutation["device_id_traintest"] != i]
				test_permutation = data_permutation[data_permutation["device_id_traintest"] == i]
			add_permutation = [train_permutation.index, test_permutation.index]
			split_permutation.append(add_permutation)

			##Drop some stuff
			# data_permutation = data_permutation.drop(columns=dropcols)

			##Create X and y dataset
			### check if ESM_timestamp is in the df (since it isnt in the "location" datasets)

			X_permutation = data_permutation.drop(columns=[label_column_name, "device_id", "timestamp"])
			y_permutation = data_permutation[label_column_name]

			##compute permutation test
			score_model, perm_scores_model, pvalue_model = permutation_test_score(best_model, X_permutation,
																				  y_permutation, scoring="balanced_accuracy",
																				  cv=split_permutation,
																				  n_permutations=n_permutations,
																				  n_jobs=-1)
			print("Permutation test done.")

			## visualize permutation test results
			fig, ax = plt.subplots(figsize=(10, 5))
			plt.title("Results of Permutation Test (Participant " + str(i) + " as Test-Data)")
			# create histogram with sns histplot
			sns.histplot(perm_scores_model, bins=50, stat="density", ax=ax)

			#add vertical line for score on original data
			#ax.hist(perm_scores_model, bins=(n_permutations/2), density=True)
			ax.axvline(score_model, ls='--', color='r')
			#score_label = (f"Score on original\ndata: {score_model:.2f}\n"
			#			   f"(p-value: {pvalue_model:.3f})")
			# put text in upper right corner
			#ax.text(0.9, 0.8, score_label, transform=ax.transAxes)
			plt.tight_layout()
			ax.set_xlabel("Balanced Accuracy")
			ax.set_ylabel("Count")
			plt.show()

			plt.savefig(path_storage + label_column_name + "_parameter_tuning-"+ parameter_tuning_active + "_test_proband-" + str(
					i) + "_Permutation.png", bbox_inches='tight', dpi = 300)
			# plt.show()


			# evaluate model on the hold out dataset
			## create predictions
			print("Start evaluating model...")
			yhat = best_model.predict(X_test_df)

			## create probabilities for each class
			probabilities = best_model.predict_proba(X_test_df)
			df_probabilities = pd.DataFrame(probabilities, columns=best_model.classes_)
			## set index of X_test_df to the index of df_probabilities
			df_probabilities = df_probabilities.set_index(X_test_df.index)

			# evaluate the model
			acc = accuracy_score(y_test_df, yhat)
			acc_balanced = balanced_accuracy_score(y_test_df, yhat)
			print('Balanced Accuracy: %.3f' % acc_balanced)
			f1 = f1_score(y_true=y_test_df, y_pred=yhat, average="weighted")
			precision = precision_score(y_true=y_test_df, y_pred=yhat, average="weighted")
			recall = recall_score(y_true=y_test_df, y_pred=yhat, average="weighted")
			# TODO document this: why I am using balanced accuracy (https://scikit-learn.org/stable/modules/model_evaluation.html#balanced-accuracy-score)
			# and weighted f1, precision and recall (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html=

			# convert categorical labels to numerical labels and save the mapping for later visualization
			#convert y_test to pandas series
			y_test_confusionmatrix = pd.Series(y_test_df)
			y_test_confusionmatrix = y_test_confusionmatrix.astype('category')
			label_mapping = dict(enumerate(y_test_confusionmatrix.cat.categories))
			#y_test_confusionmatrix = y_test_confusionmatrix.cat.codes
			#y_test_confusionmatrix = y_test_confusionmatrix.to_numpy()
			# also convert yhat to numerical labels using same mapping
			yhat_confusionmatrix = pd.Series(yhat)
			yhat_confusionmatrix = yhat_confusionmatrix.astype('category')
			label_mapping2 = dict(enumerate(yhat_confusionmatrix.cat.categories))
			#yhat_confusionmatrix = yhat_confusionmatrix.cat.codes
			#yhat_confusionmatrix = yhat_confusionmatrix.to_numpy()

			# create joint label mapping which is == confusion_matrix_order except the elements which are not in label_mapping or label_mapping2
			label_mapping_joint = list(set(label_mapping.values()) | set(label_mapping2.values()))
			label_mapping_confusion_matrix = confusion_matrix_order.copy()
			for key in label_mapping_confusion_matrix:
				if key not in label_mapping_joint:
					#delete from list
					label_mapping_confusion_matrix.remove(key)

			# Visualize Confusion Matrix with absolute values
			fig, ax = plt.subplots(figsize=(10, 5))
			#plt.gcf().subplots_adjust(bottom=0.15)
			mat = confusion_matrix(y_test_confusionmatrix, yhat_confusionmatrix, labels = label_mapping_confusion_matrix)
			sns.heatmap(mat, annot=True, annot_kws={'size': 10}, cmap=plt.cm.Greens, fmt='d', cbar=False, linewidths=0.2)
			plt.title('Confusion Matrix Absolute Values with Test-Proband ' + str(i) )
			plt.suptitle('Balanced Accuracy: {0:.3f}'.format(acc_balanced), fontsize=16)
			# add xticks and yticks from label_mapping (which is a dictionary)
			tick_marks = np.arange(len(label_mapping_confusion_matrix)) + 0.5
			plt.xticks(tick_marks, label_mapping_confusion_matrix, rotation=45)
			plt.yticks(tick_marks, label_mapping_confusion_matrix, rotation=0)
			plt.xlabel('Predicted Label')
			plt.ylabel('True Label')
			#plt.show()
			plt.savefig(path_storage + label_column_name + "_parameter_tuning-"+ parameter_tuning_active + "_test_proband-" + str(
					i) + "_ConfusionMatrix_absolute.png", bbox_inches="tight")


			# visualize confusion matrix with percentages
			# Get and reshape confusion matrix data
			matrix = mat.astype('float') / mat.sum(axis=1)[:, np.newaxis]
			plt.figure(figsize=(16, 7))
			sns.set(font_scale=1.4)
			sns.heatmap(matrix, annot=True, annot_kws={'size': 10},cmap=plt.cm.Greens, linewidths=0.2)
			# Add labels to the plot
			tick_marks = np.arange(len(label_mapping_confusion_matrix)) + 0.5
			plt.xticks(tick_marks, label_mapping_confusion_matrix, rotation=45)
			plt.yticks(tick_marks, label_mapping_confusion_matrix, rotation=0)
			plt.xlabel('Predicted Label')
			plt.ylabel('True Label')
			plt.title('Confusion Matrix Relative Values with Test-Proband ' + str(i) )
			plt.suptitle('Balanced Accuracy: {0:.3f}'.format(acc_balanced), fontsize=16)
			plt.savefig(path_storage + label_column_name +  "_parameter_tuning-"+ parameter_tuning_active + "_test_proband-" + str(
					i) + "_ConfusionMatrix_percentages.png", bbox_inches="tight")
			#plt.show()

			# apply binomial test
			print("Start binomial test...")
			# calculate sum of diagonal axis of confusion matrix
			sum_diagonal = 0
			for l in range(len(mat)):
				sum_diagonal += mat[l][l]
			# calculate number of classes in y_test
			classes = len(np.unique(y_test))
			pvalue_binom = binom_test(x=sum_diagonal, n=len(y_test), p=(1/classes), alternative='greater')
			print("P-value binomial test: ", pvalue_binom)
			print("Binomial test done.")

			# feature importance: compute SHAP values
			#TODO include here rather DF explanatory variables than SHAP values
			if feature_importance == "shap":
				print("Start computing SHAP values...")
				explainer = shap.Explainer(best_model)
				shap_values = explainer.shap_values(X_test_df)

				# Compute the absolute averages of the SHAP values for each sample and feature across all classes
				## Explanation: in shap_values are three dimensions: classes x samples x features
				## In absolute_average_shape_values, the absolute average for each feature and sample over the classes
				## is computed. The result is a matrix with two dimensions: samples x features
				absolute_average_shap_values = np.mean(np.abs(shap_values), axis=0)

				fig, ax = plt.subplots(figsize=(10, 5))
				plt.title("Feature Importance for iteration with proband " + str(i) + " as test set")
				shap.summary_plot(absolute_average_shap_values, X_test_df.iloc[0:0], plot_type="bar", show=False,
								  plot_size=(20, 10))
				plt.show()
				fig.savefig(path_storage + label_column_name + "_parameter_tuning-" + parameter_tuning_active + "_test_proband-" + str(
					i) + "_SHAPFeatureImportance.png")

				# store the SHAP values (in order to get combined SHAP values for the whole LOSOCV in the end)
				## create dictionary which contains as key the proband number and as values the SHAP values and the best_model.classes_
				shap_values_dict[i] = [shap_values, best_model.classes_]

			# store statistical test results (p-value permutation test, accuracy of that permutation iteration, pvalue binomial test) in list
			print("Start storing statistical test results...")
			permutation_pvalue.append(pvalue_model)
			permutation_modelaccuracy.append(score_model)
			pvalues_binomial.append(pvalue_binom)

			# store the resulted metrics
			test_proband.append(i)
			outer_results_acc.append(acc)
			outer_results_acc_balanced.append(acc_balanced)
			outer_results_f1.append(f1)
			outer_results_precision.append(precision)
			outer_results_recall.append(recall)
			if parameter_tuning == "yes":
				outer_results_best_params.append(best_params)

			# store the y_test and yhat and probabilities for the final accuracy computation using concatenation
			## transform y_test_df into dataframe and add column with y_pred and column with proband number

			# For each class, label the sample with the highest probability as that class, and set all others to NaN
			df_labels_predictions_intermediate = pd.DataFrame(y_test_df)
			# concatenate df_probabilities with df_labels_predictions_intermediate based on index
			df_labels_predictions_intermediate = pd.concat([df_labels_predictions_intermediate, df_probabilities], axis=1)
			df_labels_predictions_intermediate = df_labels_predictions_intermediate.rename(columns={label_column_name: "y_test"})
			df_labels_predictions_intermediate["y_pred"] = yhat
			df_labels_predictions_intermediate["test_proband"] = i
			## concatenate the dataframes
			df_labels_predictions = pd.concat([df_labels_predictions, df_labels_predictions_intermediate])


			# report progress
			t1_inner = time.time()
			print("Time for participant " + str(counter) + "/" + str(num_participants) + " has been " + str((t1_inner - t0_inner)/60) + " minutes.")
			counter += 1
		# print('>acc=%.3f,f1=%.3f,precision=%.3f,recall=%.3f, est=%.3f, cfg=%s' % (acc, f1, precision, recall, result.best_score_, result.best_params_))
		# print("Permutation-Test p-value was " + str(pvalue_model) + " and Binomial Test p-values was " + str(pvalue_binom))
		# print("The proband taken as test-data for this iteration was " + str(i))
		# print("This inner iteration has taken so many minutes: " + str((t1_inner - t0_inner)/60))

		# Save the resulting metrics:
		results_LOSOCV = pd.DataFrame()
		results_LOSOCV["Test-Proband"] = test_proband
		results_LOSOCV["Accuracy"] = outer_results_acc
		results_LOSOCV["Balanced Accuracy"] = outer_results_acc_balanced
		results_LOSOCV["Accuracy by PermutationTest"] = permutation_modelaccuracy
		results_LOSOCV["F1"] = outer_results_f1
		results_LOSOCV["Precision"] = outer_results_precision
		results_LOSOCV["Recall"] = outer_results_recall
		results_LOSOCV["P-Value Permutation Test"] = permutation_pvalue
		results_LOSOCV["P-Value Binomial Test"] = pvalues_binomial
		# add best parameters if parameter tuning was active
		if parameter_tuning == "yes":
			results_LOSOCV["Best Parameters"] = outer_results_best_params
		# add label column name as column
		results_LOSOCV["Label Column Name"] = label_column_name
		# add seconds around event as column
		# add timeperiod of features as column
		results_LOSOCV.to_csv(path_storage + label_column_name + "_parameter_tuning-"+ parameter_tuning + "_results_LOSOCV.csv")

		# save the y_test and yhat for the final accuracy computation
		df_labels_predictions.to_csv(path_storage + label_column_name +  "_parameter_tuning-"+ parameter_tuning +  "_results_labelsRealAndPredicted.csv")

		# TODO: document this!
		# compute the final metrics for this LOSOCV: here the cumulated y_test and yhat are used in order to account
		# for the fact that some test-participants have more data than others AND that some participants more label-classes
		# were present then for other participants
		balanced_accuracy_overall = balanced_accuracy_score(df_labels_predictions["y_test"], df_labels_predictions["y_pred"])
		results_overall = pd.DataFrame()
		results_overall["Label"] = [label_column_name]
		results_overall["Balanced Accuracy"] = [balanced_accuracy_overall]
		results_overall["Accuracy"] = [accuracy_score(df_labels_predictions["y_test"], df_labels_predictions["y_pred"])]
		results_overall["F1"] = [f1_score(df_labels_predictions["y_test"], df_labels_predictions["y_pred"], average="macro")]
		results_overall["Precision"] = [precision_score(df_labels_predictions["y_test"], df_labels_predictions["y_pred"], average="macro")]
		results_overall["Recall"] = [recall_score(df_labels_predictions["y_test"], df_labels_predictions["y_pred"], average="macro")]

		# visualize confusion matrix for all y_test and y_pred data
		# convert categorical labels to numerical labels and save the mapping for later visualization
		# convert y_test to pandas series
		y_test_confusionmatrix = df_labels_predictions["y_test"]
		y_test_confusionmatrix = y_test_confusionmatrix.astype('category')
		label_mapping = dict(enumerate(y_test_confusionmatrix.cat.categories))
		#y_test_confusionmatrix = y_test_confusionmatrix.cat.codes
		#y_test_confusionmatrix = y_test_confusionmatrix.to_numpy()

		# also convert yhat to numerical labels using same mapping
		yhat_confusionmatrix = df_labels_predictions["y_pred"]
		yhat_confusionmatrix = yhat_confusionmatrix.astype('category')
		label_mapping2 = dict(enumerate(yhat_confusionmatrix.cat.categories))
		#yhat_confusionmatrix = yhat_confusionmatrix.cat.codes
		#yhat_confusionmatrix = yhat_confusionmatrix.to_numpy()

		# create joint label mapping which is == confusion_matrix_order except the elements which are not in label_mapping or label_mapping2
		label_mapping_joint = list(set(label_mapping.values()) | set(label_mapping2.values()))
		label_mapping_confusion_matrix = confusion_matrix_order.copy()
		for key in label_mapping_confusion_matrix:
			if key not in label_mapping_joint:
				# delete from list
				label_mapping_confusion_matrix.remove(key)

		# Visualize Confusion Matrix with absolute values
		plt.figure(figsize=(16, 7))
		sns.set(font_scale=1.4)
		mat = confusion_matrix(y_test_confusionmatrix, yhat_confusionmatrix, labels = label_mapping_confusion_matrix)
		sns.heatmap(mat, square=True, annot=True, annot_kws={'size': 10}, cmap=plt.cm.Greens, fmt='d', cbar=False,
					linewidths=0.2)

		plt.title("Confusion Matrix Absolute Values for All Participants")
		plt.suptitle('Balanced Accuracy: {0:.3f}'.format(balanced_accuracy_overall), fontsize=16)
		# add xticks and yticks from label_mapping (which is a dictionary)
		tick_marks = np.arange(len(label_mapping_confusion_matrix)) + 0.5
		plt.xticks(tick_marks, label_mapping_confusion_matrix, rotation=45)
		plt.yticks(tick_marks, label_mapping_confusion_matrix, rotation=0)
		plt.xlabel('Predicted Label')
		plt.ylabel('True Label')
		#plt.show()
		plt.savefig(path_storage + label_column_name + "_parameter_tuning-" + parameter_tuning_active + "_ConfusionMatrix_absolute.png",
					bbox_inches="tight")
		#

		# visualize confusion matrix with percentages
		# Get and reshape confusion matrix data
		matrix = mat.astype('float') / mat.sum(axis=1)[:, np.newaxis]
		plt.figure(figsize=(16, 7))
		sns.set(font_scale=1.4)
		sns.heatmap(matrix, annot=True, annot_kws={'size': 10}, cmap=plt.cm.Greens, linewidths=0.2)
		# Add labels to the plot
		plt.xticks(tick_marks, label_mapping_confusion_matrix, rotation=45)
		plt.yticks(tick_marks, label_mapping_confusion_matrix, rotation=0)
		plt.xlabel('Predicted Label')
		plt.ylabel('True Label')
		plt.title("Confusion Matrix Relative Values for All Participants")
		plt.suptitle('Balanced Accuracy: {0:.3f}'.format(balanced_accuracy_overall), fontsize=16)
		plt.savefig(path_storage + label_column_name + "_parameter_tuning-" + parameter_tuning_active +
					"_ConfusionMatrix_percentages.png", bbox_inches="tight")
		#plt.show()

		# visualize confusion matrix with percentages and absolute values combined
		matrix = mat.astype('float') / mat.sum(axis=1)[:, np.newaxis]
		matrix_abs = mat.astype('float')
		plt.figure(figsize=(16, 7))
		sns.set(font_scale=1.4)
		sns.heatmap(matrix, annot=True, annot_kws={'size': 10}, cmap=plt.cm.Greens, linewidths=0.2)
		for i, j in itertools.product(range(mat.shape[0]), range(mat.shape[1])):
			if matrix[i, j] > 0.5:
				text_color = 'white'
			else:
				text_color = 'black'
			plt.text(j + 0.5, i + 0.75, '({0})'.format(int(matrix_abs[i, j])), ha='center', va='center', fontsize=10,
					 color=text_color)
		plt.xticks(tick_marks, label_mapping_confusion_matrix, rotation=45)
		plt.yticks(tick_marks, label_mapping_confusion_matrix, rotation=0)
		plt.xlabel('Predicted Label')
		plt.ylabel('True Label')
		plt.title(title_confusion_matrix, fontsize=16)
		#plt.suptitle('Balanced Accuracy: {0:.3f}'.format(balanced_accuracy_overall), fontsize=10)
		#plt.show()
		plt.savefig(path_storage + label_column_name +"_parameter_tuning-" + parameter_tuning_active +
					"_ConfusionMatrix_recall_absolute.png", bbox_inches="tight", dpi=600)

		# visualize confusion matrix with percentages and absolute values combined
		matrix = mat.astype('float') / mat.sum(axis=0)[:, np.newaxis]
		matrix_abs = mat.astype('float')
		plt.figure(figsize=(16, 7))
		sns.set(font_scale=1.4)
		sns.heatmap(matrix, annot=True, annot_kws={'size': 10}, cmap=plt.cm.Greens, linewidths=0.2)
		for i, j in itertools.product(range(mat.shape[0]), range(mat.shape[1])):
			if matrix[i, j] > 0.5:
				text_color = 'white'
			else:
				text_color = 'black'
			plt.text(j + 0.5, i + 0.75, '({0})'.format(int(matrix_abs[i, j])), ha='center', va='center', fontsize=10,
					 color=text_color)
		plt.xticks(tick_marks, label_mapping_confusion_matrix, rotation=45)
		plt.yticks(tick_marks, label_mapping_confusion_matrix, rotation=0)
		plt.xlabel('Predicted Label')
		plt.ylabel('True Label')
		plt.title(title_confusion_matrix, fontsize=16)
		# plt.suptitle('Balanced Accuracy: {0:.3f}'.format(balanced_accuracy_overall), fontsize=10)
		# plt.show()
		plt.savefig(path_storage + label_column_name + "_parameter_tuning-" + parameter_tuning_active +
					"_ConfusionMatrix_precision_absolute.png", bbox_inches="tight", dpi=500)

		# visualize SHAP values for whole LOSOCV
		## save raw shap values "shap_values_dict" to pickle
		with open (path_storage + label_column_name +  "_parameter_tuning-" + parameter_tuning_active + "_SHAPValues_AllLOSOCVIterations.pkl", "wb") as f:
			pickle.dump(shap_values_dict, f)

		## the a list of shap value lists: each sub-list are the joint shap values for one class; classes are ordered according to confusion_matrix_order
		shap_values_list_joint = []
		for i in range(len(confusion_matrix_order)):
			shap_values_list_joint.append(np.empty((0, len(X_test_df.columns)), float))

		for key in shap_values_dict:
			# iterate through shap_values_dict[key][1]
			classes = shap_values_dict[key][1]
			counter = 0
			for single_class in classes:
				# find out at which place this class is in confusion_matrix_order
				index = confusion_matrix_order.index(single_class)
				# append corresponding shap values to shap_values_list
				shap_values_list_joint[index] = np.append(shap_values_list_joint[index], shap_values_dict[key][0][counter], axis=0)
				counter += 1

		## visualize the shap values for whole LOSOCV and each individual class
		for single_class in confusion_matrix_order:
			index = confusion_matrix_order.index(single_class)

			fig, ax = plt.subplots(figsize=(10, 5))
			plt.title("Feature Importance for LOSOCV Combined For Class " + single_class)
			shap.summary_plot(shap_values_list_joint[index], X_test_df.iloc[0:0], plot_type="bar", show=False,
							  plot_size=(20, 10), max_display=5)
			plt.xlabel("Average of Absolute SHAP Values")
			#plt.show()
			# replace in "single_class" all "/" with "-" (otherwise the file cannot be saved)
			single_class = single_class.replace("/", "-")
			fig.savefig(path_storage + label_column_name + "_parameter_tuning-" +
						parameter_tuning_active + "_SHAPValues_class-" + single_class + "_AllLOSOCVIterations.png", bbox_inches="tight")

		## visualize the shap values for whole LOSOCV and each individual class in a grid
		# determine the number of classes and calculate the number of rows and columns for the grid
		num_classes = len(confusion_matrix_order)
		num_cols = min(num_classes, 2)  # set the maximum number of columns to 3
		num_rows = (num_classes - 1) // num_cols + 1

		# create the grid of subplots
		fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 5))
		plt.subplots_adjust(left=0.1, bottom=0.1, right=10000.9, top=0.9, wspace=0.5, hspace=0.5)

		# create title for whole figure
		fig.suptitle(title_feature_importance_grid, fontsize=16)

		# iterate over the classes and plot the SHAP summary plot on the appropriate subplot
		for i, single_class in enumerate(confusion_matrix_order):
			index = confusion_matrix_order.index(single_class)

			# determine the row and column indices for the current subplot
			row_idx = i // num_cols
			col_idx = i % num_cols

			# create a new subplot on the current grid location
			plt.subplot(num_rows, num_cols, i + 1)
			plt.title(single_class)
			shap.summary_plot(shap_values_list_joint[index], X_test_df.iloc[0:0], plot_type="bar", show=False,
							  plot_size=(20, 10), max_display=5)
			plt.xlabel("Average of Absolute SHAP Values")
			#plt.tick_params(axis='y', which='major', labelsize=8)
			#plt.tick_params(axis='y', which='major', labelrotation=45)

			# replace in "single_class" all "/" with "-" (otherwise the file cannot be saved)
			single_class = single_class.replace("/", "-")

		# adjust the spacing between the subplots and show the figure
		fig.tight_layout()
		plt.show()
		#save
		fig.savefig(path_storage + label_column_name +"_parameter_tuning-" +
					parameter_tuning_active + "_SHAPValues_AllClassesInGrid_AllLOSOCVIterations.png",
					 dpi=600)


		## visualize the shap values for whole LOSOCV and all classes combined
		### take the absolute averages of the shap values over all classes
		shap_values_averaged_samples = [np.mean(np.abs(shap_values_list_joint[i]), axis=0) for i in range(len(confusion_matrix_order))]
		shap_values_averaged_samples_classes = np.mean(np.abs(shap_values_averaged_samples), axis=0)

		### have to reshape in order to match need of summary_plot: (n_samples, n_features); create artificial samples by duplicating the shap values
		### NOTE: this wasnt possible before because sample values from different LOSOCV iteration can´t be averaged, as there are
		### sometimes different number of sample values for each class
		shap_values_averaged_samples_classes_includingartificialsamples = np.repeat(
			shap_values_averaged_samples_classes[np.newaxis, :], 2, axis=0)

		fig, ax = plt.subplots(figsize=(10, 5))
		plt.title("Feature Importance for whole LOSOCV combined (absolute average SHAP values)")
		shap.summary_plot(shap_values_averaged_samples_classes_includingartificialsamples, X_test_df.iloc[0:0], plot_type="bar", show=False,
						  plot_size=(20, 10))
		#plt.show()
		fig.savefig(
			path_storage + label_column_name + "_parameter_tuning-" +
			parameter_tuning_active + "_SHAPValues_all-classes-combined_AllLOSOCVIterations.png",
			bbox_inches="tight")


		# TODO include here also a binomial test for the final accuracy
		# TODO think how to apply permutation test here
		return results_overall, df_labels_predictions

	#DF only used for training the final deployment model: train on all data without testing
	## Note: this doesnt include LOSOCV, hyperparameter tuning, training-testing, permutation/binomial tests, etc
	def DF_sklearn_deployment(df, label_column_name, parameter_set, path_storage):
		# Transform data needed for DF
		## reset index
		df = df.reset_index(drop=True)
		## convert timestamp and ESM_timestamp to datetime
		df["timestamp"] = pd.to_datetime(df["timestamp"])

		## drop columns
		X_train_df =  df.drop(columns=[label_column_name, "device_id", "timestamp"])  # add sensor_timestamp later on here
		y_train_df = df[label_column_name]  # This is the outcome variable
		y_train_df = y_train_df.reset_index(drop=True)

		#define model and train
		model = RandomForestClassifier(**parameter_set)
		print("Start fitting model...")
		best_model = model.fit(X_train_df, y_train_df)


		return best_model