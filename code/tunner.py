RANDOM_STATE = 42
TEST_SIZE = 0.2
import select_k
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

def tune_random_forest(my_dataset, features_list):
	parameters = {'bootstrap': [True, False],
	 'max_features': ['auto', 'sqrt'],
	 'min_samples_leaf': [1, 2, 4],
	 'min_samples_split': [2, 5, 10]}

	best_parameters = {'bootstrap': None,
	 'max_features': None,
	 'min_samples_leaf': None,
	 'min_samples_split': None}
	best_clf = None
	best_list = None
	best_k = 0
	best_success = 0

	print "==== Random Forest Tunning ===="
	for k in range(1,len(features_list)):
		print "SelectK:", k
		k_list = select_k.get_features(my_dataset, features_list, k)

		data = featureFormat(my_dataset, k_list)
		labels, features = targetFeatureSplit(data)

		features_train, features_test, labels_train, labels_test = [],[],[],[]

		number_of_samples = 10
		for i in range(0,number_of_samples):
			features_train_sample, features_test_sample, labels_train_sample, labels_test_sample = \
			    train_test_split(features, labels, test_size=TEST_SIZE, random_state=(RANDOM_STATE + i))
			features_train.append(features_train_sample)
			features_test.append(features_test_sample)
			labels_train.append(labels_train_sample)
			labels_test.append(labels_test_sample)

		# Iterate through the possible settings
		for bootstrap in parameters['bootstrap']:
			for max_features in parameters['max_features']:
				for min_samples_leaf in parameters['min_samples_leaf']: 
					for min_samples_split in parameters['min_samples_split']:
						# Number of samples that recall > 0.3 and precision > 0.3
						success = 0
						rf_clf = RandomForestClassifier(bootstrap=bootstrap, max_features=max_features, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, random_state=RANDOM_STATE)
						# Iterates through the samples
						for i in range(0,number_of_samples):
							rf_clf.fit(features_train[i], labels_train[i])
							rf_labels = rf_clf.predict(features_test[i])
							rf_recall = metrics.recall_score(labels_test[i], rf_labels)
							rf_precision = metrics.precision_score(labels_test[i], rf_labels)
							#If the condition is met, than we count a success.
							if rf_recall > 0.3 and rf_precision > 0.3:
								success += 1
								if success > best_success:
									best_clf = rf_clf
									best_success = success
									best_k = k
									best_list = k_list
									best_parameters = {'bootstrap': bootstrap,
									 'max_features': max_features,
									 'min_samples_leaf': min_samples_leaf,
									 'min_samples_split': min_samples_split}

	print "best_k:",best_k
	print "best_list:",best_list
	print "best_parameters:",best_parameters
	print "best_success:",best_success
	print "=== Finishing Random Forest Tunning ==="
	return best_clf

def tune_decision_tree(my_dataset, features_list):
	parameters = {'criterion': ['gini', 'entropy'],
	 'min_samples_split': [2, 5, 10],
	 'max_depth': [10, 50, None]}

	best_parameters = {'criterion': None,
	 'min_samples_split': None,
	 'max_depth': None}

	best_clf = None
	best_list = None
	best_k = 0
	best_success = 0

	print "==== Decision Tree Tunning ===="
	for k in range(1,len(features_list)):
		print "SelectK:", k
		k_list = select_k.get_features(my_dataset, features_list, k)

		data = featureFormat(my_dataset, k_list)
		labels, features = targetFeatureSplit(data)

		features_train, features_test, labels_train, labels_test = [],[],[],[]
		# Since DecisionTree computes faster, the number of samples can be bigger to achieve higher accuracy
		number_of_samples = 10
		for i in range(0,number_of_samples):
			features_train_sample, features_test_sample, labels_train_sample, labels_test_sample = \
			    train_test_split(features, labels, test_size=TEST_SIZE, random_state=(RANDOM_STATE + i))
			features_train.append(features_train_sample)
			features_test.append(features_test_sample)
			labels_train.append(labels_train_sample)
			labels_test.append(labels_test_sample)

		# Iterate through the possible settings
		for criterion in parameters['criterion']:
			for min_samples_split in parameters['min_samples_split']:
				for max_depth in parameters['max_depth']: 
					# Number of samples that recall > 0.3 and precision > 0.3
					success = 0
					dt_clf = DecisionTreeClassifier(criterion=criterion, min_samples_split=min_samples_split, max_depth=max_depth, random_state=RANDOM_STATE)
					# Iterates through the samples
					for i in range(0,number_of_samples):
						dt_clf.fit(features_train[i], labels_train[i])
						dt_labels = dt_clf.predict(features_test[i])
						dt_recall = metrics.recall_score(labels_test[i], dt_labels)
						dt_precision = metrics.precision_score(labels_test[i], dt_labels)
						#If the condition is met, than we count a success.
						if dt_recall > 0.3 and dt_precision > 0.3:
							success += 1
							if success > best_success:
								best_clf = dt_clf
								best_success = success
								best_k = k
								best_list = k_list
								best_parameters = {'criterion': criterion,
								 'min_samples_split': min_samples_split,
								 'max_depth': max_depth}

	print "best_k:",best_k
	print "best_list:",best_list
	print "best_parameters:",best_parameters
	print "best_success:",best_success
	print "=== Finishing Decision Tree Tunning ==="
	return best_clf