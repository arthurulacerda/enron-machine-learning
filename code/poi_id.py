RANDOM_STATE = 42
TEST_SIZE = 0.2
#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
import select_k
import tunner
import plot_chart
from sklearn.preprocessing import MinMaxScaler
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = [
	# Target
	'poi',
	# Finance Attributes
	'salary',
	'deferral_payments',
	'total_payments',
	'loan_advances',
	'bonus',
	'restricted_stock_deferred',
	'deferred_income',
	'total_stock_value',
	'expenses',
	'exercised_stock_options',
	'other',
	'long_term_incentive',
	'restricted_stock',
	'director_fees',
	# Email Attributes
	# 'email_address', # not considered, because its just a string of the email, doesnt contain data to analyze.
	'to_messages',
	'from_poi_to_this_person',
	'from_messages',
	'from_this_person_to_poi',
	'shared_receipt_with_poi'
]

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Remove outliers
data_dict.pop("TOTAL", None)
data_dict.pop("LOCKHART EUGENE E", None)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", None)

### Create new feature(s)
for el in data_dict:
	if data_dict[el]['to_messages'] != 0 and data_dict[el]['to_messages'] != 'NaN':
		data_dict[el]['from_poi_rate'] = float(data_dict[el]['from_poi_to_this_person']) / data_dict[el]['to_messages']
	else:
		data_dict[el]['from_poi_rate'] = 'NaN'
	if data_dict[el]['from_messages'] != 0 and data_dict[el]['from_messages'] != 'NaN':
		data_dict[el]['to_poi_rate'] = float(data_dict[el]['from_this_person_to_poi']) / data_dict[el]['from_messages']
	else:
		data_dict[el]['to_poi_rate'] = 'NaN'

features_list.append('to_poi_rate')
features_list.append('from_poi_rate')

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)

# Tunning models according to it's best KSelection
rf_clf = tunner.tune_random_forest(my_dataset, features_list)
# from sklearn.ensemble import RandomForestClassifier
# rf_clf = RandomForestClassifier(bootstrap=False, max_features='auto', min_samples_leaf=1, min_samples_split=2, random_state=RANDOM_STATE)
# rf_list = ['poi', 'bonus', 'exercised_stock_options', 'salary', 'total_stock_value']
dt_clf = tunner.tune_decision_tree(my_dataset, features_list)
# from sklearn.tree import DecisionTreeClassifier
# dt_clf = DecisionTreeClassifier(criterion='gini', min_samples_split=10, max_depth=10, random_state=RANDOM_STATE)
# dt_list = ['poi', 'bonus', 'exercised_stock_options', 'total_stock_value']

plot_chart.k_by_clf(rf_clf, "Random Forest", dt_clf, "DecisionTree", my_dataset, features_list)
import select_k
# From the analisys on the chart (plot_chart.k_by_clf), we concluded that we should use the RandomForest model and the 3 best features.
clf = rf_clf
features_list = select_k.get_features(my_dataset, features_list, 3)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
