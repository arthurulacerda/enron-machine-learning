RANDOM_STATE = 42
TEST_SIZE = 0.2
import select_k
import matplotlib.pyplot as plt 
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

def k_by_clf(clf_1, name_clf_1, clf_2, name_clf_2, my_dataset, features_list):
	scores_clf_1 = []
	scores_clf_2 = []
	k_size = range(1, len(features_list))

	for k in k_size:
		k_list = select_k.get_features(my_dataset, features_list, k)

		data = featureFormat(my_dataset, k_list)
		labels, features = targetFeatureSplit(data)

		features_train, features_test, labels_train, labels_test = [],[],[],[]

		number_of_samples = 100
		for i in range(0,number_of_samples):
			features_train_sample, features_test_sample, labels_train_sample, labels_test_sample = \
			    train_test_split(features, labels, test_size=TEST_SIZE, random_state=(RANDOM_STATE + 2*i))
			features_train.append(features_train_sample)
			features_test.append(features_test_sample)
			labels_train.append(labels_train_sample)
			labels_test.append(labels_test_sample)

		sum_clf_1 = 0
		sum_clf_2 = 0
		for i in range(0,number_of_samples):
			clf_1.fit(features_train[i], labels_train[i])
			labels_predict_1 = clf_1.predict(features_test[i])
			sum_clf_1 += metrics.f1_score(labels_test[i], labels_predict_1)

			clf_2.fit(features_train[i], labels_train[i])
			labels_predict_2 = clf_2.predict(features_test[i])
			sum_clf_2 += metrics.f1_score(labels_test[i], labels_predict_2)

		scores_clf_1.append(sum_clf_1 / number_of_samples)
		scores_clf_2.append(sum_clf_2 / number_of_samples)

	plt.plot(k_size, scores_clf_1, label= name_clf_1, color="blue") 
	plt.plot(k_size, scores_clf_2, label= name_clf_2, color="red") 
	plt.xlabel("tamanho de K")
	plt.xticks(k_size) 
	plt.ylabel("f1_score") 
	plt.legend() 
	plt.show()
