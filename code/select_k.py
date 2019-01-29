from sklearn.feature_selection import SelectKBest
import sys
sys.path.append("./tools/")
from feature_format import featureFormat, targetFeatureSplit

def get_features(data_dict, features_list, k):
    data = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data)
    # Since we will not use .transform method, and use the .score_ attribute, we don't need to pass
    #  the k as a parameter
    k_best = SelectKBest()
    k_best.fit(features, labels)
    scores = k_best.scores_
    # Remove 'poi' from features
    features_without_target = features_list[1:]
    # Sort the scores to get the k best scores
    unsorted_list = zip(features_without_target, scores)
    sorted_list = list(reversed(sorted(unsorted_list, key=lambda x: x[1])))
    # Insert 'poi' to the k best
    k_best_features = [features_list[0]] + dict(sorted_list[:k]).keys()

    return k_best_features
