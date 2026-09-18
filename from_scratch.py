from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

iris = load_iris()

df = pd.DataFrame(iris.data,columns=iris.feature_names)
df['target'] = iris.target

X = df.drop(columns='target')
y = df['target']

print(X.columns)
print(y.head())
print(df.shape)

np.random.seed(42)

indices = np.random.permutation(len(X))
test_size = int(len(X)*0.2)

train_indices = indices[test_size:]
test_indices = indices[:test_size]

x_train = X.iloc[train_indices]
x_test = X.iloc[test_indices]

y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]

print(x_train.shape)
print(x_test.shape)

def gini_impurity(y_train):
    classes, counts = np.unique(y_train, return_count=True)
    probabilities = counts / len(y_train)
    gini = 1 - np.sum(probabilities**2)
    return gini

def find_thresholds(x_train):
    thresholds = []
    for feature_index in range(len(x_train[0])):
        feature_values = []
        for i in range(len(x_train)):
            feature_values.append(x_train[i][feature_index])
            feature_values.sort()
        feature_thresholds = []
        for i in range(len(feature_values)-1):
            threshold = (feature_values[i]+feature_values[i+1])/2
            feature_thresholds.append(threshold)
        thresholds.append(feature_thresholds)
    return thresholds

def split_data(x_train, y_train,feature_index, threshold):
    x_left = []
    x_right = []
    y_left = []
    y_right = []
    for i in range(len(x_train)):
        if x_train[i][feature_index] <= threshold:
            x_left.append(x_train[i])
            y_left.append(y_train[i])
        else:
            x_right.append(x_train[i])
            y_right.append(y_train[i])
    return x_left,x_right,y_left,y_right

def weighted_gini(left,right):
    total_samples = len(left)+len(right)
    left_gini = gini_impurity(left)
    right_gini = gini_impurity(right)
    return (len(left)/total_samples)*left_gini + (len(right)/total_samples)*right_gini

def best_split(left,right,thresholds):
    best_gini = float('inf')
    best_threshold = 0
    best_feature = 0
    best_x_left = None
    best_x_right = None
    best_y_left = None
    best_y_right = None
    for feature_index in range(len(x_train[0])):
        for threshold in thresholds[feature_index]:
            x_left, x_right, y_left, y_right = split_data(x_train, y_train, feature_index, threshold)
            if len(y_left) > 0 and len(y_right) > 0:
               gini_weighted = weighted_gini(y_left, y_right)
               if gini_weighted < best_gini:
                 best_gini = gini_weighted
                 best_threshold = threshold
                 best_feature = feature_index
                 best_x_left = x_left
                 best_x_right = x_right
                 best_y_left = y_left
                 best_y_right = y_right
    return best_threshold, best_feature, best_x_left, best_x_right, best_y_left, best_y_right

def is_pure(y_train):
    freq = {}
    for i in y_train:
        freq[i] = freq.get(i,0)+1
    if len(freq) == 1:
        return True
    else:
        return False

def majority_class(y_train):
    max_count = 0
    clas = 0
    classes,counts = np.unique(y_train,return_counts=True)
    for val, count in zip(classes, counts):
        if count > max_count:
            max_count = count
            clas = val
    return clas

def build_tree(x_train, y_train,depth=0,max_depth=4):
    if is_pure(y_train):
        return y_train[0]
    thresholds = find_thresholds(x_train)
    best_threshold, best_feature, best_x_left, best_x_right, best_y_left, best_y_right = best_split(x_train,y_train,thresholds)
    if best_y_left is None:
        return majority_class(y_train)
    left_subtree = build_tree(best_x_left, best_y_left, depth+1,max_depth)
    right_subtree = build_tree(best_x_right,best_y_right,depth+1,max_depth)
    return best_threshold, best_feature, left_subtree, right_subtree

def predict(sample,tree):
    if isinstance(tree, (int,np.integer)):
        return tree
    best_threshold, best_feature, left_subtree, right_subtree = tree
    if sample[best_feature] <= best_threshold:
        return left_subtree
    else:
        return right_subtree
