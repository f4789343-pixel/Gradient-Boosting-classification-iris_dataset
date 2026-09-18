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

x_train = x_train.to_numpy()
x_test = x_test.to_numpy()

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

def gini_impurity(y_train):
    classes, counts = np.unique(y_train, return_counts=True)
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

def best_split(x_train,y_train,thresholds):
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
    if len(y_train) == 1 or depth >= max_depth:
        return np.mean(y_train)
    thresholds = find_thresholds(x_train)
    best_threshold, best_feature, best_x_left, best_x_right, best_y_left, best_y_right = best_split(x_train,y_train,thresholds)
    if best_y_left is None:
        return majority_class(y_train)
    left_subtree = build_tree(best_x_left, best_y_left, depth+1,max_depth)
    right_subtree = build_tree(best_x_right,best_y_right,depth+1,max_depth)
    return best_threshold, best_feature, left_subtree, right_subtree

def predict(sample,tree):
    if isinstance(tree, (float,np.floating)):
        return tree
    else:
     best_threshold, best_feature, left_subtree, right_subtree = tree
     if sample[best_feature] <= best_threshold:
        return predict(sample,left_subtree)
     else:
        return predict(sample,right_subtree)

def prediction(x_train,tree):
    predictions = []
    for sample in x_train:
        predictions.append(predict(sample,tree))
    return predictions


lr = 0.1
def gradient_boosting(x_train,y_train):
    classes = np.unique(y_train)
    pred = np.zeros((len(x_train),3))
    trees = []
    y_one_hot = np.eye(len(classes))[y_train]
    for _ in range(10):
        scores = pred - np.max(pred, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        prob = exp_scores / np.sum(exp_scores, axis=1,keepdims=True)
        iteration_trees = []
        for class_index in range(len(classes)):
          error = y_one_hot[:,class_index] - prob[:,class_index]
          tree = build_tree(x_train,error)
          tree_prediction = np.array(prediction(x_train,tree))
          pred[:, class_index] += lr*tree_prediction
          iteration_trees.append(tree)
        trees.append(iteration_trees)
    return trees
print(gradient_boosting(x_train,y_train))

def test_pred(x_train,y_train,x_test):
    classes = np.unique(y_train)
    trees = gradient_boosting(x_train,y_train)
    pred = np.zeros((len(x_test),3))
    for class_index in range(len(classes)):
      for i_tree in trees:
        tree = i_tree[class_index]
        tree_prediction = np.array(prediction(x_test,tree))
        pred[:,class_index] += lr*tree_prediction
    scores = pred - np.max(pred, axis=1,keepdims=True)
    exp_scores = np.exp(scores)
    prob = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return prob
prob = test_pred(x_train,y_train,x_test)
predicted_classes = np.argmax(prob, axis=1)
print('predictions', predicted_classes)

def accuracy(y_test, predictions):

    correct = 0

    for i in range(len(y_test)):
        if y_test[i] == predictions[i]:
            correct += 1

    return correct / len(y_test)

print("Accuracy:", accuracy(y_test, predicted_classes)*100)

def confusion_matrix(y_test, predictions, classes):
    matrix = np.zeros((len(classes), len(classes)),dtype=int)

    for actual, predicted in zip(y_test,predictions):
        actual_index = np.where(classes == actual)[0][0]
        predicted_index = np.where(classes == predicted)[0][0]

        matrix[actual_index][predicted_index] += 1

    return matrix
classes = np.unique(y_test)

cm = confusion_matrix(y_test, predicted_classes, classes)
print('confusion Matrix:',cm)
def confusion_values(y_test, predictions, target):
    tp,tn,fp,fn = 0,0,0,0
    for actual,predicted in zip(y_test,predicted_classes):
     if actual == target and predicted == target:
        tp += 1
     elif actual != target and predicted != target:
        tn += 1
     elif actual != target and predicted == target:
        fp += 1
     else:
        fn += 1

    return tp,tn,fp,fn

for target in classes:
    tp,tn,fp,fn = confusion_values(y_test, predicted_classes, target)
print('TP:',tp,'TN:',tn,'FP:',fp,'FN:',fn)

if tp+fp == 0:
    precision = 0
else:
    precision = tp/(tp+fp)

if tp+fn == 0:
    recall = 0
else:
    recall = tp/(tp+fn)

if precision + recall == 0:
    f1 = 0
else:
    f1 = 2 * (precision*recall)/(precision + recall)

print('Precision:', precision)
print('Recall:', recall)
print('F1:',f1)
