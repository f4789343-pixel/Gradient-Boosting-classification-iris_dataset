import numpy as np
from gradient_boosting_iris_dataset.train import y_test, predicted_classes

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
