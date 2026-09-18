from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,precision_score,recall_score
import pandas as pd

iris = load_iris()

df = pd.DataFrame(iris.data,columns=iris.feature_names)
df['target'] = iris.target

X = df.drop(columns='target')
y = df['target']

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = GradientBoostingClassifier(n_estimators=10,learning_rate=0.1,max_depth=5,random_state=42)
model.fit(x_train,y_train)
predictions = model.predict(x_test)
accuracy = accuracy_score(y_test,predictions)

cm = confusion_matrix(y_test,predictions)
precision = precision_score(y_test, predictions, average='macro')
recall = recall_score(y_test, predictions, average='macro')
f1 = f1_score(y_test, predictions, average='macro')

print('scikit Accuracy:', accuracy)
print('Confusion Metrix:', cm)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)


