from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import logging
from gradient_boosting_iris_dataset import logging_config

logger = logging.getLogger(__name__)
logger.info('Preprocessing started')

def preprocessing():
  iris = load_iris()

  df = pd.DataFrame(iris.data,columns=iris.feature_names)
  df['target'] = iris.target

  X = df.drop(columns='target')
  y = df['target']

  np.random.seed(42)

  indices = np.random.permutation(len(X))
  test_size = int(len(X)*0.2)

  train_indices = indices[test_size:]
  test_indices = indices[:test_size]

  x_train = X.iloc[train_indices]
  x_test = X.iloc[test_indices]

  y_train = y.iloc[train_indices]
  y_test = y.iloc[test_indices]

  x_train = x_train.to_numpy()
  x_test = x_test.to_numpy()

  y_train = y_train.to_numpy()
  y_test = y_test.to_numpy()
  return x_train,x_test,y_train,y_test