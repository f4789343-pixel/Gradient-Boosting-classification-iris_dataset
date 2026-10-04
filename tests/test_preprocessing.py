from gradient_boosting_iris_dataset.preprocessing import preprocessing

def test_preprosess():
   x_train, x_test, y_train, y_test, y_train_log = preprocessing()
   assert len(x_train) > 0
   assert len(x_test) > 0