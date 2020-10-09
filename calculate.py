import pandas as pd 
from sklearn.metrics import confusion_matrix

# y_true = [0 for i in range(20)] + [1 for i in range(20)]
# print(y_true)
# y_pred = pd.read_csv("pred.csv")['0'].values.tolist()
# print(y_pred)
print(confusion_matrix(y_true, y_pred))

kfold = pd.read_csv("history.csv")
print("10-KFOLD RESULTS")
print("Average accuracy:", kfold["accuracy"].mean())
print("Average f1:", kfold["f1"].mean())
print("Average precision:", kfold["precision"].mean())
print("Average recall:", kfold["recall"].mean())