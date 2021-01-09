from Dataset import  input_data
import pandas as pd
from model import  train,predict
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # input = input_data()
    # print(input.columns.values)
    data = pd.read_csv('original.csv')
    input = data.dropna()
    y = input['default']
    X = input.drop(columns=['default'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print('hi')
    model = train(X_train,y_train)
    print('test')
    pred = predict(model, X_test)
    print(pred)
    print(classification_report(y_test, pred))
    print('===================================')
    print(confusion_matrix(y_test, pred))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
