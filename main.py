#importing pandas library
import pandas as pd

#importing train and predict function from model.py
from model import train, predict

#import train_test_split method
from sklearn.model_selection import train_test_split

#import classification_report, confusion_matrix for calculating model's performance
from sklearn.metrics import classification_report, confusion_matrix

if __name__ == '__main__':
    # Loading CSV File
    data = pd.read_csv('original.csv')
    # Dropping rows with NaN values
    input = data.dropna()

    # Splitting Dataset into train and test
    y = input['default']
    X = input.drop(columns=['default'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Training the model
    model = train(X_train, y_train)
    # making Prediction
    pred = predict(model, X_test)
    print(pred)

    # Printing perfomance Metrics
    print(classification_report(y_test, pred))
    print('===================================')
    print(confusion_matrix(y_test, pred))
