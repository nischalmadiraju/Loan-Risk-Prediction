import pandas as pd


def input_data():
    print('yo')
    data = pd.read_csv('loan_final313.csv')
    data.drop(columns=['home_ownership', 'income_category', 'term', 'application_type', 'purpose', 'interest_payments', 'loan_condition', 'grade','issue_d','final_d'], inplace=True)
    data_without_null = data.dropna()
    region = pd.get_dummies(data_without_null['region'], drop_first=True)
    # term = pd.get_dummies(data=data_without_null['Term'], drop_first=True)
    # home_o = pd.get_dummies(data=data_without_null['Home Ownership'], drop_first=True)
    # purpose = pd.get_dummies(data=data_without_null['Purpose'], drop_first=True)
    data_new = data_without_null.drop(columns=['region'], axis=1)
    data_train = pd.concat([data_new, region], axis=1)
    return data_train
