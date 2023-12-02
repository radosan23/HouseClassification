import os
import requests
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def download_data():
    if not os.path.exists('../Data'):
        os.mkdir('../Data')
    if 'house_class.csv' not in os.listdir('../Data'):
        sys.stderr.write("[INFO] Dataset is loading.\n")
        url = "https://www.dropbox.com/s/7vjkrlggmvr5bc1/house_class.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/house_class.csv', 'wb').write(r.content)
        sys.stderr.write("[INFO] Loaded.\n")


def one_hot_encode(*x_in):
    enc = OneHotEncoder(drop='first', sparse_output=False).fit(x_in[0][['Zip_area', 'Zip_loc', 'Room']])
    x_out = []
    for x in x_in:
        x_enc = enc.transform(x[['Zip_area', 'Zip_loc', 'Room']])
        x_out.append(x[['Area', 'Lon', 'Lat']].join(pd.DataFrame(x_enc, index=x.index,
                                                                 columns=enc.get_feature_names_out())))
    return x_out


def main():
    df = pd.read_csv('../Data/house_class.csv')
    X, y = df.drop('Price', axis=1), df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=X['Zip_loc'],
                                                        random_state=1)
    X_train_trans, X_test_trans = one_hot_encode(X_train, X_test)
    clf = DecisionTreeClassifier(criterion='entropy', max_features=3, splitter='best', max_depth=6,
                                 min_samples_split=4, random_state=3).fit(X_train_trans, y_train)
    prediction = clf.predict(X_test_trans)
    accuracy = accuracy_score(y_test, prediction)
    print(accuracy)


if __name__ == '__main__':
    main()
