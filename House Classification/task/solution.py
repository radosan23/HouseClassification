import os
import requests
import sys
import pandas as pd
from sklearn.model_selection import train_test_split


def download_data():
    if not os.path.exists('../Data'):
        os.mkdir('../Data')
    if 'house_class.csv' not in os.listdir('../Data'):
        sys.stderr.write("[INFO] Dataset is loading.\n")
        url = "https://www.dropbox.com/s/7vjkrlggmvr5bc1/house_class.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/house_class.csv', 'wb').write(r.content)
        sys.stderr.write("[INFO] Loaded.\n")


def main():
    df = pd.read_csv('../Data/house_class.csv')
    X = df.drop('Price', axis=1)
    y = df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=X['Zip_loc'],
                                                        random_state=1)
    print(X_train['Zip_loc'].value_counts().to_dict())


if __name__ == '__main__':
    main()
