import os
import requests
import sys
import pandas as pd

if __name__ == '__main__':
    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    # Download data if it is unavailable.
    if 'house_class.csv' not in os.listdir('../Data'):
        sys.stderr.write("[INFO] Dataset is loading.\n")
        url = "https://www.dropbox.com/s/7vjkrlggmvr5bc1/house_class.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/house_class.csv', 'wb').write(r.content)
        sys.stderr.write("[INFO] Loaded.\n")

    df = pd.read_csv('../Data/house_class.csv')
    info = {'rows': df.shape[0], 'columns': df.shape[1], 'NaNs': df.isna().any().any(),
            'max_rooms': df.Room.max(), 'mean_area': df.Area.mean(), 'zip_loc': df.Zip_loc.nunique()}
    print(*info.values(), sep='\n')
