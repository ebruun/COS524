import pandas as pd
import os

#Local Imports
from dataformat.preprocessing import Preprocessor

class Data():
    def __init__(self, folder, filename, reload = True):
        print("DATA OBJECT CREATED")

        self.path = os.path.join(".", folder,filename+".csv")
        self.pickle_path = os.path.join(".",folder,"pickle_"+filename+".pkl")

        if reload:
            print("--reloading and processing data")
            raw_df = self.read_data()

            self.data_processor = Preprocessor()
            self.processed_df = self.data_processor.preprocess_df(raw_df)

            self.processed_df.to_pickle(self.pickle_path)
        else:
            print("--processed data loaded from pickled file")
            self.processed_df = pd.read_pickle(self.pickle_path)
    
    def read_data(self):
        print("--reading data from", self.path)

        new_df = pd.read_csv(self.path)
        new_df.fillna("", inplace=True) # fills any NaN values with empty strings
        
        return new_df
