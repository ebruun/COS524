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
            print("Reloading and Processing Data")
            self.df = self.read_data()
            self.data_processor = Preprocessor()
            self.data_processor.preprocess_df(self.df)

            self.df.to_pickle(self.pickle_path)
        else:
            print("Processed Data Loaded from Pickled File")
            self.df = pd.read_pickle(self.pickle_path)

        print(self.df.head(5))
        
    
    def read_data(self):
        print("--Reading data from", self.path)

        train_df = pd.read_csv(self.path)
        train_df.fillna("", inplace=True) # fills any NaN values with empty strings
        
        return train_df

