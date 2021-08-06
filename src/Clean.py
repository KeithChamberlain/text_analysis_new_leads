
import numpy as np
from numpy.core.numeric import correlate
import pandas as pd
import re


def combine_vec(data, replace_na = True):
        '''
        Creates a Series the same length as the result to store the result, 
        and stores the string concatenaton of the columns after replacing 
        pd.NA and np.nan values with a space (otherwise only NA or NaN would
        result). 
        
        return:
            A Series with the result. If all columns are spaces (meaning all
            columns were NaN or NA), return the concatenated string to np.nan
            (experimental).
        '''
        s = data.shape
        d = pd.Series([""]*s[0])
        if replace_na:
            data.fillna(" ", inplace=True)
        for _ in np.arange(s[1]):
            d[:] += data.iloc[:,_] + " "
        d = d.replace(to_replace = "   |  ", value = " ", regex=True)
        return d


class Clean:
    def __init__(self, path, type_dict=None, type_other = str, columns="all", unnamed = None,
                 eng_dict=None, compression = None, sep=","):
        self.path = path
        self.type_dict = type_dict
        self.type_other = type_other
        self.columns = np.array(columns)
        self.eng_dict = eng_dict
        self.apriori_indexes = dict()
        self.compression = compression
        self.sep = sep
        self.unnamed = np.array(unnamed)
        self.read_csv()
        


    def read_csv(self):
        from pandas import read_csv
        self.df = read_csv(self.path, sep=self.sep, dtype=self.type_dict, compression=self.compression, low_memory=False)
        self.keep()
        self.rename()
        self.make_var()

    def keep(self):
        self.df = self.df.loc[:,self.columns].copy()
        return None

    def make_var(self, inverse=True):
        '''
        If key is combine, values is list of columns to combine:
            {"combine": [start_column, stop_column]}
        If key is string, values is a list: a string to search in the
            following column: {"string": [new_column_name, column_to_read, contains_value, 
                value_if_true, value_if_false]}
        '''
        for key, value in self.eng_dict.items():
            if key == "combine":
                self.df[key] = combine_vec(self.df.iloc[:,value[0]:value[1]].copy())
            if key == "string":
                self.df[value[0]] = np.array([" "] * self.df.shape[0]).astype(str)
                index = self.df.loc[:,value[1]].str.contains(value[2], 
                    regex=True, flags=re.IGNORECASE)
                self.df.loc[index, value[0]] = value[3]
                self.df.loc[~index,value[0]] = value[4]
        return None
    
    def rename(self):
        '''
        rename unnamed coluimns
        '''
        index = pd.Series(self.columns).str.contains("Unnamed")
        self.columns[index] = self.unnamed
        self.df.columns = self.columns
        return None


    def get_df(self):
        return self.df
    
    def to_time(self, columns):
        self.df['dt'] = pd.to_datetime(df[columns[0]] + " " + df[columns[1]], 
            infer_datetime_format=True)
        self.df["Month"] = self.df["dt"].dt.month
        self.df["Day of Week"] = self.df["dt"].dt.dayofweek
        self.df["Day of Month"] = self.df["dt"].dt.day
        self.df["Year"] = self.df["dt"].dt.year
        return "Done"

    def not_time(self, column, new_name):
        dt = pd.to_datetime(self.df[column], infer_datetime_format=True)
        self.df[new_name] = dt.dt.hour*60 + dt.dt.minute + dt.dt.second/60
        return "Done"


    def modify_df(self, modification):
        self.df = pd.concat([df, modification], axis=1)
        return "Done"
    
    def add_apriori_index(self, key, index):
        self.apriori_indexes[key] = index
        return "Done"

    def clean_counts(self, column, threshold, replacement):
        vc = self.df[column].value_counts()
        track = pd.DataFrame({"Name":np.array(vc.index[:]), "Value":np.array(vc[:])})
        track = track.loc[track["Value"]<threshold,:]
        GroupLowCount = track["Name"].to_numpy()
        self.df[column].replace(to_replace=GroupLowCount, 
            value=replacement, inplace=True)
        return "Done"


if __name__ == "__main__":
    save = True
    df = pd.read_csv("../Data/CTM_all_calls_export.csv", nrows=0)
    cols = df.columns
    
    columns = ["Tracking Source", "Search Query", "Referral", "Page", "Last URL", "Likelihood", 
        "Duration", "Date", "Time", "Direction", "Unnamed: 42", 
        "Unnamed: 43","Unnamed: 44","Unnamed: 45"]
    
    type_dict = {"Tracking Source": "category", "Search Query": str, "Referral": str, "Page": str, 
        "Last URL": str, "Likelihood": float, "Duration": str,
        "Date": str, "Time": str, "Direction": "category", "DF01": str, "DF02": str, "DF03": str,
        "DF04":str}
    unnamed = ["DF01", "DF02", "DF03", "DF04"]

    #print(df[columns].info())


    eng_dict = {"combine":[10,14], "string":["New_Lead","combine", "new","NewLead","NotNew"]}

    cln = Clean("../Data/CTM_all_calls_export.csv", type_dict=type_dict, type_other = "uint8", 
        columns = columns, unnamed=unnamed, 
        compression = None, eng_dict = eng_dict)
    

    eng_dict = {"string":["Hang_Voice", "combine", "hang|voice", "HangVoice", "NotHangVoice"]}

    cln.eng_dict = eng_dict

    cln.make_var()

    print(cln.get_df()["Tracking Source"].value_counts())
    print(cln.get_df()["Direction"].value_counts())

    cln.get_df()["Tracking Source"].replace(to_replace=["Google AdWords","Google Ads", 
        "Google Paid","Google Adwords"], value="Google Paid", inplace=True)
    cln.get_df()["Direction"].replace(to_replace=["outbound"], value=np.nan, inplace=True)

    cln.clean_counts("Tracking Source", 5000, "GroupLowCounts")
    cln.not_time("Duration", "dur_min")
    cln.to_time(["Date","Time"])

    # Value Counts Add Up
    print(cln.get_df()["New_Lead"].value_counts())
    print(cln.get_df()["Tracking Source"].value_counts())

    cln.df.drop(columns=["DF01","DF02","DF03","DF04", "Duration","Date","Time"], inplace=True)
    print(cln.get_df().info())

    df = cln.get_df()
    if save:
        df.to_csv("../data/data.csv", index=False, sep=",", compression="gzip")
    
    df2 = pd.read_csv("../data/cleaned1.csv", sep=",", compression="gzip", low_memory=False)

    df3 = pd.get_dummies(data=df['New_Lead'], prefix="nl", columns=["New_Lead"], drop_first=False)
    df4 = pd.get_dummies(data=df2['New_Lead'], prefix="nl2", columns=["New_Lead"], drop_first=False)
    print(df3.info())
    print(df4.info())

    print(np.corrcoef(df3["nl_NewLead"], df4["nl2_NewLead"], rowvar=False))
    print(np.corrcoef(df3["nl_NotNew"], df4["nl2_NotNewLead"], rowvar=False))
    print((df3["nl_NewLead"] == df4["nl2_NewLead"]).sum()+(df3["nl_NotNew"]==df4["nl2_NotNewLead"]).sum())
    print(df.shape, df2.shape)
