import numpy as np
import pandas as pd

def read_csv(path, compression="gzip", types_dict=None, types_dict_other=str, low_memory=False):
    '''
    Read_csv is a wrapper for pandas.read_csv(), assigning a compression and dtype for their
    side effects. a DataFrame is returned with any accidental "Unnamed: 0" index column dropped.
        path: file path
        compression: same as compression for pandas.read_csv()... gzip, zip, etc.
        type_dict: {"column": dtype, ...} dictionary of dtypes.
        types_dict_other: default type for types not specified in the dictionary.
        low_memory: use False if memory isn't a problem.
    return: the DataFrame object.
    '''
    from pandas import read_csv
    df=read_csv(path, compression=compression, nrows=0)
    col_names=df.columns
    print(col_names[:15]) if len(col_names) >= 15 else print(col_names)
    df=read_csv(path, compression=compression, dtype=types_dict, low_memory=low_memory)
    if "Unnamed: 0" in df.columns:
        df.drop(columns="Unnamed: 0", inplace=True)
    return df



def drop_na(df, columns):
    for column in columns:
        df = df.loc[df[column].notna(),:].copy()
    return df.copy()



if __name__ == "__main__":

    '''
    ENVIRONMENT VARIABLES
    '''
    path1 = "../data/data1protected.csv"
    path2 = "../data/data1confidential.csv"
    cleaned_path1 = "../data/data2protected.csv"
    confid_path1 = "../data/data2confidential.csv"
    save = True 
    '''
    READ DATAFILE
    '''
    df = pd.read_csv(path1, compression="gzip", sep=",", low_memory=False)
    df_ = pd.read_csv(path2, compression="gzip", sep=",", low_memory=False)

    print(df.iloc[:,:40].info())

    print(df["Direction"].value_counts())
    print(df["New_Lead"].value_counts())
    print(df["tracking_source"].value_counts())
    

    '''
    DROP NAs
    '''
    df = drop_na(df, ["tracking_source", "Direction", "dur_min"])
    df_ = drop_na(df_, ["tracking_source", "Direction", "dur_min"])

    '''
    ONE HOT ENCODE 
    '''
    df2 = pd.get_dummies(df[["New_Lead","tracking_source", "Hour","Month","Year","Direction",
        "Day of Month","Day of Week"]], 
        prefix = ["nl","src","hr","mo","yr", "dir", "day_mo", "day_wk"], prefix_sep="_", 
        drop_first=False,
        columns = ["New_Lead", "tracking_source","Hour","Month","Year", "Direction",
        "Day of Month","Day of Week"])
    
    df_2 = pd.get_dummies(df_[["New_Lead","tracking_source", "Hour","Month","Year","Direction",
        "Day of Month","Day of Week"]], 
        prefix = ["nl","src","hr","mo","yr", "dir", "day_mo", "day_wk"], prefix_sep="_", 
        drop_first=False,
        columns = ["New_Lead", "tracking_source","Hour","Month","Year", "Direction",
        "Day of Month","Day of Week"])
    
    print(df2.columns)
    '''
    DROP COLUMN
    '''
    df2.drop(columns=['nl_NotNew','src_GroupLowCounts', 'hr_23', 'mo_12', 'yr_2012',
        'dir_inbound', 'day_mo_28','day_wk_6'], inplace=True)
    
    df_2.drop(columns=['nl_NotNew','src_GroupLowCounts', 'hr_23', 'mo_12', 'yr_2012',
        'dir_inbound', 'day_mo_28','day_wk_6'], inplace=True)

    '''
    VERIFY SHAPES
    '''
    print(df2.info())
    print(df.shape[0], df2.shape[0]) 
    print(df2["nl_NewLead"].sum())
    
    '''
    CONCAT DATAFRAMES
    '''
    df2 = pd.concat([df, df2], axis=1)
    df_2 = pd.concat([df_, df_2], axis=1)
    print(df2.iloc[:,:15].info())
    
    '''
    SAVE
    '''
    if save:
        df2.to_csv(cleaned_path1, compression="gzip", index=False)
        df_2.to_csv(confid_path1, compression="gzip", index=False)

