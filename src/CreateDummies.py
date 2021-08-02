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
    types_dict.update({col: types_dict_other for col in col_names if col not in types_dict}) 
    df=read_csv(path, compression=compression, dtype=types_dict, low_memory=low_memory)
    if "Unnamed: 0" in df.columns:
        df.drop(columns="Unnamed: 0", inplace=True)
    return df



if __name__ == "__main__":
    '''
    ENVIRONMENT VARIABLE
    '''
    save = False 
    '''
    READ DATAFILE
    '''
    types_dict = {'Unnamed: 0': int, 'Direction': 'category', 'New_Lead': 'category', 'dt': str, 'Month': 'category',
        'tracking_source': 'category', 'year': 'category', 'month': 'category', 'day': 'category', 'hour':'category'}
    df = read_csv("../data/cleaned2.csv", compression="gzip", types_dict=types_dict, 
        types_dict_other = np.uint8, low_memory=False)
    '''
    DROP NAs
    '''
    df = df.loc[df["tracking_source"].notna(),:].copy()
    df = df.loc[df["Direction"].notna(),:].copy()
    #df = df.loc[df["dur_min"].notna(),:]

    '''
    ONE HOT ENCODE 
    '''
    df2 = pd.get_dummies(df[["New_Lead","tracking_source","day","hour","month","year","Direction"]], 
        prefix=["nl","src","day","hr","mo","yr", "dir"],prefix_sep="_", drop_first=False,
        columns=["New_Lead", "tracking_source","day","hour","month","year", "Direction"])
    '''
    DROP COLUMN
    '''
    df2.drop(columns=['nl_NotNewLead','src_New Patient Link', 'day_6', 'hr_23', 'mo_12', 
        'dir_inbound', 'yr_2012', 'dir_form'], inplace=True)
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
    print(df2.iloc[:,:15].info())
    '''
    DROP SUPERFLOUS COLUMNS
    '''
    df2.drop(columns=["Direction","New_Lead","dt","tracking_source","year","month","day","hour"], inplace=True)
    '''
    SAVE
    '''
    if save:
        df2.to_csv("../data/cleaned3.csv", compression="gzip", index=False)

