import numpy as np
import pandas as pd
import re
import SplitText as st
import matplotlib.pyplot as plt


if __name__ == "__main__":
    save = True  
    df = pd.read_csv("../data/cleaned1.csv", compression="gzip")
    
    df = df.loc[df["Tracking Source"].notna(),:]
    df = df.loc[df["Direction"].notna(),:]
    df = df.loc[df["dur_min"].notna(),:]

    df2 = pd.get_dummies(df, prefix=["nl","src","day","hr","dir","mo"],prefix_sep="_", drop_first=False,
        columns=["New_Lead", "Tracking Source","Day","Hour of Day","Direction","Month"])
    print(df.info())
    print(df2.info())
    print(df2.columns)

    print(df2.var(axis=0))
    df2.drop(columns=['Search Query', 'Referral', 'Page', 'Last URL','Duration', 
        'dt','mo_12','hr_23', 'day_6', 'src_New Patient Link', 'nl_NotNewLead',
        'dir_inbound'], inplace=True)
    df2['Is2020'] = df['Is2020']
    print(df2.info())
    

    if save:
        df2.to_csv("../data/cleaned2.csv",compression="gzip", index=False)
        df2.to_excel("../data/cleaned2.xlsx", sheet_name="cleaned2")

