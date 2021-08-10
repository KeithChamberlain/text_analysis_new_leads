import numpy as np
from numpy.random import PCG64, Generator, random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import makedv as md
from boot_ import boot_ci, boot_ci2
from pandas import read_csv



if __name__ == "__main__":
    string = '''
    Import Data
    '''
    print(string)
    df =read_csv("../data/cleaned2.csv", sep=",", compression="gzip", low_memory=False, nrows=0)
    types_dict = {'Direction': 'category', 'New_Lead': 'category', 'dt': str, 'Month': 'category',
        "Day of Month": "category", "Day of Week": "category", "Year":"category", "Week": "category",
        "Hang_Voice":"category","tracking_source":"category"}
    col_names=df.columns
    types_dict.update({col: "uint8" for col in col_names if col not in types_dict}) 
    df = read_csv("../data/cleaned2.csv", compression="gzip", dtype=types_dict, low_memory=False)
    print(df.iloc[:,:32].info())


    string = '''
    Issolate tracking_source == "Referral", 
    "paid" in tracking_source
    '''
    print(df["New_Lead"].value_counts())
    print(df['tracking_source'].value_counts())
    index1 = ((df["tracking_source"].str.contains("Referral", case=False)) & (df["New_Lead"] == "NewLead"))
    index2 = ((df["tracking_source"].str.contains("Referral", case=False)) & (df["New_Lead"] == "NotNewLead"))
    cols = df.columns[10:32]
    obj1 = df.loc[index1, cols].sum(axis=0)
    obj2 = df.loc[index2, cols].sum(axis=0)
    freq1 = obj1/obj1.sum()*100
    freq2 = obj2/obj2.sum()*100
    freq1[0] = freq1[0] - 5
    freq2[0] = freq2[0] - 5
    idx = np.array(freq1.index)
    idx[0] = "blog+5%"
    freq1.index = idx
    print(obj1.iloc[:22])
    print(obj1.shape)
    fig, ax = plt.subplots(figsize=(16,5))
    width=0.35
    ax.bar(x=np.array(range(22))-width/2, height=freq1[:22], width=width, label="New Leads")
    ax.bar(x=np.array(range(22))+width/2, height=freq2[:22], width=width, label="Not Lead")
    plt.xticks(range(22), freq1[:22].index, rotation=90)
    ax.set_title("New Leads vs Not Leads for Referrals", fontsize=30)
    ax.set_ylabel("Rel. Freq.", fontsize=20)
    ax.set_xlabel("Extended URL Words", fontsize=20)
    plt.legend()
    fig.savefig("../img/WFExtReferralNLNNL.jpg")
    plt.show()



    string = '''
    Issolate tracking_source == "Organic", 
    "paid" in tracking_source
    '''
    print(df["New_Lead"].value_counts())
    print(df['tracking_source'].value_counts())
    index1 = ((df["tracking_source"].str.contains("Organic", case=False)) & (df["New_Lead"] == "NewLead"))
    index2 = ((df["tracking_source"].str.contains("Organic", case=False)) & (df["New_Lead"] == "NotNewLead"))
    cols = df.columns[10:32 ]
    obj1 = df.loc[index1, cols].sum(axis=0)
    obj2 = df.loc[index2, cols].sum(axis=0)
    freq1 = obj1/obj1.sum()*100
    freq2 = obj2/obj2.sum()*100
    freq1[0] = freq1[0] - 20
    freq2[0] = freq2[0] - 20
    idx = np.array(freq1.index)
    idx[0] = "blog+20%"
    freq1.index = idx
    print(obj1.iloc[:30])
    print(obj1.shape)
    fig, ax = plt.subplots(figsize=(16,5))
    width=0.35
    ax.bar(x=np.array(range(22))-width/2, height=freq1[:22], width=width, label="New Leads")
    ax.bar(x=np.array(range(22))+width/2, height=freq2[:22], width=width, label="Not Lead")
    plt.xticks(range(22), freq1[:22].index, rotation=90)
    ax.set_title("New Leads vs Not Leads for Organic", fontsize=30)
    ax.set_ylabel("Rel. Freq.", fontsize=20)
    ax.set_xlabel("Extended URL Words", fontsize=20)
    plt.legend()
    fig.savefig("../img/WFExtOrganicNLNNL.jpg")
    plt.show()


    string = '''
    Issolate tracking_source == "Organic", 
    "paid" in tracking_source
    '''
    print(df["New_Lead"].value_counts())
    print(df['tracking_source'].value_counts())
    index1 = ((df["tracking_source"].str.contains("Paid", case=False)) & (df["New_Lead"] == "NewLead"))
    index2 = ((df["tracking_source"].str.contains("Paid", case=False)) & (df["New_Lead"] == "NotNewLead"))
    cols = df.columns[10:32]
    obj1 = df.loc[index1, cols].sum(axis=0)
    obj2 = df.loc[index2, cols].sum(axis=0)
    freq1 = obj1/obj1.sum()*100
    freq2 = obj2/obj2.sum()*100
    # freq1[0] = freq1[0] - 20
    # freq2[0] = freq2[0] - 20
    # idx = np.array(freq1.index)
    # idx[0] = "blog+20%"
    # freq1.index = idx
    print(obj1.iloc[:30])
    print(obj1.shape)
    fig, ax = plt.subplots(figsize=(16,5))
    width=0.35
    ax.bar(x=np.array(range(22))-width/2, height=freq1[:22], width=width, label="New Leads")
    ax.bar(x=np.array(range(22))+width/2, height=freq2[:22], width=width, label="Not Lead")
    plt.xticks(range(22), freq1[:22].index, rotation=90)
    ax.set_title("New Leads vs Not Leads for Paid", fontsize=30)
    ax.set_ylabel("Rel. Freq.", fontsize=20)
    ax.set_xlabel("Extended URL Words", fontsize=20)
    plt.legend()
    fig.savefig("../img/WFExtPaidNLNNL.jpg")
    plt.show()


    index = df["New_Lead"] == "NewLead"
    index2 = df["New_Lead"] == "NotNewLead"

    arr = df['tracking_source'].value_counts()
    arr1 = df.loc[index, "tracking_source"].value_counts()
    arr2 = df.loc[index2, "tracking_source"].value_counts()
    arr.sort_index(inplace=True)
    arr1.sort_index(inplace=True)
    arr2.sort_index(inplace=True)

    print(arr)
    print(arr1)
    print(arr2)
    arr3 = pd.concat([arr,arr1,arr2], axis=1)
    print(arr3)



    