import numpy as np
from numpy.random import PCG64, Generator
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import makedv as md
from boot_ import boot_ci, boot_ci2
from CreateDummies import read_csv



if __name__ == "__main__":
    string = '''
    Import Data
    '''
    print(string)
    types_dict = {'Unnamed: 0': int, 'Direction': 'category', 'New_Lead': 'category', 'dt': str, 'Month': 'category',
        'tracking_source': 'category', 'year': 'category', 'month': 'category', 'day': 'category', 'hour':'category'}
    df = read_csv("../data/cleaned2.csv", compression="gzip", types_dict=types_dict, 
        types_dict_other = np.uint8, low_memory=False)
    print(df.iloc[:,:15].info())


    string = '''
    Issolate tracking_source == "Referral", 
    "paid" in tracking_source
    '''
    print(df["New_Lead"].value_counts())
    print(df['tracking_source'].value_counts())
    index1 = ((df["tracking_source"].str.contains("Referral", case=False)) & (df["New_Lead"] == "NewLead"))
    index2 = ((df["tracking_source"].str.contains("Referral", case=False)) & (df["New_Lead"] == "NotNewLead"))
    cols = df.columns[9:38]
    obj1 = df.loc[index1, cols].sum(axis=0)
    obj2 = df.loc[index2, cols].sum(axis=0)
    freq1 = obj1/obj1.sum()*100
    freq2 = obj2/obj2.sum()*100
    freq1[0] = freq1[0] - 10
    freq2[0] = freq2[0] - 10
    idx = np.array(freq1.index)
    idx[0] = "blog+10%"
    freq1.index = idx
    print(obj1.iloc[:30])
    print(obj1.shape)
    fig, ax = plt.subplots(figsize=(16,5))
    width=0.35
    ax.bar(x=np.array(range(29))-width/2, height=freq1[:29], width=width, label="New Leads")
    ax.bar(x=np.array(range(29))+width/2, height=freq2[:29], width=width, label="Not Lead")
    plt.xticks(range(29), freq1[:29].index, rotation=90)
    ax.set_title("New Leads vs Not Leads for Referrals", fontsize=30)
    ax.set_ylabel("Rel. Freq.", fontsize=20)
    ax.set_xlabel("Extended URL Words", fontsize=20)
    plt.legend()
    #fig.savefig(path)
    plt.show()



    string = '''
    Issolate tracking_source == "Organic", 
    "paid" in tracking_source
    '''
    print(df["New_Lead"].value_counts())
    print(df['tracking_source'].value_counts())
    index1 = ((df["tracking_source"].str.contains("Organic", case=False)) & (df["New_Lead"] == "NewLead"))
    index2 = ((df["tracking_source"].str.contains("Organic", case=False)) & (df["New_Lead"] == "NotNewLead"))
    cols = df.columns[9:38]
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
    ax.bar(x=np.array(range(29))-width/2, height=freq1[:29], width=width, label="New Leads")
    ax.bar(x=np.array(range(29))+width/2, height=freq2[:29], width=width, label="Not Lead")
    plt.xticks(range(29), freq1[:29].index, rotation=90)
    ax.set_title("New Leads vs Not Leads for Organic", fontsize=30)
    ax.set_ylabel("Rel. Freq.", fontsize=20)
    ax.set_xlabel("Extended URL Words", fontsize=20)
    plt.legend()
    #fig.savefig(path)
    plt.show()


    string = '''
    Issolate tracking_source == "Organic", 
    "paid" in tracking_source
    '''
    print(df["New_Lead"].value_counts())
    print(df['tracking_source'].value_counts())
    index1 = ((df["tracking_source"].str.contains("Paid", case=False)) & (df["New_Lead"] == "NewLead"))
    index2 = ((df["tracking_source"].str.contains("Paid", case=False)) & (df["New_Lead"] == "NotNewLead"))
    cols = df.columns[9:38]
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
    ax.bar(x=np.array(range(29))-width/2, height=freq1[:29], width=width, label="New Leads")
    ax.bar(x=np.array(range(29))+width/2, height=freq2[:29], width=width, label="Not Lead")
    plt.xticks(range(29), freq1[:29].index, rotation=90)
    ax.set_title("New Leads vs Not Leads for Paid", fontsize=30)
    ax.set_ylabel("Rel. Freq.", fontsize=20)
    ax.set_xlabel("Extended URL Words", fontsize=20)
    plt.legend()
    #fig.savefig(path)
    plt.show()