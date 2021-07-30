
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

def get_combined_indices(df, column1, column2, condition1, condition2):
    indx1 = np.array(df.loc[:,column1] == condition1)
    indx2 = np.array(df.loc[:,column2] == condition2)
    indx3 = indx1 & indx2
    return np.array(indx3)


def plot_bar_cat(height1, height2, width, tick_label, label1, label2,
     title, xlab, ylab, path):

    x = np.arange(len(tick_label))
    if width is None:
        width = 0.35
    fig, ax = plt.subplots(figsize=(16, 5))
    ax1 = ax.bar(x = x-width/2, height=height1, tick_label=tick_label, 
        width=width, label = label1, align="edge")
    ax2 = ax.bar(x = x+width/2, height=height2, tick_label=tick_label, 
        width=width, label = label2, align="edge")
    ax.set_ylabel(ylab, fontsize=25)
    ax.set_xlabel(xlab, fontsize=25)
    ax.set_title(title, fontsize=35)
    ax.legend(fontsize=15)
    ax.bar_label(ax1, padding=3)
    ax.bar_label(ax2, padding=3)
    plt.show()
    fig.savefig(path)
    return fig, ax1, ax2




def plot_bar(x, height, percents, title, xlab, ylab, path):
    fig, ax = plt.subplots(figsize=(16,5))
    ax.bar(x = x, height=height)
    ax.set_title(title, fontsize=35)
    ax.set_xlabel(xlab, fontsize=25)
    ax.set_ylabel(ylab, fontsize=25)
    i = 0
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        plt.text(x+width/2, y+height*1.01,
            str(percents[i])+"%", ha="center", weight="bold", fontsize=20)
        i+=1
    plt.show()
    fig.savefig(path)
    return fig, ax


def plot_hist(x, bins, title, xlab, ylab, path, xrange = [-1, 20], density = False):
    fig, ax = plt.subplots(figsize=(16,5))
    ax.hist(x, bins=bins, density = density)
    ax.axvline(x.mean(), color='blue', linewidth=2)
    ax.set_title(title, fontsize=35)
    ax.set_xlabel(xlab, fontsize=25)
    ax.set_ylabel(ylab, fontsize=25)
    ax.set_xlim(xrange)
    plt.show()
    fig.savefig(path)
    return fig, ax



def plot_hist2(x, bins, title, xlab, ylab, path, density = False):
    fig, ax = plt.subplots(figsize=(16,5))
    ax.hist(x, bins=bins, density = density)
    ax.set_title(title, fontsize=35)
    ax.set_xlabel(xlab, fontsize=25)
    ax.set_ylabel(ylab, fontsize=25)
    plt.show()
    fig.savefig(path)
    return fig, ax




def get_query(url_, delimiters):
    start_ = -1
    stop_ = -1
    result = ""
    for indx in range(len(delimiters[0])):
        start_ = url_.find(delimiters[0][indx], )
        if start_ == -1:
            continue
        else:
            stop_ = url_.find(delimiters[1][indx])
            if stop_ == -1:
                result = url_[len(delimiters[0][indx])+start_, len(url_)]
            else:
                result = url_[start_+len(delimiters[0][indx]): stop_]
    return result
    
def get_unique(obj, start, stop):
    accum = list()
    for col in range(start, stop):
        accum.extend(obj.iloc[:,col].unique())
    accum = sorted(pd.Series(accum).unique())
    return accum

def get_search(obj, search_string = None):
    result = np.zeros(len(obj))
    if search_string:
        result = obj.str.contains(search_string, regex=True, flags=re.IGNORECASE)
    return result

def limit_result_by(obj1, obj2):
    obj1.loc[obj2] = False
    return obj1


def split(delimiters, string, maxsplit=0):
    regexPattern = '|'.join(map(re.escape, delimiters))
    return re.split(regexPattern, string, maxsplit)

    

if __name__ == "__main__":
    save = True 

    string = '''
    START --- IMPORT RAW DATA
    '''
    print(string)
    print("\n\n")
    df = pd.read_csv("../data/CTM_all_calls_export.csv", low_memory=False)
    print("Data Dimensions: ~", df.shape)
    print("\n\n\n\n\n")
    # Print Initial Info
    print(df.info())
    print("\n\n\n\n\n\n")
    

     
    string = '''
    PRUNE FILE FROM UNUSABLE COLUMNS 
    '''
    cols = np.array(df.columns)
    cols[42]="DF01"
    cols[43]="DF02"
    cols[44]="DF03"
    cols[45]="DF04"
    df.columns = cols
    keep_cols = ["Tracking Source", "Search Query", "Referral", "Page", "Last URL", "Likelihood", 
        "Duration", "Day", "Hour of Day", "Date", "Time","Direction", "DF01", "DF02","DF03","DF04"]
    print(cols)
    print("\n\n\n\n\n")
    df=df.loc[:,keep_cols].copy()
    print(df.info())
    print("\n\n\n\n\n")

    string = '''
    CONCAT STRING COLUMN
    '''
    print(string)
    df['str_column'] = st.combine_vec(df.iloc[:,12:].copy())

    

    
    string = '''
    \t\tGET INDEXES
    '''
    print(string)
    ser = np.array([" "] * df.shape[0])
    n_e = pd.DataFrame({"New_Lead":ser}, dtype=str)
    print("\n\n")
    
    string = '''
    \t\t\tGET new VS anything else INTO "Lead" COLUMN (Per company definition)
    '''
    print(string)
    n_index = get_search(df['str_column'], "new")
    
    n_e.loc[n_index,"New_Lead"] = "NewLead"
    n_e.loc[~n_index, "New_Lead"] = "NotNewLead"
    n_e.replace(to_replace = " ", value = np.nan, inplace = True)
    vcne =  n_e["New_Lead"].value_counts(dropna=False)
    print("THIS IS THE COUNTS OF 'NewLead':")
    print(vcne, " with a length of ", len(n_e["New_Lead"]))
    print("THIS IS THE SUM OF THE VALUE COUNTS: ")
    print(vcne.sum())
    print("\n\n\n\n\n")
    df["New_Lead"]=n_e["New_Lead"]



    print("\n\n\n\n\n")
    string = '''
    \tGETTING DATE AND DURATION COLUMNS
    '''

    print(string)
    print(df["Time"][:10])
    print(df["Duration"][:10])
    strstamp = df["Date"] + " " + df["Time"]
    dt = pd.to_datetime(strstamp, errors="coerce", 
        infer_datetime_format=True)
    print(dt[:10])
    
    dt2_ = pd.to_datetime(df["Duration"], errors="coerce", 
        infer_datetime_format=True, format = "%H:%M:%S")
    dt2 = dt2_.dt.hour*60 + dt2_.dt.minute + dt2_.dt.second/60
    dt3 = dt.dt.month
    dt4 = dt.dt.year
    print(dt2[:10])
    df["dt"] = dt
    df["dur_min"] = dt2
    df["Month"] = dt3
    df["Is2020"] = 1*(dt4==2020)+0*(dt4!=2020)
    df["Day"].replace(to_replace=["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"],
        value=[0,1,2,3,4,5,6], inplace=True)

    
    df["New_Lead"] = df.New_Lead.astype("category")
    df["Hour of Day"] = df["Hour of Day"].astype("category")
    df["Month"] = df["Month"].astype("category")
    # df["Is2020"] = df["Is2020"].astype("category")
    df["Day"] = df.Day.astype("category")
    df["Direction"] = df.Direction.astype("category")
    df["Tracking Source"]= df["Tracking Source"].astype("category")
    df["Search Query"]=df["Search Query"].astype(str)
    # df.reset_index(drop=True, inplace=True)
    df.drop(columns=["DF01","DF02","DF03","DF04","str_column","Time","Date"], 
        inplace=True)

    print("\n\n\n\n\n")
    string = '''
    Count Classes for Tracking Source
    '''
    print(string)
    track = df["Tracking Source"].value_counts(dropna=False)
    print(track[:])
    track = pd.DataFrame({"Name":np.array(track.index[:]), "Value":np.array(track[:])})
    track = track.loc[track["Value"]<50,:]
    GroupLowCount = track["Name"].to_numpy()
    print(GroupLowCount)
    
    df["Tracking Source"].replace(to_replace=GroupLowCount, 
        value="GroupLowCounts", inplace=True)
    df["Tracking Source"].replace(to_replace="Google AdWords", 
        value="Google Adwords", inplace=True)

    print(df["Tracking Source"].value_counts())
    
    
    print("\n\n\n\n\n")
    string='''
    Count Classes for Direction
    '''
    print(string)

    print(df["Direction"].value_counts(dropna=False))
    df["Direction"].replace(to_replace=["outbound"], value=np.nan, inplace=True)
    

    print(df.info())
    if save:
        df.to_csv("../data/cleaned1.csv", index=False, compression="gzip")
        df.to_excel("../data/cleaned1.xlsx", sheet_name="cleaned1")