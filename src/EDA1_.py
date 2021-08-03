import numpy as np
from numpy.random import PCG64, Generator
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import makedv as md
from boot_ import boot_ci, boot_ci2


def solve_np_negbin(obj, loglik = "nb2", bootstrap=False, verbose=False):
    if verbose:
        print("Solving CI's for "+loglik)
        print("obj shape ", obj.shape)

    if bootstrap:
        mu = np.array(boot_ci(obj[:,0]))
        alpha = np.array(boot_ci(obj[:,1]))

    if verbose and bootstrap:
        print(mu)
        print(alpha)

    mu_exp=np.exp(mu)

    if loglik == "nb2":
        Q = 1
    elif loglik == "nb1":
        Q = 0

    size = 1. / alpha  * mu_exp**Q
    prob = size / (size + mu_exp)   

    return prob, size



def fit_distrib2(bs, fun_text=None, sum_stat = np.mean):
    rows = 1 if len(bs.shape) < 2 else bs.shape[0]
    
    obj = list()

    print("The number of rows is ", rows)
    for _ in range(rows):
        dist_to_fit = eval(fun_text)
        if _ % 500 == 0:
            print(_, " ", dist_to_fit.params)
        obj.append(dist_to_fit.params)
    
    obj = np.array(obj)
    
    obj = sum_stat(obj, axis=0) if sum_stat is not None else obj

    return obj






def clean_miss_replace(arr, delete_na = True, to_replace=None, value=None):
    if type(arr) not in [pd.core.series.Series, np.ndarray, pd.core.frame.DataFrame]:
        arr = pd.Series(arr)
    
    if delete_na:
        arr.fillna(np.nan)
    
    index = np.isnan(arr) # Test (vectorized) if is.nan()
    nomiss = arr[~index].copy()# Reverse index

    nomiss.replace(to_replace=to_replace, value=value, inplace=True)
    return nomiss







def plot_pct_bar(ax, pct, title=None, xlab=None, ylab = None, path=None):
    plt.title(title, fontsize=35)
    plt.xlabel(xlab, fontsize=25)
    plt.ylabel(ylab, fontsize=25)
    xticklabs = ax.get_xticklabels()
    ax.set_xticklabels(xticklabs, rotation=45, horizontalalignment="right")

    i = 0
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        plt.text(x+width/2, y+height*1.01,
            str(pct[i])+"%", ha="center", weight="bold", fontsize=20)
        i+=1
    plt.savefig(path)
    plt.show()

    return ax

def plot_pct_bar2(ax, pct, title=None, xlab=None, ylab = None, path=None):
    plt.title(title, fontsize=35)
    plt.xlabel(xlab, fontsize=25)
    plt.ylabel(ylab, fontsize=25)
    xticklabs = ax.get_xticklabels()
    ax.set_xticklabels(xticklabs, rotation=45, horizontalalignment="right")

    i = 0
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        plt.text(x+width/2, y+height*1.01,
            str(pct[i])+"%", ha="center", weight="bold", fontsize=10)
        i+=1
    plt.savefig(path)
    plt.show()

    return ax


def plot_pct_bar3(ax, pct, title=None, xlab=None, ylab = None, path=None, xticklabs=None):
    plt.title(title, fontsize=35)
    plt.xlabel(xlab, fontsize=25)
    plt.ylabel(ylab, fontsize=25)
    ax.set_xticks(np.arange(12))
    if not xticklabs:
        xticklabs = ax.get_xticklabels()
    ax.set_xticklabels(labels=xticklabs, rotation=45, horizontalalignment="right")

    i = 1
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        plt.text(x+width/2, y+height*1.01,
            str(pct[i])+"%", ha="center", weight="bold", fontsize=10)
        i+=1
    plt.savefig(path)
    plt.show()

    return ax

def plot_pct_bar4(ax, pct, title=None, xlab=None, ylab = None, path=None, xticklabs=None):
    plt.title(title, fontsize=35)
    plt.xlabel(xlab, fontsize=25)
    plt.ylabel(ylab, fontsize=25)
    
    if not xticklabs:
        xticklabs = ax.get_xticklabels()
    ax.set_xticks(np.arange(len(xticklabs)))
    ax.set_xticklabels(labels=xticklabs, rotation=45, horizontalalignment="right",
        fontsize=15)

    i = 0
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        plt.text(x+width/2, y+height*1.01,
            str(pct[i])+"%", ha="center", weight="bold", fontsize=10)
        i+=1
    plt.savefig(path)
    plt.show()

    return ax

if __name__ == "__main__":
    rng = Generator(PCG64(seed=None))

    df=pd.read_csv("../data/Cleaned2.csv", compression="gzip", 
        low_memory=False)

    print(df.info())

    df['dt'] = pd.to_datetime(df['dt'], errors="coerce", 
        infer_datetime_format=True)


    df["day"] = df.day.astype("category")
    df["Direction"] = df.Direction.astype("category")

    print("\n\n")

    vc = df.value_counts(["Direction","New_Lead"])
    ax = vc.plot(kind="bar", xlabel = None, figsize=(16,5))
    result_pct = vc/vc.sum()*100
    ax_pct = plot_pct_bar(ax, round(result_pct,1), title="New Lead by Direction", 
        xlab="New Lead & Direction",
        ylab = "Counts", path = "../img/DirectionLeadBar_.jpg")

    print("\n\n\n\n\n")


    print("\n\n\n\n\n")
    string = '''
    \tUNIVARIATE ANALYSES
    '''


    print(df["Direction"].value_counts())
    bar_ = pd.DataFrame({"x":df['Direction'].cat.categories, 
        "height":df["Direction"].value_counts(sort = False)})
    bar_ = bar_.sort_values(by = "height", ascending=False)
    bar_["percentage"] = np.round(bar_["height"]/bar_['height'].sum()*100, 1)
    
    md.plot_bar(x=bar_["x"], height=bar_["height"], percents=bar_["percentage"], 
        title = "Communication Direction by Itself", 
        xlab = "Communication Type/Direction",
        ylab = "Counts", path = "../img/bar3_.jpg")
    print("\n\n\n\n\n")
    index = df.loc[:,"New_Lead"] == "NewLead"
    df["new"] = df.loc[index,"New_Lead"]

    vcd = df.value_counts(["day","new"], sort = False)

    print(vcd)
    print(vcd.mean())
    ax = vcd.plot(kind="line", xlabel = None, figsize=(16,5), linewidth=7)
    ax.set_ylim(0, 8000)
    result_pct = vcd/vcd.sum()*100
    ax_pct = plot_pct_bar(ax, round(result_pct,1), title="New Lead by day of Week", 
        xlab="New Lead Only & day of Week",
        ylab = "Counts", path = "../img/dayNewLeadLine_.jpg")
    
    print(vcd.mean())
    ax = vcd.plot(kind="bar", xlabel = None, figsize=(16,5))
    ax.set_ylim(0, 8000)
    result_pct = vcd/vcd.sum()*100
    ax_pct = plot_pct_bar(ax, round(result_pct,1), title="New Lead by day of Week", 
        xlab="New Lead Only & day of Week",
        ylab = "Counts", path = "../img/dayNewLeadbar_.jpg")
    
    
    print("\n\n\n\n\n")

    index = md.get_search(df.loc[:,"New_Lead"], "NewLead")
    index = pd.Series(index)
    index.fillna(False, inplace=True)
    df["new3"] = df.loc[index,"New_Lead"]
    vcd2 = df.value_counts(["Direction", "new3"])
    ax = vcd2[:].plot(kind="bar", xlabel = None, figsize=(16,5))
    result_pct = vcd2[:]/vcd2[:].sum()*100
    ax_pct = plot_pct_bar(ax, round(result_pct,1), title="New Lead by Direction", 
        xlab="New Lead Only & Direction",
        ylab = "Counts", path = "../img/DirectionNewLeadBar_.jpg")
    print("\n\n\n\n\n")




    df['dt_dow']=df['dt'].dt.dayofweek
    df['dt_year']=df['dt'].dt.year
    df['dt_month']=df['dt'].dt.month
    df['dt_hour']=df['dt'].dt.hour
    df['dt_day']=df['dt'].dt.day
    df['dt_week']=df['dt'].dt.isocalendar().week
    for year in range(2012, 2021):
        df[year]=df['dt'].dt.year==year
   
    string='''
    The forgotten variable... Week of Month
    '''


    print(string)
    dt_week = df[["dt_week", "new3"]].groupby(by=["dt_week"]).count()

    dt_week.sort_index(inplace=True)
    ax = dt_week.plot(kind="line", xlabel=None, figsize=(16,5), legend=None, linewidth=7)
    ax.set_xlabel("New Leads & Weekly Seasonality", fontsize=25)
    ax.set_ylabel("Counts", fontsize=25)
    ax.set_title("New Leads by Week", fontsize=35)
    ax.set_ylim(0,4500)
    fig = plt.gcf()
    fig.savefig("../img/AnnualWeeklySeasonLeadLine_.jpg")
    plt.show()


    print("\n\n\n\n\n")


    dtdow = df[["dt_dow", "new3"]].groupby(by=["dt_dow"]).count()
    print(dtdow)

    dtmonth = df[["dt_month", "new3"]].groupby(by=['dt_month']).count()
    dtyear = df[["dt_year","new3"]].groupby(by=["dt_year"]).count().reset_index()
    dthour = df[["dt_hour","new3"]].groupby(by=["dt_hour"]).count()
    # for year in range(2012,2021):
    #     locals()[f'dt{year}']=df[[year, "dt_month", 'new3']].groupby(by=[year, 'dt_month']).count()
    #     locals()[f'dt{year}'].plot(kind="line", xlabel=None, legend=True)
    # plt.show()

    


    dthour.sort_index(inplace=True)
    print(dthour/dthour.sum())
    ax = dthour.plot(kind="line", xlabel=None, figsize=(16,5), legend=None, linewidth=7)
    ax.set_xlabel("New Leads & Hour Seasonality", fontsize=25)
    ax.set_ylabel("Counts", fontsize=25)
    ax.set_title("New Leads by Hour", fontsize=35)
    ax.set_ylim(0,17000)
    fig = plt.gcf()
    fig.savefig("../img/HourSeasonLeadLine_.jpg")
    plt.show()

    ax = dthour.plot(kind="bar", xlabel=None, figsize=(16,5), legend=None)
    ax.set_ylim(0,17000)
    result_pct = dthour.loc[:,"new3"]/dthour.loc[:,"new3"].sum()*100
    ax_pct = plot_pct_bar2(ax, round(result_pct,1),
        title = "New Leads by Hour", xlab = "New Leads & Hour Seasonality",
        ylab = "Counts",
        path = "../img/HourSeasonLeadbar_.jpg")


    
    print(dtmonth)
    dtmonth.sort_index(inplace=True)
    ax = dtmonth.plot(kind="line", xlabel=None, figsize=(16,5), legend=None, linewidth=7)
    ax.set_xlabel("New Leads & Month Seasonality", fontsize=25)
    ax.set_ylabel("Counts", fontsize=25)
    ax.set_title("New Leads by Month", fontsize=35)
    ax.set_ylim(0,19000)
    fig = plt.gcf()
    fig.savefig("../img/MonthSeasonLeadLine_.jpg")
    plt.show()
    

    ax = dtmonth.plot(kind="bar", xlabel=None, figsize=(16,5), legend=None)
    result_pct = dtmonth.loc[:,"new3"]/dtmonth.loc[:,"new3"].sum()*100
    ax_pct = plot_pct_bar3(ax, round(result_pct,1),
        title = "New Leads by Month", xlab = "New Leads & Month Seasonality",
        ylab = "Counts",
        path = "../img/MonthSeasonLeadbar_.jpg")

    print("\n\n\n\n\n")



    # gpmn = df[["New_Lead", "dur_min"]].groupby(by="New_Lead").aggregate([np.mean, np.std, np.count_nonzero])
    # print(gpmn)
    # threshold = 40
    # mask = gpmn["dur_min"]["count_nonzero"] > 40
    # gpmnm = gpmn.loc[mask,:]
    # print(gpmn)
    # print(gpmnm)
    # yerr=gpmnm["dur_min"]["std"]
    # ax = gpmnm["dur_min"]["mean"].plot(kind="bar", yerr=yerr, figsize=(16,5))
    # ax.set_title("New Lead by Call Duration", fontsize=35)
    # ax.set_xlabel("New Lead & Mean Call Duration", fontsize=25)
    # ax.set_ylabel("Time (min)", fontsize=25)
    # xt = ax.get_xticklabels()
    # ax.set_xticklabels(xt, rotation=45, horizontalalignment="right")
    # fig = plt.gcf()
    # fig.savefig("../img/DurationLeadBarError_.jpg", dpi=400)
    # plt.show()


    # gpmn2 = df[["dur_min", "New_Lead"]].groupby(by="New_Lead").aggregate([np.mean, np.std,  np.count_nonzero ])
    # threshold = 40
    # mask = gpmn2["dur_min"]["count_nonzero"] > 40
    # gpmn2m = gpmn2.loc[mask,:]
    # print(gpmn2)
    # print(gpmn2m)
    # yerr=gpmn2m["dur_min"]["std"]
    # ax = gpmn2m["dur_min"]["mean"].plot(kind="bar", yerr=yerr, figsize=(16,5))
    # ax.set_title("New Leads by Call Duration", fontsize=35)
    # ax.set_xlabel("New Leads & Mean Call Duration", fontsize=25)
    # ax.set_ylabel("Time (min)", fontsize=25)
    # xt = ax.get_xticklabels()
    # ax.set_xticklabels(xt, rotation=45, horizontalalignment="right")
    # plt.savefig("../img/DurationLeadsBarError_.jpg")
    # plt.show()
    
    # def get_cmap(n, name='hsv'):
    #     '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    #     RGB color; the keyword argument name must be a standard mpl colormap name.'''
    #     return plt.cm.get_cmap(name, n)

    # ax = sns.catplot(x=df["New_Lead"], y=df["Likelihood"], data=df, 
    #     estimator=np.median, kind="box", )
    # #ax.set_title("Boxplot: Likelihood by New Lead", fontsize=35)
    # fig = plt.gcf()
    # fig.savefig("../img/LikelihoodNewLeadboxplot_.jpg")
    # plt.show()

    fig, ax = plt.subplots(figsize=(16,5))
    for year in range(2016, 2022):
        color = list(rng.choice(range(256), size=3))
        ax.plot(df[["dt_year","new3","dt_month"]].loc[df["dt_year"]==year, ["new3","dt_month"]].groupby(by="dt_month").count(), 
            label=year, linewidth=7)
    ax.set_title("New Leads Annually Across All Clients", fontsize=35)
    ax.set_xlabel("Month", fontsize=25)
    ax.set_ylabel("Count of New Leads", fontsize=25)
    plt.legend()
    fig.savefig("../img/YearOverYearLine_.jpg")
    plt.show()

    fig, ax = plt.subplots(figsize=(16,5))
    for year in range(2014, 2020):
        color = list(rng.choice(range(256), size=3))
        ax.plot(df[["dt_year","new3","dt_month"]].loc[df["dt_year"]==year,
            ["new3","dt_month"]].groupby(by="dt_month").count(),
            label=year, linewidth=7)
    ax.set_title("New Leads Annually Across All Clients", fontsize=35)
    ax.set_xlabel("Month", fontsize=25)
    ax.set_ylabel("Count of New Leads", fontsize=25)
    ax.set_xlim(0,13)
    plt.legend()
    fig.savefig("../img/YearOverYearLine2_.jpg")
    plt.show()

    