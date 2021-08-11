import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
from sklearn import linear_model
from sklearn import metrics
from sklearn.dummy import DummyClassifier
from joblib import dump, load
from sklearn.base import clone 
from sklearn import model_selection
from sklearn.metrics import roc_curve


def feature_importance(path=None, compression=None, xlab = None, savepath=None):
    if path is not None:
        featImp = pd.read_csv(path, compression=compression)
        featImp.columns = ["Scores", "Names"]
    featImp.sort_values(by="Scores", ascending = False, inplace=True)
    # print(featImp)
    # print(np.std(featImp["Scores"]))
    fig, ax = plt.subplots(figsize=(16,7))
    ax.bar(x= featImp["Names"], height=featImp["Scores"]), #yerr=np.std(featImp["Scores"])), 
    ax.set_title("Drop One F1 Score Feature Importance", fontsize=35)
    ax.set_xlabel(xlab, fontsize=25)
    ax.set_ylabel("Change in F1 Score", fontsize=25)
    ax.set_xticklabels(labels=featImp["Names"], rotation = 90)
    plt.show()
    fig.savefig(savepath)
    return fig, ax


if __name__ == "__main__":
    save = True
    
    path_X_train_confid = "../data/X__train_confidential.csv"
    path_y_train_confid = "../data/y__train_confidential.csv"
    path_X_test_confid = "../data/X__test_confidential.csv"
    path_y_test_confid = "../data/y__test_confidential.csv"


    path1_bestmodelpart_confidential = "../data/best_model_part_confidentail.gz"
    path2_bestmodelfull_confidential = "../data/best_model_full_confidential.gz"

    X_train_confid_full = pd.read_csv(path_X_train_confid, 
        compression="gzip", low_memory=False)
    
    y_train_confid = pd.read_csv(path_y_train_confid, 
        compression="gzip", low_memory=False).to_numpy().reshape(-1,).ravel()
    X_test_confid_full = pd.read_csv(path_X_test_confid, 
        compression="gzip", low_memory=False)
    y_test_confid = pd.read_csv(path_y_test_confid, 
        compression="gzip", low_memory=False).to_numpy().reshape(-1,).ravel()
    # Drop Unused
    X_train_confid_full = X_train_confid_full.iloc[:,18:]
    X_test_confid_full = X_test_confid_full.iloc[:,18:]


    X_train_confid_part = X_train_confid_full.iloc[:,:93]
    X_test_confid_part = X_test_confid_full.iloc[:,:93]
    X_train_confid_orig = X_train_confid_full.iloc[:,93:]
    X_test_confid_orig = X_test_confid_full.iloc[:,93:]


    model = load("../data/best_model_08_08_conf_orig.gz")
    modelPart = load("../data/best_model_08_08_part_conf.gz")
    modelFull = load("../data/best_model_08_08_full_conf.gz")

    print("Data Imported")
    print("\n\n\n\n\n")
    
    
    string = '''
    Get Baseline Model
    '''
    print(string)
    model_baseline = DummyClassifier(strategy="stratified", constant=True)
    model_baseline.fit(X_test_confid_full, y_test_confid)

    string = '''
    Get ROC Curve
    ''' 
    print(string)
    roc_display = metrics.plot_roc_curve(model_baseline, X_test_confid_full, y_test_confid, 
        name="DummyClassifier; Stratified")
    roc_display = metrics.plot_roc_curve(model, X_test_confid_orig, y_test_confid, 
        ax=roc_display.ax_, name = "Random Forest without NLP Features")
    roc_display = metrics.plot_roc_curve(modelPart, X_test_confid_part, y_test_confid, 
        ax=roc_display.ax_, name = "Random Forest with NLP Features only")
    roc_display = metrics.plot_roc_curve(modelFull, X_test_confid_full, y_test_confid, 
        ax=roc_display.ax_, name="Random Forest Combined")
    roc_display.ax_.set_title("ROC Curve", fontsize=35)
    roc_display.ax_.set_xlabel("False Alarm Rate", fontsize=25)
    roc_display.ax_.set_ylabel("Sensitivity", fontsize=25)
    plt.show()
    if save:
        roc_display.figure_.savefig("../img/ROC.jpg")
    print("\n\n\n\n\n")
    
    string = '''
    GET PR Curve
    '''
    print(string)
    pr_display = metrics.plot_precision_recall_curve(model_baseline, X_test_confid_full, y_test_confid,
        name="Dummy Classifier; Stratified")
    pr_display = metrics.plot_precision_recall_curve(model, X_test_confid_orig, y_test_confid, 
        ax = pr_display.ax_, name = "Random Forest without NLP Features")
    pr_display = metrics.plot_precision_recall_curve(modelPart, X_test_confid_part, y_test_confid, 
        ax=pr_display.ax_, name = "Random Forest with NLP Features only")
    pr_display = metrics.plot_precision_recall_curve(modelFull, X_test_confid_full, y_test_confid, 
        ax=pr_display.ax_, name="Random Forest Combined")
    pr_display.ax_.set_title("Precision/Recall Curve", fontsize=35)
    pr_display.ax_.set_xlabel("Recall", fontsize=25)
    pr_display.ax_.set_ylabel("Precision", fontsize=25)
    plt.legend()
    plt.show()
    if save:
        pr_display.figure_.savefig("../img/PrecRec.jpg")
