import numpy as np
from numpy.random import Generator, PCG64
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
from sklearn import linear_model
from sklearn.dummy import DummyClassifier
from sklearn import metrics
from sklearn import ensemble
from sklearn import tree
from sklearn import naive_bayes
from joblib import dump, load
from sklearn import model_selection
from sklearn.base import clone 
import seaborn as sns
import timeit as time
import shap


def drop_col_feat_imp(model, X_train, y_train, X_test, y_test, random_state = 42):
    # https://towardsdatascience.com/explaining-feature-importance-by-example-of-a-random-forest-d9166011959e
    # Eryk Lewinson
    # Feb 11, 2019


    # clone the model to have the exact same specification as the one initially trained
    model_clone = clone(model)
    # set random_state for comparability
    model_clone.random_state = random_state

    # training and scoring the benchmark model
    model_clone.fit(X_train, y_train)
    y_pred = model_clone.predict(X_test)
    benchmark_score = metrics.f1_score(y_true=y_test, y_pred=y_pred)
    # list for storing feature importances
    importances = []
    
    # iterating over all columns and storing feature importance (difference between benchmark and new model)
    for col in X_train.columns:
        model_clone = clone(model)
        model_clone.random_state = random_state
        model_clone.fit(X_train.drop(col, axis = 1), y_train)
        y_pred = model_clone.predict(X_test.drop(col, axis=1))
        drop_col_score = metrics.f1_score(y_true=y_test, y_pred=y_pred)
        importances.append(benchmark_score - drop_col_score)
    
    importances_df = pd.DataFrame(X_train.columns, importances)
    return importances_df




def plot_bar(x, height, percents, title, xlab, ylab, path, bar_font_size=20, bar_rotation=0):
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
        boo = 1.01 if bar_rotation == 0 else 1.1
        plt.text(x+width/2, y+height * boo,
            str(percents[i])+"%", ha="center", weight="bold", fontsize=bar_font_size, 
                rotation=bar_rotation)
        i+=1
    plt.xticks(rotation=90)
    plt.show()
    fig.savefig(path)
    return fig, ax


class FitModels:
    def __init__(self, X_train, y_train, X_test, y_test, models_list, hyperparams_list, model_name_list,
        paths_list, scoring = "f1", n_jobs=-1, verbose=2):
        '''
        X_train/test, y_train/test are the usual suspects, can be a DataFrame or an np.arrray
        model list: a list of model functions.
        hyperparams_list: a list of dict() that contain the hyperparameter settings
            for each model if not using the out of the box settings.
        paths_list: list of paths as follows...
            [0] - model scores table path
            [1] - best model path
            [2] - feature importance path
            [3] - roc plot path
            [4] - P/R plot path
        [scoring, n_jobs, verbose] = other parameters to pass to cross_val_score()
        '''
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.models_list = models_list 
        self.hyperparams_list = hyperparams_list 
        self.model_name_list = model_name_list
        self.paths_list = paths_list
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random = dict()
        self.random_sel = dict()
        self.define_models()
        self.fit_models()
        self.print_cvscores()
        self.save_cvscores()




    def fit_models(self):
        '''
        '''
        i = 0
        self.cvscores = list()
        for model in self.train_models.values():
            self.cvscores.append([self.model_name_list[i], 
                model_selection.cross_val_score(model, self.X_train, self.y_train, 
                cv=5, scoring=self.scoring, n_jobs=self.n_jobs, verbose=self.verbose)])
            i +=1
        return None


    def print_cvscores(self):
        '''
        '''
        i = 0
        for row in self.cvscores:
            print(i, " ", row[0], "\t", np.sum(row[1])/len(row[1]), " ", np.std(row[1]))
            i += 1
        return None


    def run_random(self, model_name, model, model_parameters, hyperparam_ranges, 
        scoring = ["f1", "roc_auc", "balanced_accuracy"], refit = "f1"):
        '''
        '''
        self.random[model_name] = model(**model_parameters)
        self.random_sel[model_name] = model_selection.RandomizedSearchCV(estimator = \
            self.random[model_name], 
            param_distributions = hyperparam_ranges, n_iter = 100, cv = 5, verbose=2, 
            scoring = scoring, refit = refit, return_train_score = True)
        self.random_sel[model_name].fit(self.X_train, self.y_train)
        print("\n\n\n")
        print(model_name, " best parameters ", self.random_sel[model_name].best_params_)
        print(model_name, " best score ", self.random_sel[model_name].best_score_)
        print("\n\n\n")
        return None

    def run_full(self, model, hyperparams, 
        score=metrics.f1_score, score_params={"labels":None,"pos_label":1}):
        '''
        Fit the full model, given a sscoring metric and hyperhparameters dictionary.
        Then save the model for later.
        '''
        self.model = model(**hyperparams)
        self.model.fit(self.X_train, self.y_train)
        self.save_best()

        scored = self.test(score, score_params)
        return print(f"{model} score is ", scored)
        
    def run_grid(self, model_name, model, model_parameters, hyperparam_ranges):
        pass

    def define_models(self):
        self.train_models = dict()
        i = 0
        for model in self.models_list:
            self.train_models[f'model_{i}_'] = model(**self.hyperparams_list[i])
            i+=1
        return None

    def save_best(self):
        dump(self.model, self.paths_list[1], compress=True)

    def save_cvscores(self):
        index = list()
        arr = list()
        for row in self.cvscores:
            index.append(row[0])
            arr.append(row[1])
        index = np.array(index)
        arr = np.array(arr)
        df = pd.DataFrame(index=index, data=arr, columns=np.array(["cv1","cv2","cv3","cv4","cv5"]))
        df.to_csv(self.paths_list[0], index=True)

    def get_cvscores(self):
        return self.cvscores
    
    def read_cvscores(self):
        model_scores = pd.read_csv(self.paths_list[0])
        return model_scores

    def test(self, score, score_params):
        y_pred = self.model.predict(self.X_test)
        y_true = self.y_test
        scored = score(y_true=y_true, y_pred=y_pred, **score_params)
        return scored


if __name__ == "__main__":
    plot_class = False
    save_models= True
    save_FIDropOne= True


    path1_bestmodelpart_protected = "../data/best_model_part_protected.gz"
    path1_bestmodelpart_confidential = "../data/best_model_part_confidentail.gz"
    path2_bestmodelfull_protected = "../data/best_model_full_protected.gz"
    path2_bestmodelfull_confidential = "../data/best_model_full_confidential.gz"

    path_X_train_protected = "../data/X_train_protected.csv"
    path_y_train_protected = "../data/y_train_protected.csv"
    path_X_train_confid = "../data/X__train_confidential.csv"
    path_y_train_confid = "../data/y__train_confidential.csv"
    path_X_test_protected = "../data/X_test_protected.csv"
    path_y_test_protected = "../data/y_test_protected.csv"
    path_X_test_confid = "../data/X__test_confidential.csv"
    path_y_test_confid = "../data/y__test_confidential.csv"

    run_part = False
    run_part_random = False
    run_part_grid = False
    run_full = False
    run_full_random = False
    run_full_grid = False

    run_protected = False
    run_confidential = False
    run_train = False
    run_test = False
    run_orig_protect = False
    run_orig_confid  = False
    run_shap_orig_protect = False
    run_shap_orig_confid = False
    run_shap_prot_part = False
    run_shap_conf_part = False
    run_shap_prot_full = False
    run_shap_conf_full = False
    plot = True
    

    string = '''
    READ DATA
    '''
    print(string)

    X_train_protected_full = pd.read_csv(path_X_train_protected, 
        compression="gzip", low_memory=False)

    y_train_protected = pd.read_csv(path_y_train_protected, 
        compression="gzip", low_memory=False).to_numpy().reshape(-1,).ravel()
    X_test_protected_full = pd.read_csv(path_X_test_protected, 
        compression="gzip", low_memory=False)
    y_test_protected = pd.read_csv(path_y_test_protected, 
        compression="gzip", low_memory=False).to_numpy().reshape(-1,).ravel()
    
    
    # Drop unused
    X_train_protected_full = X_train_protected_full.iloc[:,18:]
    X_test_protected_full = X_test_protected_full.iloc[:,18:]
    

    X_train_protected_part = X_train_protected_full.iloc[:,:93]
    X_test_protected_part = X_test_protected_full.iloc[:,:93]
    X_train_protected_orig = X_train_protected_full.iloc[:,93:]
    X_test_protected_orig = X_test_protected_full.iloc[:,93:]

    print(X_train_protected_orig.iloc[:,:50].info()) 
    print(X_train_protected_orig.iloc[:,50:100].info())   
    
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


    string = '''
    IF FIT MODELS
    '''
    print(string)
    models_list = [linear_model.LogisticRegression,
        ensemble.RandomForestClassifier,
        ensemble.GradientBoostingClassifier,
        tree.DecisionTreeClassifier,
        naive_bayes.ComplementNB,
        DummyClassifier]
    model_name_list = ["Logistic", "RandomForestBest", "GradientBoostingBest", "DecisionTreeBest", 
        "NaiveBayesBest", "DummyClassifier"]
    hyperparams_list = [{"class_weight":"balanced", "max_iter":2000, "warm_start": True, "n_jobs":-1},
        {"class_weight":'balanced', "oob_score":True, "n_jobs":-1, "n_estimators":90, "max_depth":40, 
            "min_samples_split":2, "min_samples_leaf":2, "max_features":'sqrt'},
        {"n_estimators": 500, "min_samples_leaf":2, "max_depth":40, "n_iter_no_change":2,
            "learning_rate":0.1, "subsample":0.8, "verbose":2},
        {'min_samples_split': 2, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 40,
            'class_weight':"balanced"},
         {'norm': False, 'fit_prior': True, 'alpha': 0.6},
        {"strategy":"stratified","constant":True}]
    paths_list = ["../data/model_scores_08_08_full_protect.csv",
                  "../data/best_model_08_08_full_protect.gz"]

    if run_train:
        if run_protected:
            string = '''
                RUN PROTECTED
                '''
            print(string)
            string = '''
                \t RUN FULL MODELS
                '''
            print(string)
            cls_prot_full = FitModels(X_train = X_train_protected_full, y_train = y_train_protected, 
                X_test = X_test_protected_full, y_test = y_test_protected, paths_list=paths_list,
                models_list=models_list, model_name_list=model_name_list, 
                hyperparams_list = hyperparams_list)
            if run_full:
                string = '''
                    \t\t RUN FULL BEST
                    '''
                print(string)
                cls_prot_full.run_full(ensemble.RandomForestClassifier, 
                    {"class_weight":'balanced', "oob_score":True, "n_jobs":-1, 
                    "n_estimators":90, "max_depth":40, "min_samples_split":2, 
                    "min_samples_leaf":2, "max_features":'sqrt'})
                '''
                <class 'sklearn.ensemble._forest.RandomForestClassifier'> score is  0.4937348045633065
                '''
            if run_full_random:
                string = '''
                    \t\t RUN PROTECTED FULL RANDOM GRADIENTS
                    '''
                print(string)
                hyperparam_ranges = {"alpha": [0.2,.4,.6,.8], "fit_prior": [True, False],
                   "norm":[True,False]}
                cls_prot_full.run_random("NaiveBayes", naive_bayes.ComplementNB, 
                   model_parameters={"class_prior":None}, hyperparam_ranges=hyperparam_ranges)
                '''
                NaiveBayes  best parameters  {'norm': False, 'fit_prior': True, 'alpha': 0.6}
                NaiveBayes  best score  0.4011599164003304
                '''
                hyperparam_ranges = {"max_depth":[40,50,100], "min_samples_split":[2,3,4,5], 
                   "min_samples_leaf":[2,3,4], "max_features":["sqrt","log2"]}
                cls_prot_full.run_random("DecisionTree", tree.DecisionTreeClassifier,
                   model_parameters={"class_weight":"balanced"}, hyperparam_ranges=hyperparam_ranges)
                '''
                {'min_samples_split': 2, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 40}
                DecisionTree  best score  0.436027523048325
                '''
            if run_part:
                string = '''
                    \t RUN PARTIAL MODEL
                    '''
                print(string)

                paths_list = ["../data/model_scores_08_08_part_protect.csv",
                    "../data/best_model_08_08_part_protect.gz"]
                cls_prot_part = FitModels(X_train = X_train_protected_part, y_train = y_train_protected, 
                    X_test = X_test_protected_part, y_test = y_test_protected, paths_list=paths_list,
                    models_list=models_list, model_name_list=model_name_list, 
                    hyperparams_list = hyperparams_list)
                    
            if run_full:
                string = '''
                    \t\t RUN PARTIAL BEST
                    '''
                print(string)
                cls_prot_part.run_full(ensemble.RandomForestClassifier, 
                    {"class_weight":'balanced', "oob_score":True, "n_jobs":-1, 
                    "n_estimators":90, "max_depth":40, "min_samples_split":2, 
                    "min_samples_leaf":2, "max_features":'sqrt'})
                '''
                <class 'sklearn.ensemble._forest.RandomForestClassifier'> score is  0.4370005821029793
                '''
        if run_confidential:
            string = '''
                \n\n\n\n\n
                RUN CONFIDENTIAL MODELS
                '''
            print(string)
            paths_list = ["../data/model_scores_08_08_conf_full.csv",
                  "../data/best_model_08_08_full_conf.gz"]
            cls_conf_full = FitModels(X_train = X_train_confid_full, y_train = y_train_confid, 
                X_test = X_test_confid_full, y_test = y_test_confid, paths_list=paths_list,
                models_list=models_list, model_name_list=model_name_list, 
                hyperparams_list = hyperparams_list)

            #fit_models()
            if run_full:
                string = '''
                    \t RUN FULL BEST CONFIDENTIAL
                    '''
                print(string)

                cls_conf_full.run_full(ensemble.RandomForestClassifier, 
                    {"class_weight":'balanced', "oob_score":True, "n_jobs":-1, 
                    "n_estimators":90, "max_depth":40, "min_samples_split":2, 
                    "min_samples_leaf":2, "max_features":'sqrt'})
                '''
                <class 'sklearn.ensemble._forest.RandomForestClassifier'> score is  0.5348408504362787
                '''
            if run_part:
                string = '''
                    \t\t RUN PARTIAL MODELS CONFIDENTIAL 
                    '''
                print(string)
                paths_list = ["../data/model_scores_08_08_conf_part.csv",
                  "../data/best_model_08_08_part_conf.gz"]
                cls_conf_part = FitModels(X_train = X_train_confid_part, y_train = y_train_confid, 
                    X_test = X_test_confid_part, y_test = y_test_confid, paths_list=paths_list,
                    models_list=models_list, model_name_list=model_name_list, 
                    hyperparams_list = hyperparams_list)
            if run_full:
                string = '''
                    \t\t RUN PARTIAL BEST CONFIDENTIAL
                    '''
                print(string)
                cls_conf_part.run_full(ensemble.RandomForestClassifier, 
                    {"class_weight":'balanced', "oob_score":True, "n_jobs":-1, 
                    "n_estimators":90, "max_depth":40, "min_samples_split":2, 
                    "min_samples_leaf":2, "max_features":'sqrt'})
                '''
                <class 'sklearn.ensemble._forest.RandomForestClassifier'> score is  0.4649347153048863
                '''

    if run_orig_protect:
        string = '''
            \t RUN ORIGINAL PROTECTED MODELS
            '''
        print(string)
        paths_list = ["../data/model_scores_08_08_prot_orig.csv",
                  "../data/best_model_prot_orig.gz"]

        cls_prot_orig = FitModels(X_train = X_train_protected_orig, y_train = y_train_protected, 
            X_test = X_test_protected_orig, y_test = y_test_protected, paths_list=paths_list,
            models_list=models_list, model_name_list=model_name_list, 
            hyperparams_list = hyperparams_list)
        
        cls_prot_orig.run_full(ensemble.RandomForestClassifier, 
            {"class_weight":'balanced', "oob_score":True, "n_jobs":-1, 
            "n_estimators":90, "max_depth":40, "min_samples_split":2, 
            "min_samples_leaf":2, "max_features":'sqrt'})
        '''
        <class 'sklearn.ensemble._forest.RandomForestClassifier'> score is  0.3809466312484907

        '''

    if run_orig_confid:
        string = '''
            \t RUN ORIGINAL CONFID MODELS
            '''
        print(string)
        paths_list = ["../data/model_scores_08_08_conf_orig.csv",
                  "../data/best_model_08_08_conf_orig.gz"]

        cls_conf_orig = FitModels(X_train = X_train_confid_orig, y_train = y_train_confid, 
            X_test = X_test_confid_orig, y_test = y_test_confid, paths_list=paths_list,
            models_list=models_list, model_name_list=model_name_list, 
            hyperparams_list = hyperparams_list)
        
        cls_conf_orig.run_full(ensemble.RandomForestClassifier, 
            {"class_weight":'balanced', "oob_score":True, "n_jobs":-1, 
            "n_estimators":90, "max_depth":40, "min_samples_split":2, 
            "min_samples_leaf":2, "max_features":'sqrt'})
        '''
        <class 'sklearn.ensemble._forest.RandomForestClassifier'> score is  0.3850397041886403
        '''
    shap_save = ["../data/shap_val_orig_protect.gz",
                 "../data/shap_val_orig_confid.gz",
                 "../data/shap_val_part_protect.gz",
                 "../data/shap_val_part_confid.gz",
                 "../data/shap_val_full_protect.gz",
                 "../data/shap_val_full_confid.gz"]
    shap_img = ["../img/shap_org_protect.jpg",
                "../img/shap_org_confid.jpg",
                "../img/shap_part_protect.jpg",
                "../img/shap_part_confid.jpg",
                "../img/shap_full_protect.jpg",
                "../img/shap_full_confid.jpg", 
                "../img/shap_full_protect.jpg"]
    if run_shap_orig_protect:
        string = '''
            SHAP VALUES: ORIGINAL MODEL PROTECTED VIEW
            '''
        print(string)
        paths_list = ["../data/model_scores_08_08_prot_orig.csv",
                  "../data/best_model_08_08_prot_orig.gz"]
        model = load(paths_list[1])
        
        t0 = time.default_timer()
        explainer = shap.TreeExplainer(model, data=X_train_protected_orig)
        shap_values = explainer.shap_values(X_train_protected_orig)
        t1 = (time.default_timer() - t0)/60
        print("Time to run Shap Values is :", t1)
        dump(shap_values, shap_save[0], compress=True)
        fig = plt.gcf()
        shap.summary_plot(shap_values[1], X_train_protected_orig)
        fig.savefig(shap_img[0], bbox_inches='tight', dpi=400)
        plt.show()

    if run_shap_orig_confid:
        string = '''
            SHAP VALUES: ORIGINAL MODEL CONFIDENTIAL VIEW
            '''
        print(string)
        paths_list = ["../data/model_scores_08_08_conf_orig.csv",
                  "../data/best_model_08_08_conf_orig.gz"]
        model = load(paths_list[1])
        
        t0 = time.default_timer()
        explainer = shap.TreeExplainer(model, data=X_train_confid_orig)
        shap_values = explainer.shap_values(X_train_confid_orig)
        t1 = (time.default_timer() - t0)/60
        print("Time to run Shap Values is :", t1)
        dump(shap_values, shap_save[1], compress=True)
        fig = plt.gcf()
        shap.summary_plot(shap_values[1], X_train_confid_orig)
        fig.savefig(shap_img[1], bbox_inches='tight', dpi=400)
        plt.show()

    if run_shap_prot_part:
        string = '''
            SHAP VALUES: PARTIAL MODEL PROTECTED VIEW
            '''
        print(string)
        paths_list = ["../data/model_scores_08_08_part_protect.csv",
                  "../data/best_model_08_08_part_protect.gz"]
        model = load(paths_list[1])
        
        t0 = time.default_timer()
        explainer = shap.TreeExplainer(model, data=X_train_protected_part)
        shap_values = explainer.shap_values(X_train_protected_part)
        t1 = (time.default_timer() - t0)/60
        print("Time to run Shap Values is :", t1)
        dump(shap_values, shap_save[2], compress=True)
        fig = plt.gcf()
        shap.summary_plot(shap_values[1], X_train_protected_part)
        fig.savefig(shap_img[2], bbox_inches='tight', dpi=400)
        plt.show()

    if run_shap_conf_part:
        string = '''
            SHAP VALUES: PARTIAL MODEL CONFIDENTIAL VIEW
            '''
        print(string)
        paths_list = ["../data/model_scores_08_08_part_confid.csv",
                  "../data/best_model_08_08_part_conf.gz"]
        model = load(paths_list[1])
        
        t0 = time.default_timer()
        explainer = shap.TreeExplainer(model, data=X_train_confid_part)
        shap_values = explainer.shap_values(X_train_confid_part)
        t1 = (time.default_timer() - t0)/60
        print("Time to run Shap Values is :", t1)
        dump(shap_values, shap_save[3], compress=True)
        fig = plt.gcf()
        shap.summary_plot(shap_values[1], X_train_confid_part)
        fig.savefig(shap_img[3], bbox_inches='tight', dpi=400)
        plt.show()


    if run_shap_prot_full:
        string = '''
            SHAP VALUES: FULL MODEL PROTECTED VIEW
            '''
        print(string)
        paths_list = ["../data/model_scores_08_08_full_protect.csv",
                  "../data/best_model_08_08_full_protect.gz"]
        model = load(paths_list[1])
        
        t0 = time.default_timer()
        explainer = shap.TreeExplainer(model, data=X_train_protected_full)
        shap_values = explainer.shap_values(X_train_protected_full)
        t1 = (time.default_timer() - t0)/60
        print("Time to run Shap Values is :", t1)
        dump(shap_values, shap_save[4], compress=True)
        fig = plt.gcf()
        shap.summary_plot(shap_values[1], X_train_protected_full)
        fig.savefig(shap_img[4], bbox_inches='tight', dpi=400)
        plt.show()


    if run_shap_conf_full:
        string = '''
            SHAP VALUES: FULL MODEL CONFIDENTIAL VIEW
            '''
        print(string)
        paths_list = ["../data/model_scores_08_08_full_confid.csv",
                  "../data/best_model_08_08_full_conf.gz"]
        model = load(paths_list[1])
        
        t0 = time.default_timer()
        explainer = shap.TreeExplainer(model, data=X_train_confid_full)
        shap_values = explainer.shap_values(X_train_confid_full)
        t1 = (time.default_timer() - t0)/60
        print("Time to run Shap Values is :", t1)
        dump(shap_values, shap_save[5], compress=True)
        fig = plt.gcf()
        shap.summary_plot(shap_values[1], X_train_confid_full)
        fig.savefig(shap_img[5], bbox_inches='tight', dpi=400)
        plt.show()


    if plot:
        print(string)
        shap_values = load(shap_save[1])
        fig = plt.gcf()
        ax = plt.gca()
        ax.set_title("Shapley - Original Model", fontsize=25)
        shap.summary_plot(shap_values[1], X_train_confid_orig, max_display=30)
        fig.savefig(shap_img[1], bbox_inches='tight', dpi=400)
        plt.show()


        print(string)
        shap_values = load(shap_save[3])
        fig = plt.gcf()
        ax = plt.gca()
        ax.set_title("Shapley - Partial Model", fontsize=25)
        shap.summary_plot(shap_values[1], X_train_confid_part)
        
        fig.savefig(shap_img[3], bbox_inches='tight', dpi=400)
        plt.show()

        print(string)
        shap_values = load(shap_save[3])
        X_train_confid_part.columns = X_train_protected_part.columns
        fig = plt.gcf()
        ax = plt.gca()
        ax.set_title("Shapley - Partial Model: Public", fontsize=25)
        shap.summary_plot(shap_values[1], X_train_confid_part)
        fig.savefig(shap_img[2], bbox_inches='tight', dpi=400)
        plt.show()


        print(string)
        shap_values = load(shap_save[5])
        fig = plt.gcf()
        ax = plt.gca()
        ax.set_title("Shapley - Full Model", fontsize=25)
        shap.summary_plot(shap_values[1], X_train_confid_full)
        fig.savefig(shap_img[5], bbox_inches='tight', dpi=400)
        plt.show()


        print(string)
        shap_values = load(shap_save[5])
        X_train_confid_full.columns = X_train_protected_full.columns
        fig = plt.gcf()
        ax = plt.gca()
        ax.set_title("Shapley - Full Model: Public", fontsize=25)
        shap.summary_plot(shap_values[1], X_train_confid_full)
        fig.savefig(shap_img[4], bbox_inches='tight', dpi=400)
        plt.show()
        print(X_train_protected_full.columns)

    


