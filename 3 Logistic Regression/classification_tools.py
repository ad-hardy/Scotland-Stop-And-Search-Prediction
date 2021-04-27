import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from operator import itemgetter

def drop_multi_cols(dfs:list, cols:list):
    """drop the specified columns from the specified dataframes"""
    for df in dfs:
        df.drop(cols, axis=1, inplace=True)
    
    return dfs

def predict_with_threshold(model, data, threshold:float=0.5):
    
    assert(isinstance(threshold, float))

    return np.where(model.predict_proba(data)[:,1]>threshold,1,0)


def list_onehot_columns(df, column_prefix):
    """returns a list of columns from a dataframe that begin wih the given column_prefix"""
    return df.columns[df.columns.str.contains(f"{column_prefix}.*", regex=True)].to_list()


class BinaryLogisticClassifier():

    def __init__(self, X_train, X_val, y_train, y_val, col_names, max_iter=100, n_jobs=1, penalty="none", data_name_1="Training", data_name_2="Validation",
        C=1, class_weight=None, solver="lbfgs"):

        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.col_names = col_names
        self.data_name_1 = data_name_1
        self.data_name_2 = data_name_2

        self.model = LogisticRegression(
            penalty=penalty,
            max_iter=max_iter,
            n_jobs=n_jobs,
            class_weight=class_weight,
            C=C,
            solver=solver
            )
        self.fit()

    def fit(self):
       self.model.fit(self.X_train, self.y_train)

    def predict(self, threshold=0.5):
        self.y_train_pred = predict_with_threshold(self.model, self.X_train, threshold=threshold)
        self.y_val_pred = predict_with_threshold(self.model, self.X_val, threshold=threshold)
        return self.y_train_pred, self.y_val_pred

    def analyse(self, threshold=0.5, verbose=True, ROC=True,conf_matrix=False):

        self.predict(threshold=threshold)
        self.get_confusion_matrix()

        self.get_accuracy()
        if verbose:
            print("{} accuracy: {:.3f}\n{} accuracy:{:.3f}".format(self.data_name_1, self.accuracy_train, self.data_name_2, self.accuracy_val))

        if ROC and verbose:
            self.plot_roc()

        if verbose and conf_matrix:
            self.plot_confusion_matrix()
        
        self.get_tpr_fpr()

        if verbose and conf_matrix:
            print("{:>12} | {:^5} | {:^5} | {:^5} | {:^5}".format("Data", "TPR", "FPR", "TNR", "FNR"))
            print("{:>12} | {:.3f} | {:.3f} | {:.3f} | {:.3f}".format(self.data_name_1, self.tpr_train, self.fpr_train, self.tnr_train, self.fnr_train))
            print("{:>12} | {:.3f} | {:.3f} | {:.3f} | {:.3f}".format(self.data_name_2, self.tpr_val, self.fpr_val, self.tnr_val, self.fnr_val))

    def get_tpr_fpr(self):
        self.tpr_train = self.tp_train/(self.tp_train + self.fn_train)
        self.fpr_train = self.fp_train/(self.fp_train + self.tn_train)
        self.tnr_train = self.tn_train/(self.fp_train + self.tn_train)
        self.fnr_train  = self.fn_train/(self.tp_train + self.fn_train)

        self.tpr_val = self.tp_val/(self.tp_val + self.fn_val)
        self.fpr_val = self.fp_val/(self.fp_val + self.tn_val)
        self.tnr_val = self.tn_val/(self.fp_val + self.tn_val)
        self.fnr_val  = self.fn_val/(self.tp_val + self.fn_val)

    def get_accuracy(self):
        self.accuracy_train = (self.tp_train + self.tn_train)/(self.y_train.count())
        self.accuracy_val = (self.tp_val + self.tn_val)/(self.y_val.count())

    def get_confusion_matrix(self):

        self.conf_mat_train = confusion_matrix(self.y_train, self.y_train_pred)
        self.tn_train, self.fp_train, self.fn_train, self.tp_train = self.conf_mat_train.ravel()
        self.conf_mat_val = confusion_matrix(self.y_val, self.y_val_pred)
        self.tn_val, self.fp_val, self.fn_val, self.tp_val = self.conf_mat_val.ravel()

    def plot_confusion_matrix(self):
    
        cm = self.conf_mat_train

        plt.clf()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
        classNames = ['Negative','Positive']
        plt.title(f'Confusion Matrix - {self.data_name_1}')
        plt.ylabel('Truth')
        plt.xlabel('Predicted')
        tick_marks = np.arange(len(classNames))
        plt.xticks(tick_marks, classNames, rotation=45)
        plt.yticks(tick_marks, classNames)
        s = [['TN','FP'], ['FN', 'TP']]
        for i in range(2):
            for j in range(2):
                plt.text(j-0.25,i, f"{str(s[i][j])} = {str(cm[i][j])}")
        self.conf_mat_train_plt = plt
        self.conf_mat_train_plt.show()


        cm = self.conf_mat_val

        plt.clf()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
        classNames = ['Negative','Positive']
        plt.title(f'Confusion Matrix - {self.data_name_2}')
        plt.ylabel('Truth')
        plt.xlabel('Predicted')
        tick_marks = np.arange(len(classNames))
        plt.xticks(tick_marks, classNames, rotation=45)
        plt.yticks(tick_marks, classNames)
        s = [['TN','FP'], ['FN', 'TP']]
        for i in range(2):
            for j in range(2):
                plt.text(j-0.25,i, f"{str(s[i][j])} = {str(cm[i][j])}")
        self.conf_mat_val_plt = plt
        self.conf_mat_val_plt.show()

    def plot_roc(self, size=800, title_prefix=''):

        width = size
        height = width

        fpr_train, tpr_train, threshold_train = roc_curve(self.y_train, self.model.predict_proba(self.X_train)[:,1])
        fpr_val, tpr_val, threshold_val = roc_curve(self.y_val, self.model.predict_proba(self.X_val)[:,1])

        self.roc_curve = go.Figure()
        self.roc_curve.update_layout(
            autosize=False,
            width=width,
            height=height,
            xaxis_title="False Positive Rate (Sensitivty)",
            yaxis_title="True Positive Rate (1-Precision)",
            title=f"{title_prefix} ROC Curve",
            yaxis_range=[0,1],
            xaxis_range=[0,1],
            legend=dict(
                x=0.78,
                y=0.1,
                title_text='Data'
                )
            )

        self.roc_curve.add_trace(go.Scatter(x=fpr_train, y=tpr_train, name=self.data_name_1, mode="lines"))
        self.roc_curve.add_trace(go.Scatter(x=fpr_val, y=tpr_val, name=self.data_name_2, mode="lines"))
        self.roc_curve.add_trace(go.Scatter(x=[0,1], y =[0,1], name="Baseline", mode="lines", line=dict(color = 'rgba(50,50,50,0.2)')))

        self.get_auc_roc()

        self.roc_curve.add_annotation(
            x=0.2,y=0.9,
            text="AUC ROC score:</br></br>   {}: {:.3f}</br>   {}: {:.3f}".format(self.data_name_1, self.auc_roc_train, self.data_name_2, self.auc_roc_val),
            align="left",
            showarrow=False,
            )

        self.roc_curve.show()
  
    def get_auc_roc(self):
        self.auc_roc_train = roc_auc_score(self.y_train, self.model.predict_proba(self.X_train)[:,1])
        self.auc_roc_val = roc_auc_score(self.y_val, self.model.predict_proba(self.X_val)[:,1])

        return self.auc_roc_train, self.auc_roc_val

    def get_feature_weights(self, print_coef=True, sort_absolute=True):
        """Get a dictionary of the features (key) and their weights (value) from a logistic regression, sorted by absolute value.
        
        Model should be an sklearn (fitted) logistic regression object
        
        Columns should be a list of column names matching the indices of the fitted data.
        
        TODO: Doesn't work on single feature models."""

        assert len(self.col_names) == len(self.model.coef_.squeeze())

        self.coefficients = dict(zip(self.col_names, self.model.coef_.squeeze()))

        if print_coef:

            #get absolute values
            coefficients_abs = [ (v, abs(v), k) for k,v in self.coefficients.items()]
            sort_col = 0
            if sort_absolute:
                sort_col = 1
 
            coefficients_sorted = sorted(coefficients_abs, key=itemgetter(sort_col), reverse=True)

            # print from largest to smallest
            print("Coefficents:")
            for val, abs_val, key in coefficients_sorted:
                print("{: .2e}: {}".format(val, key))