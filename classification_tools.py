import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

def drop_multi_cols(dfs:list, cols:list):
    """drop the specified columns from the specified dataframes"""
    for df in dfs:
        df.drop(cols, axis=1, inplace=True)
    
    return dfs

def predict_with_threshold(model, data, threshold:float=0.5):
    
    assert(isinstance(threshold, float))

    return np.where(model.predict_proba(data)[:,1]>threshold,1,0)


def list_onehot_columns(df, column_prefix):
    return df.columns[df.columns.str.contains(f"{column_prefix}.*", regex=True)].to_list()


class ModelAnalysis():

    def __init__(self, model, X_train, X_val, y_train, y_val, col_names):
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.col_names = col_names
        self.model = model

    def analyse(self):
        self.accuracy_train = self.model.score(self.X_train, self.y_train)
        self.accuracy_val = self.model.score(self.X_train, self.y_train)
        print("Training accuracy: {:.3f}\nValidation accuracy:{:.3f}".format(self.accuracy_train, self.accuracy_val))

        self.roc_train, self.auc_train = self.calc_roc(self.X_train, self.y_train, title="Training ROC Curve")
        self.roc_train.show()
        print("Train AUC ROC score: {:.3f}".format(self.auc_train))
        self.roc_val, self.auc_val = self.calc_roc(self.X_val, self.y_val, title="Validation ROC Curve")
        self.roc_val.show()
        print("AUC ROC score: {:.3f}".format(self.auc_val))

        self.y_train_pred = predict_with_threshold(self.model, self.X_train, threshold=0.5)
        self.conf_mat = confusion_matrix(self.y_train, self.y_train_pred)
        self.tn, self.fp, self.fn, self.tp = self.conf_mat.ravel()

        print("True Negatives: {:,}\nFalse Positives: {:,}\nFalse Negatives: {:,}\nTrue Positives: {:,}\n".format(self.tn, self.fp, self.fn, self.tp))

        self.tpr = self.tp/(self.tp+self.fn)
        self.fpr = self.fp/(self.tn+self.fp)

        print("True positive rate (selectivity): {:3f}".format(self.tpr))
        print("False positive rate (fall-out): {:3f}".format(self.fpr))

        self.plot_confusion_matrix()

        print("Coefficents:")
        self.coefficients = get_feature_weights(self.col_names, self.model)

    def plot_confusion_matrix(self):
    
        # conf_mat = confusion_matrix(y_true=labels, y_pred=predictions)
        print('Confusion matrix:\n', self.conf_mat)

        ax_labels = ['Class 0', 'Class 1']
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(self.conf_mat, cmap=plt.cm.Blues)
        fig.colorbar(cax)
        ax.set_xticklabels([''] + ax_labels)
        ax.set_yticklabels([''] + ax_labels)
        plt.title("Confusion Matrix")
        plt.xlabel('Predicted')
        plt.ylabel('Expected')
        plt.show()

    def calc_roc(self, data, labels, size=800, title="ROC Curve"):
        """Plots the receiver operating characteristics curve"""

        fpr, tpr, threshold = roc_curve(labels, self.model.predict_proba(data)[:,1])

        width = size
        height = width

        roc = px.line(x=fpr, y=tpr, width=width, height=height)
        baseline = px.line(x=[0,1], y =[0,1], width=width, height=height)
        baseline.update_traces(line=dict(color = 'rgba(50,50,50,0.2)'))

        fig = go.Figure(data=roc.data + baseline.data)
        fig.update_layout(
            autosize=False,
            width=width,
            height=height,
            xaxis_title="False Positive Rate (Sensitivty)",
            yaxis_title="True Positive Rate (1-Precision)",
            title=title,
            yaxis_range=[0,1],
            xaxis_range=[0,1]
            )

        auc_roc = roc_auc_score(labels, self.model.predict_proba(data)[:,1])

        return fig, auc_roc

    def get_feature_weights(self, print_coef=True):
        """Get a dictionary of the features (key) and their weights (value) from a logistic regression, sorted by absolute value.
        
        Model should be an sklearn (fitted) logistic regression object
        
        Columns should be a list of column names matching the indices of the fitted data.
        
        TODO: Doesn't work on single feature models."""


        assert len(self.col_names) == len(self.model.coef_.squeeze())

        self.coefficients = dict(zip(self.col_names, self.model.coef_.squeeze()))

        if print_coef:

            #sort by absolute value
            coefficients_sorted = [ (abs(v), v, k) for k,v in self.coefficients.items()]
            coefficients_sorted.sort(reverse=True)

            # print from largest to smallest
            for abs_val, val, key in coefficients_sorted:
                print("{: .2e}: {}".format(val, key))
