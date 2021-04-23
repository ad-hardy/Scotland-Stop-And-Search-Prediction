def drop_multi_cols(dfs:list, cols:list):
    """drop the specified columns from the specified dataframes"""
    for df in dfs:
        df.drop(cols, axis=1, inplace=True)
    
    return dfs

def get_feature_weights(columns:list, model, print_coef=True):
    """Get a dictionary of the features (key) and their weights (value) from a logistic regression, sorted by absolute value.
    
    Model should be an sklearn (fitted) logistic regression object
    
    Columns should be a list of column names matching the indices of the fitted data.
    
    TODO: Doesn't work on single feature models."""


    assert len(columns) == len(model.coef_.squeeze())

    coefficients = dict(zip(columns, model.coef_.squeeze()))

    #sort by absolute value
    coefficients_sorted = [ (abs(v), v, k) for k,v in coefficients.items()]
    coefficients_sorted.sort(reverse=True)

    if print_coef:
        # print from largest to smallest
        for abs_val, val, key in coefficients_sorted:
            print("{: .2e}: {}".format(val, key))

    return coefficients_sorted


import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def calc_roc(model, data, labels, size=800):
    """Plots the receiver operating characteristics curve"""

    fpr, tpr, threshold = roc_curve(labels, model.predict_proba(data)[:,1])

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
        title="ROC Curve",
        yaxis_range=[0,1],
        xaxis_range=[0,1]
        )

    auc_roc = roc_auc_score(labels, model.predict_proba(data)[:,1])

    return fig, auc_roc

import numpy as np

def predict_with_threshold(model, data, threshold:float=0.5):
    
    assert(isinstance(threshold, float))

    return np.where(model.predict_proba(data)[:,1]>threshold,1,0)


from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

def plot_confusion_matrix(conf_mat):
    
    # conf_mat = confusion_matrix(y_true=labels, y_pred=predictions)
    print('Confusion matrix:\n', conf_mat)

    ax_labels = ['Class 0', 'Class 1']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + ax_labels)
    ax.set_yticklabels([''] + ax_labels)
    plt.title("Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('Expected')
    plt.show()

