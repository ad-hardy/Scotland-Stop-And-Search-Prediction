def drop_multi_cols(dfs:list, cols:list):
    """drop the specified columns from the specified dataframes"""
    for df in dfs:
        df.drop(cols, axis=1, inplace=True)
    
    return dfs

def get_feature_weights(columns:list, model):
    """Get a dictionary of the features (key) and their weights (value) from a logistic regression.
    
    Model should be an sklearn (fitted) logistic regression object
    
    Columns should be a list of column names matching the indices of the fitted data."""


    assert len(columns) == len(model.coef_.squeeze())
    return dict(zip(columns, model.coef_.squeeze()))

import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve

def plot_roc(model, data, labels, size=800):
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

    return fig

import numpy as np

def predict_with_threshold(model, data, threshold:float=0.5):
    
    assert(isinstance(threshold, float))

    return np.where(model.predict_proba(data)[:,1]>threshold,1,0)