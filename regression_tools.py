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


def select_multi_cols(dfs:list, cols:list):
    """drop the specified columns from the specified dataframes"""
    for df in dfs:
        df = df[cols]
    
    return dfs