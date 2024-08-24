from sklearn.metrics import r2_score


# Custom r2 needed because tf 2.11 does not have this as a metric
# Additionally, have to use tf 2.11 because of cluster constraints
def r2_score_wrapper(y_true, y_pred):
    """Custom metric function to calculate R-squared."""
    return r2_score(y_true, y_pred)
