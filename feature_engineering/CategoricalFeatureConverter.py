from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
import pandas as pd

class CategoricalFeatureConverter(BaseEstimator, TransformerMixin):
    def __init__(self, whichone="label"):
        self.whichone = whichone
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        cat = pd.Categorical(X)
        cat_variables = X[cat.categories.values]
        if(self.whichone == "label"):
            for i in cat_variables:
                le = LabelEncoder()
                X[i] = le.fit_transform(X[i].astype(str))
        elif(self.whichone == "onehot"):
            cat_dummies = pd.get_dummies(cat_variables, drop_first=True)
            X = X.drop(cat.categories.values, axis=1)
            X = pd.concat([X, cat_dummies], axis=1)
        else:
            raise
        return X
