from scipy.stats.mstats import winsorize
from typing import Literal
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin

class HandlingOutliers(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self 
    
    def transform(self, X, y=None):
        
        X['weathersit'] = winsorize(X['weathersit'], limits=(0.01, 0.01))
        X['hum'] = winsorize(X['hum'], limits=(0.01, 0.01))
        X['windspeed'] = winsorize(X['windspeed'], limits=(0.01, 0.01))
        return X
    
    def set_output(self, transform: Literal['default', 'pandas']):
        return super().set_output(transform=transform)