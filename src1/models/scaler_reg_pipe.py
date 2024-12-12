from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

def get_pipeline_model(parameters=None):
    return Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])