from sklearn.preprocessing import MinMaxScaler
import pandas as pd


class DataTransformer:
    def __init__(self, categorical_features: list = None):
        self.categorical_features = categorical_features
        self.min_max_scaler = MinMaxScaler(feature_range=(-1, 1))

    def scaler(self):
        pass

    def one_hot_encoder(self):
        pass

    def transform(self, data: pd.DataFrame):
        if self.categorical_features:
            self.one_hot_encoder()
        df_continuous = data.drop(columns=self.categorical_features)
        X_continuous = df_continuous.values  # returns a numpy array
        X_scaled = self.min_max_scaler.fit_transform(X_continuous)
        df_scaled = pd.DataFrame(X_scaled, columns=df_continuous.columns)
        return df_scaled

    def inverse_transform(self, data: pd.DataFrame):
        pass
