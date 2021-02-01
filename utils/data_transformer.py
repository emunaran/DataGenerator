from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import pandas as pd
import numpy as np
from utils.utils import softmax2onehot


class DataTransformer:
    def __init__(self, ignore_categorical=False, categorical_features: list = None):
        self.ignore_categorical = ignore_categorical
        self.categorical_features = categorical_features

        self.min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.one_hot_encoder = OneHotEncoder(handle_unknown='ignore')

        self.continuous_features = None
        self.features_categories_lens = None
        self.encoded_column_names = None

    def _min_max_scale(self, data: pd.DataFrame):
        df_continuous = data.drop(columns=self.categorical_features)
        self.continuous_features = df_continuous.columns
        X_continuous = df_continuous.values  # returns a numpy array
        X_scaled = self.min_max_scaler.fit_transform(X_continuous)
        df_scaled = pd.DataFrame(X_scaled, columns=df_continuous.columns)
        return df_scaled

    def _one_hot_encode(self, data: pd.DataFrame):
        self.features_categories_lens = data[self.categorical_features].nunique()
        self.one_hot_encoder.fit(data[self.categorical_features])
        X_encoded = self.one_hot_encoder.transform(data[self.categorical_features]).toarray()
        df_encoded = pd.DataFrame(X_encoded)
        self.encoded_column_names = df_encoded.columns
        return df_encoded

    def transform(self, data: pd.DataFrame):
        processed_df = self._min_max_scale(data)
        if self.categorical_features and not self.ignore_categorical:
            df_encoded = self._one_hot_encode(data)
            processed_df = pd.concat([processed_df, df_encoded], axis=1)
        return processed_df

    def inverse_transform(self, data: pd.DataFrame):
        # inverse transform
        fake_data_continuous_scale = data[self.continuous_features]
        fake_data_continuous = self.min_max_scaler.inverse_transform(fake_data_continuous_scale)
        final_df = pd.DataFrame(fake_data_continuous, columns=self.continuous_features)

        if self.categorical_features and not self.ignore_categorical:
            converted_to_one_encode_list = []
            fake_data_categorical_encoded = data[self.encoded_column_names].values
            fake_data_categorical_encoded_list = np.array_split(fake_data_categorical_encoded,
                                                                list(self.features_categories_lens)[:-1], axis=1)
            for f_name, fake_data_categorical in zip(self.categorical_features, fake_data_categorical_encoded_list):
                fake_data_categorical = softmax2onehot(fake_data_categorical)
                converted_to_one_encode_list.append(fake_data_categorical)
            converted_one_encoded_array = np.concatenate(converted_to_one_encode_list, axis=1)
            fake_data_categorical = self.one_hot_encoder.inverse_transform(converted_one_encoded_array)
            final_df[self.categorical_features] = fake_data_categorical

        return final_df
