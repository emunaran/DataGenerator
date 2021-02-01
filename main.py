
import argparse
from model.gan.process import GANProcess
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_regression
from utils.data_transformer import DataTransformer

ALGORITHM_TYPE=['GAN', 'WGAN']
CATEGORICAL_FEATURES = ['Outcome']


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='GAN', required=False, choices=ALGORITHM_TYPE)
    parser.add_argument('--data-set', type=str, default='simulated', required=False)
    parser.add_argument('--hidden-size', type=int, default=256, required=False)
    parser.add_argument('--epochs', type=int, default=500, required=False)
    parser.add_argument('--min-epochs', type=int, default=20, required=False)
    parser.add_argument('--learning-rate', type=int, default=0.0001, required=False)
    parser.add_argument('--batch-size', type=int, default=64, required=False)
    parser.add_argument('--discrim-update-num', type=int, default=2, required=False)
    parser.add_argument('--generator-update-num', type=int, default=1, required=False)
    parser.add_argument('--discriminator-error-threshold', type=float, default=0.95, required=False)
    parser.add_argument('--generator-success-threshold', type=float, default=0.95, required=False)
    parser.add_argument('--load-model', type=str, required=False)
    parser.add_argument('--logdir', type=str, default='logs', required=False)
    parser.add_argument('--train', default=False, action='store_true')

    args = parser.parse_args()

    if args.data_set == 'simulated':
        # Simulate a dataset
        X, y = make_regression(n_samples=1000, n_features=8, random_state=0)
        feature_list = list(map(lambda x: f"F{x}", range(X.shape[1])))
        df = pd.DataFrame(X, columns=feature_list)
        df['y'] = y
    else:
        df = pd.read_csv(args.data_set, index_col=0, parse_dates=True)
        for cat_feature in CATEGORICAL_FEATURES:
            if cat_feature not in df.columns:
                raise Exception(f'{cat_feature} is not exist in the given data')

    data_transformer = DataTransformer(CATEGORICAL_FEATURES)
    processed_df = data_transformer.transform(df)

    if args.algorithm == 'GAN':
        process = GANProcess(args, processed_df)
    else:
        pass

    process.run()


if __name__ == '__main__':
    main()