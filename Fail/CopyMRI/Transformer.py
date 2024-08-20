from FileDataFrame import *
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
# from sklearn.pipeline import Pipeline


class TransformLayer(StandardScaler):
    def __init__(self, **_hyparams): super().__init__(**_hyparams)
    def fit(self, X, y=None): pass
    def transform(self, X, y=None): pass


class DataTrainer(TransformLayer):
    def __init__(self, **_hyparams):
        super().__init__(**_hyparams)
        self.putall = Putall()

    def transform(self, df):
        x, y = self.putall.call(df=df)
        return x, y


def get_ins_scaler():
    return StandardScaler()


class PCACustomizer(PCA):
    def __init__(self, **_hyparams): super().__init__(**_hyparams)

    def transform(self, X_std):
        X_pca = super().transform(X_std)
        # Get the explained variance ratio for each principal component
        explained_variance_ratio = self.explained_variance_ratio_
        # Choose the number of principal components to retain
        # You can use a threshold for the cumulative explained variance ratio
        # or specify the number of components directly
        n_components = np.sum(np.cumsum(explained_variance_ratio) < 0.95) + 1
        # Reduce the dimensionality of the data
        X_reduced = X_pca[:, :n_components]
        return X_reduced
