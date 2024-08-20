
from skimage.feature import hog
from Extractor import *
from sklearn.cluster import KMeans


class Segment:
    def __init__(self, n_clusters=3) -> None:
        k = n_clusters
        self._kmeans = KMeans(n_clusters=k, random_state=42)

    def get_X_from_image(self, gray_image):
        fd, hog_image = hog(gray_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                            orientations=9, block_norm='L2-Hys', visualize=True)
        hog_image = hog_image.flatten()
        retype_image = retype_image(retype_image, istype='ubyte')
        local_mean = filters.rank.mean(retype_image, np.ones((3, 3))).flatten()
        local_entropy = filters.rank.entropy(
            retype_image, np.ones((3, 3))).flatten()
        local_max = filters.rank.maximum(
            retype_image, np.ones((3, 3))).flatten()

        a = hog_image.reshape(len(hog_image), -1)
        b = local_mean.reshape(len(local_mean), -1)
        c = local_entropy.reshape(len(local_entropy), -1)
        d = local_max.reshape(len(local_max), -1)
        X = np.concatenate((a, b, c, d), axis=1)
        return X

    def call(self, gray_image):
        # Perform K-means clustering
        X = self.get_X_from_image(gray_image=gray_image)
        labels = self._kmeans.fit_predict(X=X)

        # You can use the cluster centers to assign labels to your feature array
        cluster_centers = self._kmeans.cluster_centers_
        segmented_features_map = cluster_centers[labels]

        return segmented_features_map
