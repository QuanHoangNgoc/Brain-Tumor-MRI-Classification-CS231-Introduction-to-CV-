import Const
import json
from collections import defaultdict
from joblib import Parallel, delayed
import pywt
from skimage.morphology import disk
from skimage.filters.rank import entropy
from sklearn.decomposition import PCA
from skimage.feature import local_binary_pattern
from skimage import io, color
from skimage.transform import resize
from skimage.feature import hog
from skimage import io, color, img_as_ubyte
from Image import *


import numpy as np
from skimage import io, color, filters
from skimage.filters import gabor_kernel
import matplotlib.pyplot as plt


class GaborExtraction(ExtractLayer):
    def __init__(self):
        super().__init__()

    def mono_call(self, gray_image):
        image = copy_mat(gray_image)
        if len(image.shape) == 3:  # Check if the image is colored
            image = color.rgb2gray(image)  # Convert to grayscale
        # Define scales and orientations
        scales = [4, 8, 16]  # Example scales
        orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Example orientations
        # Initialize list to hold the features
        texton_features = []
        # Apply Gabor filter at each scale and orientation
        for scale in scales:
            for theta in orientations:
                # Create Gabor kernel
                kernel = gabor_kernel(frequency=1.0/scale, theta=theta)

                # Filter the image and get the real and imaginary responses
                filtered_real, filtered_imag = filters.gabor(
                    image, frequency=1.0/scale, theta=theta)

                # Extract features (you can use the mean, standard deviation, etc.)
                features = {
                    'scale': scale,
                    'orientation': theta,
                    'mean_real': np.mean(filtered_real),
                    'stddev_real': np.std(filtered_real),
                    'mean_imag': np.mean(filtered_imag),
                    'stddev_imag': np.std(filtered_imag),
                }
                texton_features.append(features)

        # Convert type output to np
        for i in range(len(texton_features)):
            texton_features[i] = list(texton_features[i].values())
        texton_features = np.array(texton_features)
        return texton_features


# from skimage.feature import greycomatrix, greycoprops


def greycomatrix(image, distances, angles, levels=256, symmetric=False, normed=False):
    max_dist = max(distances)
    S = image.shape
    G = np.zeros((levels, levels, len(distances),
                 len(angles)), dtype=np.uint32)

    for d, distance in enumerate(distances):
        for a, angle in enumerate(angles):
            for i in range(S[0]):
                for j in range(S[1]):
                    i_off = int(i + distance * np.sin(angle))
                    j_off = int(j + distance * np.cos(angle))
                    if i_off < 0 or j_off < 0 or i_off >= S[0] or j_off >= S[1]:
                        continue
                    G[image[i, j], image[i_off, j_off], d, a] += 1

    if symmetric:
        G += G.transpose(1, 0, 2, 3)

    if normed:
        G = G.astype(np.float64)
        G /= G.sum()

    return G


def greycoprops(P, prop='contrast'):
    (num_level, num_level2, num_dist, num_angle) = P.shape
    assert num_level == num_level2
    results = np.zeros((num_dist, num_angle), dtype=np.float64)

    if prop == 'contrast':
        for i in range(num_level):
            for j in range(num_level):
                results += (i - j) ** 2 * P[i, j]
    elif prop == 'dissimilarity':
        for i in range(num_level):
            for j in range(num_level):
                results += np.abs(i - j) * P[i, j]
    elif prop == 'homogeneity':
        for i in range(num_level):
            for j in range(num_level):
                results += P[i, j] / (1. + np.abs(i - j))
    elif prop == 'energy':
        for i in range(num_level):
            for j in range(num_level):
                results += P[i, j] ** 2
    elif prop == 'correlation':
        # Implementation of correlation here is more complex and requires mean and std dev calculations
        pass
    elif prop == 'ASM':
        for i in range(num_level):
            for j in range(num_level):
                results += P[i, j] ** 2
    else:
        raise ValueError("Invalid prop: {}".format(prop))

    return results


class GLCMExtraction(ExtractLayer):
    def __init__(self):
        super().__init__()

    def mono_call(self, gray_image):
        # !!!Convert the image to 8-bit unsigned integer format
        _image = img_as_ubyte(copy_mat(gray_image))
        # Compute the GLCM
        # The distances and angles should be chosen based on the scale and orientation of the textures
        distances = [1]  # List of pixel pair distance offsets
        # List of pixel pair angles in radians (e.g., 0 for horizontal pairs)
        angles = [0]

        glcm = greycomatrix(_image, distances=distances,
                            angles=angles, symmetric=True, normed=True)

        # Compute GLCM properties
        contrast = greycoprops(glcm, 'contrast')
        dissimilarity = greycoprops(glcm, 'dissimilarity')
        homogeneity = greycoprops(glcm, 'homogeneity')
        energy = greycoprops(glcm, 'energy')
        correlation = greycoprops(glcm, 'correlation')
        ASM = greycoprops(glcm, 'ASM')  # Angular Second Moment (ASM)

        # Print the results
        return np.array([contrast[0, 0], dissimilarity[0, 0], homogeneity[0, 0], energy[0, 0],
                         correlation[0, 0], ASM[0, 0]])


class HOGExtraction(ExtractLayer):
    def __init__(self):
        super().__init__()
        self.DisplayImage = DISPLAY_IMAGE

    def mono_call(self, gray_image):
        # Compute HOG features
        # pixels_per_cell: Size (in pixels) of a cell.
        # cells_per_block: Number of cells in each block.
        # orientations: Number of orientation bins.
        # block_norm: Block normalization method.
        # visualize: If True, also returns an image of the HOG.
        fd, hog_image = hog(copy_mat(gray_image), pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                            orientations=9, block_norm='L2-Hys', visualize=True)

        # fd contains the HOG feature descriptor
        # hog_image contains the visualization of HOG if visualize=True
        # Display the HOG image
        if (self.DisplayImage):
            import matplotlib.pyplot as plt
            plt.imshow(hog_image, cmap='gray')
            plt.show()
        return fd, hog_image


class LBPExtraction(ExtractLayer):
    def __init__(self):
        super().__init__()
        # Parameters for LBP
        self.Radius = 3
        self.N_points = 8 * self.Radius
        self.DisplayImage = DISPLAY_IMAGE

    def mono_call(self, gray_image):
        # !!!Convert the image to 8-bit unsigned integer format
        _image = img_as_ubyte(copy_mat(gray_image))
        # Compute LBP features
        lbp_image = local_binary_pattern(
            _image, self.N_points, self.Radius, method='uniform')
        # Flatten the LBP image to create a feature vector
        lbp_features = lbp_image.flatten()

        # Display the LBP image
        if (self.DisplayImage):
            import matplotlib.pyplot as plt
            plt.imshow(lbp_image, cmap='gray')
            plt.show()
        return lbp_features, lbp_image


class EntropyMeanWalavet(ExtractLayer):
    def __init__(self):
        super().__init__()
        self.DisplayImage = DISPLAY_IMAGE

    def mono_call(self, gray_image):
        # Flatten the image to turn it into a 1D array
        flat_image = copy_mat(gray_image).flatten()
        # Perform PCA
#         pca = PCA(n_components=2)  # Adjust the number of components as needed
#         transformed_data = pca.fit_transform(flat_image.reshape(-1, 1))
        # transformed_data now contains the principal components of the image

        # Calculate the entropy
        _image = img_as_ubyte(gray_image)
        # Use a disk of radius 5 for the local neighborhood
        entropy_image = entropy(_image, disk(5))
        # Display the entropy image
        if (self.DisplayImage):
            import matplotlib.pyplot as plt
            plt.imshow(entropy_image, cmap='gray')
            plt.show()

        # Calculate the mean
        mean_value = np.mean(gray_image)

        # Perform a 2D discrete wavelet transform
        coeffs = pywt.dwt2(gray_image, 'haar')
        # coeffs is a tuple: (cA, (cH, cV, cD))
        # cA: Approximation coefficients
        # cH: Horizontal detail coefficients
        # cV: Vertical detail coefficients
        # cD: Diagonal detail coefficients
        # You can reconstruct the image using the inverse transform
        reconstructed_image = pywt.idwt2(coeffs, 'haar')
        # Display the reconstructed image
        if (self.DisplayImage):
            import matplotlib.pyplot as plt
            plt.imshow(reconstructed_image, cmap='gray')
            plt.show()

        return np.append(np.append(entropy_image.flatten(), mean_value),
                         reconstructed_image.flatten()), entropy_image, reconstructed_image


# LABEL = ['glioma_tumor',
#          'meningioma_tumor',
#          'no_tumor',
#          'pituitary_tumor']
LABEL = Const.LABEL


class Putall(ExtractLayer):
    def __init__(self):
        super().__init__()
        self.block_0 = Enhancement()
        self.block_1 = GaborExtraction()
        self.block_2 = GLCMExtraction()
        self.block_3 = HOGExtraction()
        self.block_4 = LBPExtraction()
        self.block_5 = EntropyMeanWalavet()
        self.DisplayImage = DISPLAY_IMAGE
        self.Verbose = False

    def store_all_data(self, row_list): pass
    # if (Const.STORE_ALL_DATA_PUT_ALL == False):
    #     ut.mess("NOT SAVE ALL DATA")

    # Const.CNT_EXTRACT += 1
    # current_folder = os.getcwd()
    # sub_folder = os.path.join(
    #     current_folder, "sub_folder_{}".format(Const.CNT_EXTRACT))
    # if not os.path.exists(sub_folder):
    #     os.makedirs(sub_folder)

    # ut.over(row_list, "row_list " + str(Const.CNT_EXTRACT))
    # cnt = -1
    # name = Const.NAME

    # for c in range(len(row_list[0])):
    #     xi = row_list[0][c]
    #     if (isinstance(xi, tuple)):
    #         for t in range(len(xi)):
    #             cnt += 1
    #             # ut.mess(cnt, c, t)
    #             arr = np.array([row[c][t] for row in row_list])
    #             name_np_file = name[cnt] + ".npy"
    #             np_file_path = os.path.join(sub_folder, name_np_file)
    #             with open(np_file_path, 'wb') as f:
    #                 np.save(f, arr)

    #     else:
    #         cnt += 1
    #         # ut.mess(cnt, c)
    #         arr = np.array([row[c] for row in row_list])
    #         name_np_file = name[cnt] + ".npy"
    #         np_file_path = os.path.join(sub_folder, name_np_file)
    #         with open(np_file_path, 'wb') as f:
    #             np.save(f, arr)

    # ut.note_verbose(True, "data saved!!!")

    def call(self, df):
        ut.mess(df)
        x_training, y_training = [], []
        for i in range(len(df)):
            folder = df['folder'].iloc[i]
            y = LABEL.index(folder)
            y_training.append(y)
        ut.note_verbose(True, "done y_training")

        x_training = joblib_loop(df)
        ut.note_verbose(True, "done x_training")

        x_training, y_training = np.array(x_training), np.array(y_training)
        ut.over(x_training, "x_training")
        ut.over(y_training, "y_training")
        return x_training, y_training

    def mono_call(self, path): pass
    # org_image = read_image(path, as_gray=True)
    # # !!!In the example above, anti_aliasing=True is used to apply a Gaussian filter
    # # to smooth the image before downsampling, which can help reduce aliasing artifacts.
    # org_image = ski.transform.resize(org_image, SIZE, anti_aliasing=True)
    # if (self.DisplayImage):
    #     show_image(org_image, 0)
    # if (self.Verbose):
    #     ut.over(org_image, "org-image: " + path)
    # x = self.block_0.call(org_image)
    # x1 = self.block_1.call(x)
    # x2 = self.block_2.call(x)
    # x3 = self.block_3.call(x)
    # x4 = self.block_4.call(x)
    # x5 = self.block_5.call(x)
    # complex_fd = np.array([])
    # for xi in (x1, x2, x3, x4, x5):
    #     complex_fd = np.append(complex_fd, xi.flatten())
    # return complex_fd


_putall = Putall()


def parallel_return_task(path):
    # ut.note_verbose(True, "load {}".format(path))
    org_image = read_image(path, as_gray=True)
    # !!!In the example above, anti_aliasing=True is used to apply a Gaussian filter
    # to smooth the image before downsampling, which can help reduce aliasing artifacts.
    org_image = ski.transform.resize(org_image, SIZE, anti_aliasing=True)
    x = copy_mat(org_image)

    blocks = [_putall.block_0, _putall.block_1, _putall.block_2,
              _putall.block_3, _putall.block_4, _putall.block_5]

    for control in Const.FEATURE_PIPELINE:
        z_list = []
        for insight in control:
            block = blocks[insight[0]]
            z = block.call(x)
            if (isinstance(z, tuple) == True):
                z = z[insight[1]]
            z_list.append(z)
        if (len(z_list) > 1):
            out = np.array([])
            for z in z_list:
                out = np.append(out, z.flatten())
        else:
            out = z_list[0]
        x = out
    return x


def joblib_loop(df: pd.DataFrame):
    pics = df['file_path'].to_list()
    # return [parallel_return_task(path) for path in pics]
    return Parallel(n_jobs=8)(delayed(parallel_return_task)(path) for path in pics)
