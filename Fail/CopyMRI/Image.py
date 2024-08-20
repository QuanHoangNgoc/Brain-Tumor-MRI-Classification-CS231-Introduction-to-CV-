from Utils import *

######################################################################################################
# CONST FOR IMAGE
######################################################################################################
COLOR = 'gray'
SIZE = (256, 256)
TYPE = 'float'
VALUE = (0, 1)
DISPLAY_IMAGE = False


def copy_mat(x):
    try:
        return x.copy()
    except:
        return x


class ExtractLayer(keras.layers.Layer):
    def __init__(self, **_hyparams): super().__init__(**_hyparams)
    def call(self, gray_image): return self.mono_call(gray_image)
    def mono_call(self, *argv): pass


class Enhancement(ExtractLayer):
    def init_params(self):
        # Params
        self.Sigma = 2
        self.Clip_limit_value = 2.0
        self.Tile_grid_size_value = (8, 8)

    def __init__(self):
        super().__init__()
        self.init_params()

    def mono_call(self, gray_image):
        # Apply Gaussian blur to the image
        blurred_image = ski.filters.gaussian(
            copy_mat(gray_image), sigma=self.Sigma)
        # Perform the linear combination as per the given expression
        # Note that the images must be in floating point representation for this operation
        combined_image = 1.5 * gray_image - 0.5 * blurred_image
        # Clip the values to the valid range [0, 1] to ensure a valid image
        combined_image = np.clip(combined_image, 0, 1)
        # Apply CLAHE
        clahe_image = ski.exposure.equalize_adapthist(
            combined_image,
            clip_limit=self.Clip_limit_value, nbins=256, kernel_size=self.Tile_grid_size_value)
        return clahe_image


# examples
if (__name__ == "__main__"):
    path = os.path.join(os.getcwd(), 'Dataset\Testing\glioma_tumor\image.jpg')
    org_image = read_image(path)
    ut.over(org_image)
    show_image(org_image, 0)

    enh = Enhancement()
    image = enh.call(org_image)
    ut.over(image)
    show_image(image, 0)
