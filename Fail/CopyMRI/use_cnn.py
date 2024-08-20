from keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import Precision, Recall, Accuracy
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adamax
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from tensorflow import keras
from TransformPipeler import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
print(tf.__version__)
print(keras.__version__)

######################################################################################
# Data Preprocessing
######################################################################################


def create_keras_dataframe(path):
    df = get_dataframe(path, rate=0.999)
    df.rename(columns={'file_path': 'Class Path',
              'folder': 'Class'}, inplace=True)
    df = df.drop('file_name', axis=1)
    return df


path = os.path.join(os.getcwd(), 'Dataset2', 'Training')
df = create_keras_dataframe(path=path)
# print(df)
df_test = create_keras_dataframe(
    path=os.path.join(os.getcwd(), 'Dataset2', 'Testing'))
# print(df_test)


######################################################################################
# Image Generating
######################################################################################

IMAGE_SIZE = (299, 299)
BATCH_SIZE = 16


def create_generate_data(train_df, test_df):
    batch_size = BATCH_SIZE
    img_size = IMAGE_SIZE

    _gen = ImageDataGenerator(rescale=1/255, brightness_range=(0.8, 1.2))

    test_gen = ImageDataGenerator(rescale=1/255)

    train_gen = _gen.flow_from_dataframe(
        train_df, x_col='Class Path', y_col='Class', batch_size=batch_size, target_size=img_size, shuffle=True)

    test_gen = test_gen.flow_from_dataframe(
        test_df, x_col='Class Path', y_col='Class', batch_size=batch_size, target_size=img_size, shuffle=False)

    return train_gen, test_gen


train_gen, test_gen = create_generate_data(df, df_test)


######################################################################################
# Hypothesis
######################################################################################
IMAGE_SHAPE = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)


def get_hypothesis_model_scorer_optimizer():
    img_shape = IMAGE_SHAPE
    path = os.path.join(os.getcwd(), 'SavedModel', 'base_model.h5')
    if os.path.exists(path):
        base_model = keras.models.load_model(path)
    else:

        base_model = tf.keras.applications.Xception(
            include_top=False, weights="imagenet", input_shape=img_shape, pooling='max')
        base_model.save("base_model.h5")

    # for layer in base_model.layers:
    #     layer.trainable = False

    model = Sequential([
        base_model,
        Flatten(),
        Dropout(rate=0.3),
        Dense(128, activation='relu'),
        Dropout(rate=0.25),
        Dense(4, activation='softmax')
    ])

    model.compile(Adamax(learning_rate=0.001),
                  loss=CategoricalCrossentropy(),
                  metrics=[Accuracy(), Precision(), Recall()])

    model.summary()
    return model


######################################################################################
# Testing Hypothesis
######################################################################################


def get_checkpoint_best_model():
    # Define the ModelCheckpoint callback
    checkpoint = ModelCheckpoint("best_model.h5",
                                 monitor='val_accuracy',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max')
    return checkpoint


def fit_evaluating(hypo, train_gen, test_gen):
    checkpoint = get_checkpoint_best_model()
    hist_dict = hypo.fit(train_gen, epochs=10,
                         validation_data=test_gen, shuffle=False,
                         callbacks=[checkpoint])
    return hist_dict, hypo


RUN = True
if (RUN):
    hist_dict, trained_model = fit_evaluating(
        get_hypothesis_model_scorer_optimizer(), train_gen, test_gen)
