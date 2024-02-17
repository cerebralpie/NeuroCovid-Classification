import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, InputLayer,
    Conv2D, MaxPooling2D, UpSampling2D,
    InputLayer, Concatenate, Flatten,
    Reshape, Lambda, Embedding, dot,
    Dropout, GlobalAveragePooling2D
)
from tensorflow.keras.models import Model, load_model, Sequential


def build_siamese_model(inputShape, embeddingDim=48):
    # specify the inputs for the feature extractor network
    inputs = Input(inputShape)

    # define the first set of CONV => RELU => POOL => DROPOUT layers
    x = Conv2D(64, (2, 2), padding="same", activation="relu")(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    # second set of CONV => RELU => POOL => DROPOUT layers
    x = Conv2D(64, (2, 2), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    # prepare the final outputs
    pooledOutput = GlobalAveragePooling2D()(x)
    outputs = Dense(embeddingDim)(pooledOutput)
    # build the model
    model = Model(inputs, outputs)
    # return the model to the calling function
    return model