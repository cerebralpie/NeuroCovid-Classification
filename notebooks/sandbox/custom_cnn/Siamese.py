import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, InputLayer,
    Conv2D, MaxPooling2D, UpSampling2D,
    InputLayer, Concatenate, Flatten,
    Reshape, Lambda, Embedding, dot
)
from tensorflow.keras.models import Model, load_model, Sequential


def siamese_model():

    # Encoder
    input_layer = Input((256, 256, 3))
    layer1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_layer)
    layer2 = MaxPooling2D((2, 2), padding='same')(layer1)
    layer3 = Conv2D(8, (3, 3), activation='relu', padding='same')(layer2)
    layer4 = MaxPooling2D((2, 2), padding='same')(layer3)
    layer5 = Flatten()(layer4)
    embeddings = Dense(16, activation=None)(layer5)
    norm_embeddings = tf.nn.l2_normalize(embeddings, axis=-1)

    # Model
    model = Model(inputs=input_layer, outputs=norm_embeddings)

    input1 = Input((256, 256, 3))
    input2 = Input((256, 256, 3))

    left_model = model(input1)
    right_model = model(input2)

    dot_product = dot([left_model, right_model], axes=3, normalize=False)

    siamese = Model(inputs=[input1, input2], outputs=dot_product)

    return siamese
