import segmentation_models as sm
from tensorflow.keras.models import Model


def vgg_unet(input_shape: tuple[int, int, int],
               num_classes: int = 1,
               output_activation: str = 'sigmoid'
) -> Model:

    model = sm.Unet('vgg19', classes=num_classes, input_shape=input_shape,
                    activation=output_activation, encoder_weights='imagenet')

    return model


def densenet_unet(input_shape: tuple[int, int, int],
                  num_classes: int = 1,
                  output_activation: str = 'sigmoid'
) -> Model:

    model = sm.Unet('densenet201', classes=num_classes, input_shape=input_shape,
                    activation=output_activation, encoder_weights='imagenet')

    return model


def inceptionv3_unet(input_shape: tuple[int, int, int],
                     num_classes: int,
                     output_activation: str
) -> Model:

    model = sm.Unet('inceptionv3', classes=num_classes, input_shape=input_shape,
                    activation=output_activation, encoder_weights='imagenet')

    return model


def mobilenetv2_unet(input_shape: tuple[int, int, int],
                     num_classes: int,
                     output_activation: str
) -> Model:

    model = sm.Unet('mobilenetv2', classes=num_classes, input_shape=input_shape,
                    activation=output_activation, encoder_weights='imagenet')

    return model
