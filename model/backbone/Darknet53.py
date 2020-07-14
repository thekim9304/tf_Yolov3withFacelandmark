import tensorflow as tf

from tensorflow.keras.layers import (
    Add,
    Conv2D,
    Input,
    LeakyReLU,
    BatchNormalization,
)


def DarknetConv(inputs, filters, kernel_size, strides, name):
    x = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        name=name + '_conv2d',
        use_bias=False,
        # kernel_regularizer=tf.keras.regularizers.l2(0.0005)
    )(inputs)
    x = BatchNormalization(name=name + '_bn')(x)
    x = LeakyReLU(alpha=0.1, name=name + '_leakyrelu')(x)
    return x


def DarknetResidual(inputs, filters1, filters2, name):
    shortcut = inputs
    x = DarknetConv(
        inputs, filters=filters1, kernel_size=1, strides=1, name=name + '_1x1')
    x = DarknetConv(
        x, filters=filters2, kernel_size=3, strides=1, name=name + '_3x3')
    x = Add(name=name + '_add')([shortcut, x])
    return x


def Darknet(shape=(256, 256, 3)):
    # YoloV3:
    # Table 1. Darknet-53.
    inputs = Input(shape=shape)

    x = DarknetConv(inputs, 32, kernel_size=3, strides=1, name='conv2d_0')

    x = DarknetConv(x, 64, kernel_size=3, strides=2, name='conv2d_1')
    # 1x residual blocks
    for i in range(1):
        x = DarknetResidual(x, 32, 64, 'residual_0_' + str(i))

    x = DarknetConv(x, 128, kernel_size=3, strides=2, name='conv2d_2')
    # 2x residual blocks
    for i in range(2):
        x = DarknetResidual(x, 64, 128, 'residual_1_' + str(i))

    x = DarknetConv(x, 256, kernel_size=3, strides=2, name='conv2d_3')
    # 8x residual blocks
    for i in range(8):
        x = DarknetResidual(x, 128, 256, 'residual_2_' + str(i))

    y0 = x

    x = DarknetConv(x, 512, kernel_size=3, strides=2, name='conv2d_4')
    # 8x residual blocks
    for i in range(8):
        x = DarknetResidual(x, 256, 512, 'residual_3_' + str(i))

    y1 = x

    x = DarknetConv(x, 1024, kernel_size=3, strides=2, name='conv2d_5')
    # 4x residual blocks
    for i in range(4):
        x = DarknetResidual(x, 512, 1024, 'residual_4_' + str(i))

    y2 = x

    return tf.keras.Model(inputs, (y0, y1, y2), name='darknet_53')
