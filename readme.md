# Function wrapper layer
## Example usage
```Python
from keras_function_wrapper.layers import FunctionWrapperLayer
from tensorflow.keras.layers import Conv2D
import tensorflow as tf


def resnet_block_relu6(x, kernel_size, intermediate_channels):
    output_channels = x.shape[-1]
    out = Conv2D(intermediate_channels, kernel_size, padding='same')(x)
    out = tf.nn.relu6(out)
    out = Conv2D(output_channels, kernel_size, padding='same')(out)
    out = out + x
    return out


class ResnetBlockRelu6(FunctionWrapperLayer):
    def __init__(self, kernel_size=(3, 3), intermediate_channels=32):
        super(ResnetBlockRelu6, self).__init__(
            lambda x: resnet_block_relu6(x, kernel_size, intermediate_channels)
        )
```