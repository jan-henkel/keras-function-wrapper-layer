from tensorflow.keras.layers import Layer, Input
from tensorflow.keras import Model


class FunctionWrapperLayer(Layer):
    def __init__(self, fn):
        super(FunctionWrapperLayer, self).__init__()
        self.fn = fn

    def build(self, input_shapes):
        super(FunctionWrapperLayer, self).build(input_shapes)
        if type(input_shapes) is list:
            inputs = [Input(shape[1:]) for shape in input_shapes]
        else:
            inputs = Input(input_shapes[1:])
        outputs = self.fn(inputs)
        self.fn_model = Model(inputs=inputs, outputs=outputs)
        self.fn_model.compile()

    def call(self, x):
        return self.fn_model(x)
