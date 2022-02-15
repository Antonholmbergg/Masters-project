from tensorflow import keras


class Fc_model(keras.Model):
    
    def __init__(self, layers_widths, activation):
        super(Fc_model, self).__init__()
        self.layer_widths = layers_widths
        self.dense_layers = []
        self.input_layer = keras.layers.Dense(10)
        for width in self.layer_widths:
            self.dense_layers.append(keras.layers.Dense(width, activation=activation))
        self.output_layer = keras.layers.Dense(8, activation=activation)

    def call(self, inputs):
        x = self.input_layer(inputs)
        for layer in self.dense_layers:
            x = layer(x)
        return self.output_layer(x)


