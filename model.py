import tensorflow as tf
from tensorflow.keras import layers, Model # type: ignore

@tf.keras.utils.register_keras_serializable()
class TCNGenerator(tf.keras.Model):
    def __init__(self, input_size, **kwargs):
        super(TCNGenerator, self).__init__()
        self.input_size = input_size

        self.conv1 = layers.Conv1D(filters=64, kernel_size=5, dilation_rate=1, activation='relu', padding='causal', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.bn1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(0.2)

        self.conv2 = layers.Conv1D(filters=64, kernel_size=5, dilation_rate=2, activation='relu', padding='causal', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.bn2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(0.2)

        self.conv3 = layers.Conv1D(filters=64, kernel_size=3, dilation_rate=4, activation='relu', padding='causal', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.bn3 = layers.BatchNormalization()
        self.dropout3 = layers.Dropout(0.2)

        self.residual_conv = layers.Conv1D(filters=64, kernel_size=1, padding='same')

        self.global_pool = layers.GlobalAveragePooling1D()

        self.dense1 = layers.Dense(32, activation='relu')
        self.dropout_dense1 = layers.Dropout(0.1)

        self.output_layer = layers.Dense(1)

    def call(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.dropout3(out)

        residual = self.residual_conv(x)
        out = layers.add([out, residual])
        out = tf.keras.activations.relu(out)

        out = self.global_pool(out)

        out = self.dense1(out)
        out = self.dropout_dense1(out)

        out = self.output_layer(out)

        return out

    def get_config(self):
        config = super(TCNGenerator, self).get_config()
        config.update({"input_size": self.input_size})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(input_size=10, **config)

class DiscriminatorMLP(tf.keras.Model):
    def __init__(self):
        super(DiscriminatorMLP, self).__init__()

        self.flatten = layers.Flatten()  
        self.dense1 = layers.Dense(128)  
        self.leaky_relu1 = layers.LeakyReLU(negative_slope=0.2) 

        self.dense2 = layers.Dense(64)   
        self.leaky_relu2 = layers.LeakyReLU(negative_slope=0.2)  

        self.dense3 = layers.Dense(32) 
        self.leaky_relu3 = layers.LeakyReLU(negative_slope=0.2) 

        self.output_layer = layers.Dense(1, activation='sigmoid')  

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.leaky_relu1(x)
        x = self.dense2(x)
        x = self.leaky_relu2(x)
        x = self.dense3(x)
        x = self.leaky_relu3(x)
        output = self.output_layer(x)
        return output
