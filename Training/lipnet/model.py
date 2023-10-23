import tensorflow as tf
from layers import CTC

class LipNet(object):
    def __init__(self, img_c=3, img_w=100, img_h=50, frames_n=100, absolute_max_string_len=54, output_size=28):
        self.img_c = img_c
        self.img_w = img_w
        self.img_h = img_h
        self.frames_n = frames_n
        self.absolute_max_string_len = absolute_max_string_len
        self.output_size = output_size
        self.build()

    def build(self):
        if tf.keras.backend.image_data_format() == 'channels_first':
            input_shape = (self.img_c, self.frames_n, self.img_w, self.img_h)
        else:
            input_shape = (self.frames_n, self.img_w, self.img_h, self.img_c)

        self.input_data = tf.keras.Input(name='the_input', shape=input_shape, dtype='float32')

        self.zero1 = tf.keras.layers.ZeroPadding3D(padding=(1, 2, 2), name='zero1')(self.input_data)
        self.conv1 = tf.keras.layers.Conv3D(32, (3, 5, 5), strides=(1, 2, 2), kernel_initializer='he_normal', name='conv1')(self.zero1)
        self.batc1 = tf.keras.layers.BatchNormalization(name='batc1')(self.conv1)
        self.actv1 = tf.keras.layers.Activation('relu', name='actv1')(self.batc1)
        self.drop1 = tf.keras.layers.SpatialDropout3D(0.5)(self.actv1)
        self.maxp1 = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max1')(self.drop1)

        self.zero2 = tf.keras.layers.ZeroPadding3D(padding=(1, 2, 2), name='zero2')(self.maxp1)
        self.conv2 = tf.keras.layers.Conv3D(64, (3, 5, 5), strides=(1, 1, 1), kernel_initializer='he_normal', name='conv2')(self.zero2)
        self.batc2 = tf.keras.layers.BatchNormalization(name='batc2')(self.conv2)
        self.actv2 = tf.keras.layers.Activation('relu', name='actv2')(self.batc2)
        self.drop2 = tf.keras.layers.SpatialDropout3D(0.5)(self.actv2)
        self.maxp2 = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max2')(self.drop2)

        self.zero3 = tf.keras.layers.ZeroPadding3D(padding=(1, 1, 1), name='zero3')(self.maxp2)
        self.conv3 = tf.keras.layers.Conv3D(96, (3, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal',
                                            name='conv3')(self.zero3)
        self.batc3 = tf.keras.layers.BatchNormalization(name='batc3')(self.conv3)
        self.actv3 = tf.keras.layers.Activation('relu', name='actv3')(self.batc3)
        self.drop3 = tf.keras.layers.SpatialDropout3D(0.5)(self.actv3)
        self.maxp3 = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max3')(self.drop3)

        self.resh1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(self.maxp3)

        #  tf.compat.v1.keras.layers.GRU instead of tf.keras.layers.Dense to resolve compatibility
        #  issues with weights tensorflow 1.10
        self.gru_1 = tf.keras.layers.Bidirectional(
            tf.compat.v1.keras.layers.GRU(256, return_sequences=True, kernel_initializer='Orthogonal', name='gru1'),
            merge_mode='concat')(self.resh1)
        self.gru_2 = tf.keras.layers.Bidirectional(
            tf.compat.v1.keras.layers.GRU(256, return_sequences=True, kernel_initializer='Orthogonal', name='gru2'),
            merge_mode='concat')(self.gru_1)

        # transforms RNN output to character activations:
        self.dense1 = tf.keras.layers.Dense(self.output_size, kernel_initializer='he_normal', name='dense1')(self.gru_2)

        self.y_pred = tf.keras.layers.Activation('softmax', name='softmax')(self.dense1)

        self.labels = tf.keras.Input(name='the_labels', shape=[self.absolute_max_string_len], dtype='float32')
        self.input_length = tf.keras.Input(name='input_length', shape=[1], dtype='int64')
        self.label_length = tf.keras.Input(name='label_length', shape=[1], dtype='int64')

        self.loss_out = CTC('ctc', [self.y_pred, self.labels, self.input_length, self.label_length])

        self.model = tf.keras.Model(inputs=[self.input_data, self.labels, self.input_length, self.label_length],
                                    outputs=[self.dense1, self.loss_out])

    def summary(self):
        self.model.summary()
