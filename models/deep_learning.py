from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dropout, Input, BatchNormalization, Dense, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


def create_custom_model():
    activation = 'relu'
    regularizer = l2(0.0001)
    dropout_rate = 0.4

    audio_inputs = Input((100_000, 2))
    x = BatchNormalization()(audio_inputs)

    x = Conv1D(filters=64, kernel_size=80, strides=4, activation=activation, kernel_regularizer=regularizer, padding='valid')(x)
    x = MaxPooling1D(4)(x)
    x = BatchNormalization()(x)

    for filters in [128, 256, 512]:
        x = Conv1D(filters=filters, kernel_size=3, activation=activation, kernel_regularizer=regularizer, padding='same')(x)
        x = MaxPooling1D(2)(x)
        x = BatchNormalization()(x)

    x = Flatten()(x)

    axes_inputs = Input((3,))
    x = concatenate([x, axes_inputs])
    x = BatchNormalization()(x)

    x = Dense(256, activation=activation, kernel_regularizer=regularizer)(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(128, activation=activation, kernel_regularizer=regularizer)(x)
    x = Dropout(dropout_rate)(x)

    # Output layer
    outputs = Dense(1, activation='sigmoid', kernel_regularizer=regularizer)(x)

    model = Model(inputs=[audio_inputs, axes_inputs], outputs=outputs)

    print(model.summary())

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['accuracy']
    )

    return model

