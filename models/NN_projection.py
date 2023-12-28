from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, BatchNormalization, InputLayer
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2

def create_model() -> Model:
    activation = 'tanh'
    regularizer = l2(0.001)
    model = Sequential([
        InputLayer(input_shape=(10, 35, 1)),
        BatchNormalization(),
        Conv2D(filters=8, kernel_size=3, strides=2, activation=activation, kernel_regularizer=regularizer),
        Conv2D(filters=8, kernel_size=2, strides=2, activation=activation, kernel_regularizer=regularizer),
        Flatten(),
        Dense(8, activation=activation, kernel_regularizer=regularizer),
        Dense(1, activation='sigmoid', kernel_regularizer=regularizer)
    ])
    # Output a summary of the model's architecture
    print(model.summary())
    # Use a binary crossentropy loss function and an Adam optimizer, and print the accuracy while training
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    # Return the untrained model
    return model
