from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, InputLayer
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2

def create_banded_frequency_model():
    regularizer = l2(0.001)
    dropout = 0.25
    activation = 'tanh'
    model = Sequential([
        InputLayer(input_shape=(16,)),
        BatchNormalization(),
        Dense(12, activation=activation, kernel_regularizer=regularizer),
        Dropout(dropout),
        Dense(8, activation=activation, kernel_regularizer=regularizer),
        Dropout(dropout),
        Dense(1, activation='sigmoid', kernel_regularizer=regularizer)
    ])
    # Output a summary of the model's architecture
    print(model.summary())
    # Use a mean squared error loss function and an Adam optimizer, and print the accuracy while training
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['accuracy']
    )
    # Return the untrained model
    return model
