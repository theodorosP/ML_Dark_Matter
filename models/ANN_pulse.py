from tensorflow.keras.layers import Dense, BatchNormalization, InputLayer, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2

def create_model() -> Model:
    activation = 'relu'
    regularizer = l2(0.001)
    model = Sequential([
        InputLayer(input_shape=(255,)),
        BatchNormalization(),
        Dense(24, activation=activation, kernel_regularizer=regularizer),
        Dropout(0.5),
        Dense(12, activation=activation, kernel_regularizer=regularizer),
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

