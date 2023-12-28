from tensorflow.keras.layers import Dense, BatchNormalization, InputLayer
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

def create_time_zero_model():
    activation = 'tanh'
    model = Sequential([
        InputLayer(input_shape=(9,)),
        BatchNormalization(),
        Dense(24, activation=activation),
        Dense(12, activation=activation),
        Dense(6, activation=activation),
        Dense(3)
    ])
    # Output a summary of the model's architecture
    print(model.summary())
    # Use a mean squared error loss function and an Adam optimizer, and print the accuracy while training
    model.compile(
        optimizer=Adam(lr=0.0001),
        loss='mse'
    )
    # Return the untrained model
    return model
