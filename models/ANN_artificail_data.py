from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, InputLayer
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2

def create_acoustic_parameter_model():
    """Create and return a new instance of the fully connected network for Acoustic Parameter simulation"""
    # Create a neural network model that includes several dense layers with hyperbolic tangent activations, L2 regularization, and batch normalization
    regularizer = l2(0)
    dropout = 0
    activation = 'tanh'
    model = Sequential([
        InputLayer(input_shape=(16,)),
        BatchNormalization(),
        Dense(12, activation=activation, kernel_regularizer=regularizer),
        Dropout(dropout),
        Dense(8, activation=activation, kernel_regularizer=regularizer),
        Dropout(dropout),
        Dense(1, kernel_regularizer=regularizer)
    ])
    # Output a summary of the model's architecture
    print(model.summary())
    # Use a mean squared error loss function and an Adam optimizer; do not print accuracy because this is a regression task
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    # Return the untrained model
    return model
