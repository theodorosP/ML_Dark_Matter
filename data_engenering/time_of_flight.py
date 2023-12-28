import numpy as np
from scipy.optimize import minimize

# Speed of sound in millimeters per second in the CF3I present in the PICO-60 vessel for run 1
SPEED_OF_SOUND = 106_000

# Initial positions of piezos in 3D space, in millimeters, relative to the same origin as bubble positions
INITIAL_PIEZO_POSITIONS = np.array([
    [x1, y1, z1],
    [x2, y2, z2],
    # Add other piezo positions as needed
])

def calculate_expected_times_of_flight(bubble_position, piezo_positions):
    """Calculate the expected times of flight from a bubble at a certain point to each of the piezos."""
    distances = np.linalg.norm(bubble_position - piezo_positions, axis=1)
    return distances / SPEED_OF_SOUND

def timing_error(bubble_position, piezo_timings, piezo_positions):
    """Calculate the mean squared error of the expected timings versus those observed."""
    expected_times = calculate_expected_times_of_flight(bubble_position, piezo_positions)
    squared_errors = (expected_times - piezo_timings) ** 2
    return np.mean(squared_errors)

def localize_bubble(piezo_timings, piezo_positions):
    """Calculate the position of a bubble using error optimization based on piezo timings."""
    # Initial guess for bubble position
    initial_guess = np.zeros(3)

    # Define the objective function for optimization
    objective_function = lambda bubble_position: timing_error(bubble_position, piezo_timings, piezo_positions)

    # Optimize and return the position of the bubble
    result = minimize(objective_function, initial_guess, method='BFGS')
    return result.x

