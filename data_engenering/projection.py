import numpy as np
from scipy.ndimage import zoom
from data_processing.pmt_positions import X_POSITIONS, Y_POSITIONS, Z_POSITIONS

def pmt_map_projection(pulse_counts: np.ndarray) -> np.ndarray:
    row_positions = np.unique(Z_POSITIONS)
    
    values_by_row = [[] for _ in range(len(row_positions))]
    for pulse_count, x_position, y_position, z_position in zip(pulse_counts, X_POSITIONS, Y_POSITIONS, Z_POSITIONS):
        row_index = np.where(row_positions == z_position)[0][0]
        values_by_row[row_index].append((pulse_count, x_position, y_position))
    
    largest_row_size = max(len(row) for row in values_by_row)
    map_image = np.zeros(shape=(largest_row_size, len(values_by_row)), dtype=int)
    
    for row_index, row in enumerate(values_by_row):
        angles = [np.angle(x + (y * 1j)) for _, x, y in row]
        sorted_indices = np.argsort(angles)
        sorted_pulse_counts = np.array([count for count, _, _ in row])[sorted_indices]
        
        if len(sorted_pulse_counts) == largest_row_size:
            map_image[:, row_index] = sorted_pulse_counts
        else:
            map_image[:, row_index] = zoom(sorted_pulse_counts, largest_row_size / len(sorted_pulse_counts))
    
    map_image = np.expand_dims(map_image, axis=-1)
    
    return map_image
