import os
from typing import List, Tuple
import numpy as np
import joblib


def load_deap_data(file_path: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    with open(os.path.expanduser(file_path), 'rb') as file:
        return joblib.load(file)


def load_simulated_deap_data() -> List[Tuple[np.ndarray, np.ndarray]]:
    return load_deap_data('file1.pkl')


def load_real_world_deap_data() -> List[Tuple[np.ndarray, np.ndarray]]:
    return load_deap_data('file1.pkl')


