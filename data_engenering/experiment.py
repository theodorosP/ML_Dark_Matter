import json
import os
import time
from typing import Tuple, List

import numpy as np

from data_processing.bubble_data_point import BubbleDataPoint
from data_processing.event_data_set import EventDataSet


def save_test(event_data_set: EventDataSet, validation_ground_truths: np.ndarray, validation_network_outputs: np.ndarray, epoch: Optional[int] = None, prefix: str = '') -> None:
    validation_events = [event_data_set.validation_events[event_index] for event_index in range(len(event_data_set.validation_initial_input_indices) - 1) for _ in range(event_data_set.validation_initial_input_indices[event_index + 1] - event_data_set.validation_initial_input_indices[event_index])] if event_data_set.validation_initial_input_indices is not None else event_data_set.validation_events
    output_list = [
        {
            'unique_bubble_index': event.unique_bubble_index,
            'ground_truth': ground_truth,
            'network_output': network_output[0]
        }
        for event, ground_truth, network_output in zip(validation_events, validation_ground_truths.tolist(), validation_network_outputs.tolist())
    ]
    os.makedirs(os.path.expanduser(f'~/{prefix}'), exist_ok=True)
    json_file_path = os.path.expanduser(f'~/{prefix}/time{int(time.time())}_epoch{epoch}.json')
    with open(json_file_path, 'w') as output_file:
        json.dump(output_list, output_file)
    print('Data saved at', json_file_path)


def load_test(json_file_path: str) -> Tuple[List[BubbleDataPoint], np.ndarray, np.ndarray]:
    with open(os.path.expanduser(json_file_path)) as input_file:
        input_list = json.load(input_file)
    unique_bubble_indices, ground_truths, network_outputs = zip(*((
        bubble_information['unique_bubble_index'],
        bubble_information['ground_truth'],
        bubble_information['network_output']
    ) for bubble_information in input_list))
    events = EventDataSet.load_specific_indices(unique_bubble_indices)
    return events, np.array(ground_truths), np.array(network_outputs)
