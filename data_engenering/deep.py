import json
import os
import time
from typing import Tuple, Optional, List

import numpy as np


def save_test(
    ground_truths: np.ndarray,
    network_outputs: np.ndarray,
    events,
    epoch: Optional[int] = None,
    prefix: str = ''
) -> None:
    output_list = []
    for ground_truth, network_output, event in zip(
        ground_truths.tolist(), network_outputs.tolist(), events
    ):
        event_information = {
            'ground_truth': ground_truth,
            'network_output': network_output,
            'identifier': event[2:]
        }
        output_list.append(event_information)
    os.makedirs(os.path.expanduser(f'~/{prefix}'), exist_ok=True)
    json_file_path = os.path.expanduser(f'~/{prefix}/time{int(time.time())}_epoch{epoch}.json')
    with open(json_file_path, 'w') as output_file:
        json.dump(output_list, output_file)
    print('Data saved at', json_file_path)


def load_test(json_file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(os.path.expanduser(json_file_path)) as input_file:
        input_list = json.load(input_file)
    ground_truths = []
    network_outputs = []
    identifiers = []
    for event_information in input_list:
        ground_truths.append(event_information['ground_truth'])
        network_outputs.append(event_information['network_output'])
        identifiers.append(event_information['identifier'])
    return np.array(ground_truths), np.array(network_outputs), np.array(identifiers)
