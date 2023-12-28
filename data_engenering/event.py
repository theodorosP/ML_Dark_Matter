import os
import pickle
import random
from typing import Callable, List, Optional, Set, Tuple

import numpy as np
from sklearn.externals import joblib

from data_processing.bubble_data_point import BubbleDataPoint, RunType, TriggerCause, load_bubble_audio


RUN_1_PATH = os.path.expanduser('~/run1merged.pkl')
RUN_2_PATH = os.path.expanduser('~/run2alldata.pkl')
VALIDATION_EXAMPLES = 128


class EventDataSet:

    training_initial_input_indices = None
    validation_initial_input_indices = None
    data_from_file_cache = None

    @classmethod
    def load_data_from_file(cls, use_run_1: bool = False) -> List[BubbleDataPoint]:
        if cls.data_from_file_cache is not None:
            return cls.data_from_file_cache

        path = RUN_1_PATH if use_run_1 else RUN_2_PATH
        loader = pickle if use_run_1 else joblib

        with open(path, 'rb') as pickle_file:
            data_list = loader.load(pickle_file)

        cls.data_from_file_cache = data_list
        return data_list

    def __init__(
        self,
        keep_run_types: Optional[Set[RunType]],
        use_wall_cuts: bool,
        use_run_1: bool = False,
        use_temperature_and_pressure_cuts: bool = False
    ) -> None:
        events = self.load_data_from_file(use_run_1)

        if keep_run_types is not None:
            events = [event for event in events if event.run_type in keep_run_types]

        events = [event for event in events if self.passes_standard_cuts(event)]

        if use_wall_cuts:
            events = [event for event in events if self.passes_fiducial_cuts(event) and self.passes_audio_wall_cuts(event)]

        if use_temperature_and_pressure_cuts:
            events = [event for event in events if self.passes_temperature_and_pressure_cuts(event)]

        self.validation_events = random.sample(events, VALIDATION_EXAMPLES)
        events = [event for event in events if event not in self.validation_events]

        random.shuffle(events)
        self.training_events = events

    @staticmethod
    def passes_standard_cuts(event: BubbleDataPoint) -> bool:
        return (
            event.run_type != RunType.GARBAGE
            and event.trigger_cause == TriggerCause.CAMERA_TRIGGER
            and (
                not hasattr(event, 'logarithmic_acoustic_parameter')
                or event.logarithmic_acoustic_parameter > -100
            )
            and event.num_bubbles_image <= 1
            and (
                not hasattr(event, 'num_bubbles_pressure')
                or (event.num_bubbles_pressure >= 0.7 and event.num_bubbles_pressure <= 1.3)
            )
            and event.time_since_target_pressure > 25
            and event.x_position != -100
        )

    @staticmethod
    def passes_fiducial_cuts(event: BubbleDataPoint) -> bool:
        return event.z_position <= 523 and (
            (
                event.distance_to_wall > 6
                and event.z_position > 0
                and event.z_position <= 400
            ) or (
                event.distance_to_wall > 6
                and event.z_position <= 0
                and event.distance_from_center <= 100
            ) or (
                event.distance_to_wall > 13
                and event.distance_from_center > 100
                and event.z_position <= 0
            ) or (
                event.distance_to_wall > 13
                and event.z_position > 400
            )
        )

    @staticmethod
    def passes_audio_wall_cuts(event: BubbleDataPoint) -> bool:
        return (
            event.pressure_not_position_corrected < 1.3
            and event.pressure_not_position_corrected > 0.7
        ) and (
            event.acoustic_parameter_12 < 300
            and event.acoustic_parameter_12 > 45
        )

    @staticmethod
    def passes_temperature_and_pressure_cuts(event: BubbleDataPoint) -> bool:
        return (
            event.pressure_setting == 21
            and np.abs(event.pressure_readings[0] - event.pressure_setting) < 1
            and np.abs(event.temperature_readings[2] - 16.05) < 1
        )

    @classmethod
    def load_specific_indices(cls, specific_unique_indices: List[int]) -> List[BubbleDataPoint]:
        all_data = cls.load_data_from_file()
        return [
            next(bubble for bubble in all_data if bubble.unique_bubble_index == unique_index)
            for unique_index in specific_unique_indices
        ]

    def banded_frequency_alpha_classification(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        (training_inputs, training_ground_truths), (validation_inputs, validation_ground_truths) = [
            (
                np.stack([
                    event.banded_frequency_domain[1:, :, 2].flatten()
                    for event in events
                ]),
                np.array([event.run_type == RunType.LOW_BACKGROUND for event in events])
            )
            for events in [self.training_events, self.validation_events]
        ]
        return training_inputs, training_ground_truths, validation_inputs, validation_ground_truths

    def ap_simulation(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        (training_inputs, training_ground_truths), (validation_inputs, validation_ground_truths) = [
            (
                np.stack([
                    np.concatenate([
                        event.banded_frequency_domain_raw[1:, :, 2].flatten()
                    ])
                    for event in events
                ]),
                np.array([event.logarithmic_acoustic_parameter for event in events])
            )
            for events in [self.training_events, self.validation_events]
        ]
        return training_inputs, training_ground_truths, validation_inputs, validation_ground_truths

    def position_from_time_zero(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        (training_inputs, training_ground_truths), (validation_inputs, validation_ground_truths) = [
            (
                np.stack([
                    event.piezo_time_zero - np.min(event.piezo_time_zero)
                    for event in events
                ]),
                np.stack([
                    [event.x_position, event.y_position, event.z_position]
                    for event in events
                ])
            )
            for events in [self.training_events, self.validation_events]
        ]
        return training_inputs, training_ground_truths, validation_inputs, validation_ground_truths

    def position_from_waveform(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        training_audio_inputs = []
        training_ground_truths = []
        validation_audio_inputs = []
        validation_ground_truths = []

        for events, audio_inputs, ground_truths in zip(
            [self.training_events, self.validation_events],
            [training_audio_inputs, validation_audio_inputs],
            [training_ground_truths, validation_ground_truths]
        ):
            for event in events:
                audio = load_bubble_audio(event)
                if not audio:
                    continue

                audio_inputs += audio
                ground_truths.append([
                    event.x_position,
                    event.y_position,
                    event.z_position
                ])

        return (
            np.array(training_audio_inputs),
            np.array(training_ground_truths),
            np.array(validation_audio_inputs),
            np.array(validation_ground_truths)
        )

    def audio_alpha_classification(self, loading_function: Callable[[BubbleDataPoint], List[np.ndarray]], include_positions: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        training_audio_inputs = []
        training_position_inputs = []
        training_ground_truths = []
        self.training_initial_input_indices = []
        validation_audio_inputs = []
        validation_position_inputs = []
        validation_ground_truths = []
        self.validation_initial_input_indices = []

        for events, audio_inputs, position_inputs, ground_truths, initial_input_indices in zip(
            [self.training_events, self.validation_events],
            [training_audio_inputs, validation_audio_inputs],
            [training_position_inputs, validation_position_inputs],
            [training_ground_truths, validation_ground_truths],
            [self.training_initial_input_indices, self.validation_initial_input_indices]
        ):
            for event in events:
                audio = loading_function(event)
                if not audio:
                    continue

                initial_input_indices.append(len(audio_inputs))
                audio_inputs += audio

                for _ in range(len(audio)):
                    position_inputs.append([
                        event.x_position,
                        event.y_position,
                        event.z_position
                    ])

                ground_truths += [event.run_type == RunType.LOW_BACKGROUND] * len(audio)

        if include_positions:
            return (
                [
                    np.array(training_audio_inputs),
                    np.array(training_position_inputs)
                ],
                np.array(training_ground_truths),
                [
                    np.array(validation_audio_inputs),
                    np.array(validation_position_inputs)
                ],
                np.array(validation_ground_truths)
            )
        else:
            return (
                np.array(training_audio_inputs),
                np.array(training_ground_truths),
                np.array(validation_audio_inputs),
                np.array(validation_ground_truths)
            )
