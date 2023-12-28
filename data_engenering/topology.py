import csv
from typing import List, Optional, Tuple

import numpy as np


class SurfaceTopologyNode:
    def __init__(self, identifier: int, position: Tuple[float, float, float], connections: List[int], values: List[float]) -> None:
        self.identifier = identifier
        self.x_position, self.y_position, self.z_position = position
        self.raw_connections_clockwise = connections
        self.values = values
        self.connections = None

    def set_connections_clockwise(self, connected_nodes_clockwise: List['SurfaceTopologyNode']) -> None:
        y_positions = [node.y_position if node != -1 else float('-inf') for node in connected_nodes_clockwise]
        highest_node_index = np.argmax(y_positions)
        self.connections = [
            self.raw_connections_clockwise[(highest_node_index + index_offset) % len(self.raw_connections_clockwise)]
            for index_offset in range(len(connected_nodes_clockwise))
        ]


class SurfaceTopologySet:
    def __init__(self, csv_path: Optional[str] = None, values: Optional[List[List[float]]] = None,
                 positions: Optional[List[Tuple[float, float, float]]] = None, ground_truths: Optional[List[bool]] = None) -> None:
        self.nodes = []
        self.ground_truths = ground_truths

        if csv_path is not None:
            self.load_from_csv(csv_path, positions, values)

    def load_from_csv(self, csv_path: str, positions: List[Tuple[float, float, float]], values: List[List[float]]) -> None:
        try:
            with open(csv_path) as connection_file:
                connection_reader = csv.reader(connection_file)
                for line, position, node_values in zip(connection_reader, positions, values):
                    line_numeric = [int(string) for string in line]
                    identifier, connections = line_numeric[0], line_numeric[1:]
                    node = SurfaceTopologyNode(identifier, position, connections, node_values)
                    self.nodes.append(node)
        except (FileNotFoundError, PermissionError) as e:
            print(f"Error opening file: {e}")

        self.calculate_connections()

    def calculate_connections(self) -> None:
        for node in self.nodes:
            connected_nodes_clockwise = [self.get_node(node_identifier) if node_identifier != -1 else -1
                                         for node_identifier in node.raw_connections_clockwise]
            node.set_connections_clockwise(connected_nodes_clockwise)

    def get_node(self, identifier: int) -> Optional[SurfaceTopologyNode]:
        return next((node for node in self.nodes if node.identifier == identifier), None)


