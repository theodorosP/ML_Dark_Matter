from tensorflow.keras.layers import concatenate, Dense, Input
from tensorflow.keras.models import Model
import numpy as np
import copy
from typing import List, Dict, Union, Optional

class CustomTopologicalCNN:

    def __init__(self, surface_topology_set, convolutional_layers, remaining_model, optimizer, loss, epochs, validation_size, class_weight):
        # Placeholder for input tensors
        for node in surface_topology_set.nodes:
            node.tensor = Input(shape=(1,))
        input_layers = [node.tensor for node in surface_topology_set.nodes]
        layer_nodes = surface_topology_set.nodes

        for convolutional_layer in convolutional_layers:
            layer_nodes = self.convolve_surface_topology(layer_nodes, **convolutional_layer)

        combined_tensor = concatenate([node.tensor for node in layer_nodes])
        output = remaining_model(combined_tensor)

        model = Model(inputs=input_layers, outputs=output)
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        inputs = [list(node.values) for node in surface_topology_set.nodes]
        input_permutation = np.random.permutation(len(inputs[0]))
        inputs = np.array([[input_list[index] for index in input_permutation] for input_list in inputs])
        ground_truths = np.array([surface_topology_set.ground_truths[index] for index in input_permutation])
        validation_inputs, training_inputs = np.split(inputs, [validation_size], axis=1)
        validation_ground_truths, training_ground_truths = np.split(ground_truths, [validation_size])

        validation_inputs = [array for array in validation_inputs]
        training_inputs = [array for array in training_inputs]

        performance_statistics = []

        for epoch in range(epochs):
            print('Epoch', epoch)
            model.fit(training_inputs, training_ground_truths, validation_data=(validation_inputs, validation_ground_truths), class_weight=class_weight)
            validation_predictions = model.predict(validation_inputs)[:, 0]

            if epoch >= epochs - 10:
                performance_statistics.append(evaluate_predictions(validation_ground_truths, validation_predictions, None, epoch, set_name='validation'))

        statistics_mean = np.mean(np.stack(performance_statistics, axis=0), axis=0)
        true_positives, true_negatives, false_positives, false_negatives = statistics_mean
        neck_alphas_removed = true_positives / (true_positives + false_negatives)
        nuclear_recoils_removed = false_positives / (false_positives + true_negatives)
        print('Neck alphas removed:', neck_alphas_removed)
        print('Nuclear recoils removed:', nuclear_recoils_removed)

    @classmethod
    def convolve_surface_topology(cls, surface_topology_nodes, kernel_radius, filters, activation, regularizer):
        filters_layer = Dense(filters, activation=activation, kernel_regularizer=regularizer)
        modified_nodes = []

        for node in surface_topology_nodes:
            kernel = cls.form_kernel(node, surface_topology_nodes, kernel_radius)

            if kernel is None:
                continue

            tensors = [node.tensor for node in kernel]
            combined_tensor = concatenate(tensors)

            node_copy = copy.copy(node)
            node_copy.tensor = filters_layer(combined_tensor)

            modified_nodes.append(node_copy)

        return modified_nodes

    @staticmethod
    def form_kernel(node, searchable_nodes, radius):
        nodes = []

        def traverse_node_tree(search_node, depth):
            if search_node is None:
                return False

            nodes.append(search_node)

            if depth == 0:
                return True

            searchable_identifiers = [searchable_node.identifier for searchable_node in searchable_nodes]
            connected_nodes = []

            for connection in search_node.connections:
                if connection not in searchable_identifiers:
                    connected_nodes.append(None)
                else:
                    connected_nodes.append([possible_node for possible_node in searchable_nodes if possible_node.identifier == connection][0])

            for connected_node in connected_nodes:
                success = traverse_node_tree(connected_node, depth - 1)

                if not success:
                    return False

            return True

        success = traverse_node_tree(node, radius)

        if not success:
            return None

        return nodes
