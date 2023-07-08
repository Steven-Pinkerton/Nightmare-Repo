import torch
import pytest
from src.controller import Controller  # Make sure to adjust this import to your project structure
import unittest
import torch
import torch.nn as nn

class TestController(unittest.TestCase):
    def setUp(self):
        self.input_size = 100
        self.hidden_size = 50
        self.num_layers = 1
        self.output_size = 100
        self.controller = Controller(self.input_size, self.hidden_size, self.num_layers, self.output_size)

    def test_lstm_processing(self):
        x = torch.randn(1, 10, self.input_size)
        lstm_output, state = self.controller(x)
        self.assertEqual(lstm_output.size(), (1, 10, self.hidden_size))
        self.assertEqual(state[0].size(), (self.num_layers, 1, self.hidden_size))  # hidden state
        self.assertEqual(state[1].size(), (self.num_layers, 1, self.hidden_size))  # cell state

    def test_fc_mapping(self):
        x = torch.randn(1, 10, self.input_size)
        lstm_output, _ = self.controller(x)
        controller_output = self.controller.fc(lstm_output[:, -1, :])
        self.assertEqual(controller_output.size(), (1, self.output_size))

    def test_update_controller_output(self):
        x = torch.randn(1, 10, self.input_size)
        read_vectors = [torch.randn(1, self.input_size) for _ in range(self.input_size)]
        controller_output, _ = self.controller(x)
        updated_output = self.controller.update_controller_output(controller_output, read_vectors)
        self.assertEqual(updated_output.size(), (1, self.output_size))

if __name__ == '__main__':
    unittest.main()