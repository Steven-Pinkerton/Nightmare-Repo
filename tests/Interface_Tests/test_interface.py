import unittest
import torch.nn as nn
from your_module import InterfaceVectors  # replace with the actual import
from your_module import Memory  # replace with the actual import

class TestInterfaceVectors(unittest.TestCase):
    def setUp(self):
        self.controller_size = 100
        self.memory_N = 128
        self.memory_M = 20
        self.num_heads = 1
        self.memory = Memory(self.memory_N, self.memory_M, self.controller_size)
        self.interface_vectors = InterfaceVectors(self.controller_size, self.memory_N, self.memory_M, self.num_heads, self.memory)


    def test_calculate_interface_vector_size(self):
        expected_size = self.memory_M * 2 + self.memory_N * 3 + 5
        actual_size = self.interface_vectors.calculate_interface_vector_size()
        self.assertEqual(actual_size, expected_size, "Size of the interface vector is not as expected.")
        
    def test_split_interface_vector(self):
        iv_size = self.interface_vectors.calculate_interface_vector_size()
        iv = torch.randn(1, iv_size)
        read_vectors, write_vectors = self.interface_vectors.split_interface_vector(iv, self.num_heads)
        
        # Check the size and number of read and write vectors
        self.assertEqual(len(read_vectors), self.num_heads, "Number of read vectors is not as expected.")
        self.assertEqual(len(write_vectors), self.num_heads, "Number of write vectors is not as expected.")
        for rv in read_vectors:
            self.assertTrue(all([rv_i.shape == (1, self.memory_M) for rv_i in rv]), "Shape of read vectors is not as expected.")
        for wv in write_vectors:
            self.assertTrue(all([wv_i.shape == (1, self.memory_M) for wv_i in wv]), "Shape of write vectors is not as expected.")
            
    def test_forward(self):
        # Create some input
        x = torch.randn(1, self.controller_size)

        # Since forward method also uses the current memory, let's create some
        current_memory = torch.randn(1, self.memory_N, self.memory_M)

        read_data_combined, write_ws, erase_vectors, add_vectors = self.interface_vectors.forward(x, current_memory)

        # check if the read_data_combined is list and has same length as number of heads
        self.assertTrue(isinstance(read_data_combined, list), "Expected type of read_data_combined is list.")
        self.assertEqual(len(read_data_combined), self.num_heads, "The length of read_data_combined does not match with num_heads.")

        # check if write_ws, erase_vectors, and add_vectors are lists and have the same length
        for name, var in zip(["write_ws", "erase_vectors", "add_vectors"], [write_ws, erase_vectors, add_vectors]):
            self.assertTrue(isinstance(var, list), f"Expected type of {name} is list.")
            self.assertEqual(len(var), self.num_heads, f"The length of {name} does not match with num_heads.")
            
    def test_addressing(self):
        # Create some inputs
        k = torch.randn(self.memory_N, self.memory_M)
        beta = torch.randn(self.memory_N, self.memory_M)
        g = torch.randn(self.memory_N, self.memory_M)
        s = torch.randn(self.memory_N, self.memory_M)
        gamma = torch.randn(self.memory_N, self.memory_M)

        w = self.interface_vectors.addressing(k, beta, g, s, gamma)

        # Check if the output w is a tensor and has the same shape as inputs
        self.assertTrue(isinstance(w, torch.Tensor), "Expected output of addressing method is torch.Tensor.")
        self.assertTupleEqual(w.shape, k.shape, "The shape of output tensor does not match with the shape of input tensors.")