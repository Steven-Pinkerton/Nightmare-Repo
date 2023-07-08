class TestMemory(unittest.TestCase):
    def setUp(self):
        self.memory = Memory(N=10, M=10, controller_size=10, levels=3)

        # Mock the output from InterfaceVectors class for testing
        self.read_ws = torch.rand(3, 10)
        self.write_ws = torch.rand(3, 10)
        self.erase_vectors = torch.rand(3, 10, 10)
        self.add_vectors = torch.rand(3, 10, 10)

    def test_write(self):
        # Capture the memory state before write operation
        pre_dynamic_memory = self.memory.dynamic_memory.memory.detach().clone()
        pre_hierarchical_memory = [level.memory.detach().clone() for level in self.memory.hierarchical_memory.memory_levels]
        pre_linkage_matrix = self.memory.temporal_linkage_matrix.matrix.detach().clone()

        # Perform the write operation
        self.memory.write(self.read_ws, self.write_ws, self.erase_vectors, self.add_vectors)

        # Test that the dynamic memory has changed
        self.assertTrue(torch.all(pre_dynamic_memory != self.memory.dynamic_memory.memory))

        # Test that all levels of the hierarchical memory have changed
        for i, level in enumerate(self.memory.hierarchical_memory.memory_levels):
            self.assertTrue(torch.all(pre_hierarchical_memory[i] != level.memory))

        # Test that the temporal linkage matrix has changed
        self.assertTrue(torch.all(pre_linkage_matrix != self.memory.temporal_linkage_matrix.matrix))

        # If possible, add tests to verify that the new memory states are correct.
        # This would involve manually calculating what the new states should be and comparing these expected states to the actual new states.

if __name__ == "__main__":
    unittest.main()