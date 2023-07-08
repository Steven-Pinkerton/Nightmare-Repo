
## Test that memory is correctly initalized.
def test_memory_initialization(self):
    dm = DynamicMemory(10, 5, 20)
    self.assertEqual(dm.memory.shape, (10, 5))
    self.assertAlmostEqual(torch.mean(dm.memory).item(), 0, delta=0.1)
    
    
## Test that memory allocation works correctly.
def test_memory_allocation(self):
    dm = DynamicMemory(10, 5, 20)
    old_N = dm.N
    dm._allocate_memory()
    self.assertEqual(dm.N, old_N + 1)
    self.assertEqual(dm.memory.shape, (old_N + 1, 5))
    self.assertEqual(dm.linkage_matrix.matrix.shape, (old_N + 1, old_N + 1))
    
## Test that memory deallocation works correctly.
def test_memory_deallocation(self):
    dm = DynamicMemory(10, 5, 20)
    old_N = dm.N
    dm._deallocate_memory()
    self.assertEqual(dm.N, old_N - 1)
    self.assertEqual(dm.memory.shape, (old_N - 1, 5))
    self.assertEqual(dm.linkage_matrix.matrix.shape, (old_N - 1, old_N - 1))
    
## Test the read method.
def test_read(self):
    dm = DynamicMemory(10, 5, 20)
    ws = torch.ones(10) / 10  # Uniform read weights
    read_vectors, _ = dm.read(ws)
    self.assertEqual(read_vectors.shape, (10, 5))
    self.assertAlmostEqual(torch.mean(read_vectors).item(), 0, delta=0.1)
    
## Test the resert method.
def test_reset(self):
    dm = DynamicMemory(10, 5, 20)
    dm.reset()
    self.assertTrue(torch.equal(dm.memory, torch.zeros(10, 5)))
    self.assertTrue(torch.equal(dm.linkage_matrix.matrix, torch.zeros(10, 10)))
    self.assertTrue(torch.equal(dm.linkage_matrix.precedence_weight, torch.zeros(10)))
    
