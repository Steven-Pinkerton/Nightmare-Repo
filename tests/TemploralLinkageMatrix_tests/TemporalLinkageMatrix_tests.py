
## Test Inialization
def test_initialization(self):
    N = 10
    tlm = TemporalLinkageMatrix(N)
    self.assertEqual(tlm.matrix.shape, (N, N))
    self.assertEqual(tlm.precedence_weight.shape, (N,))
    self.assertTrue(torch.equal(tlm.matrix, torch.zeros(N, N)))
    self.assertTrue(torch.equal(tlm.precedence_weight, torch.zeros(N)))
    
## Test Update
def test_update(self):
    N = 10
    tlm = TemporalLinkageMatrix(N)
    w = torch.ones(N) / N  # Create a uniform distribution over the slots
    tlm.update(w)
    self.assertTrue(torch.allclose(tlm.matrix, torch.zeros(N, N), atol=1e-4))
    self.assertTrue(torch.allclose(tlm.precedence_weight, w, atol=1e-4))

## Test Read Weights
def test_read(self):
    N = 10
    tlm = TemporalLinkageMatrix(N)
    w = torch.ones(N) / N
    wf, wb = tlm.read(w)
    self.assertTrue(torch.allclose(wf, torch.zeros(N), atol=1e-4))
    self.assertTrue(torch.allclose(wb, torch.zeros(N), atol=1e-4))
    
## Test Memory Allocation
def test_allocation(self):
    N = 10
    tlm = TemporalLinkageMatrix(N)
    tlm.allocate()
    self.assertEqual(tlm.matrix.shape, (N+1, N+1))
    self.assertEqual(tlm.precedence_weight.shape, (N+1,))
    
## Test Memory Deallocation
def test_deallocation(self):
    N = 10
    tlm = TemporalLinkageMatrix(N)
    tlm.deallocate(0)  # Remove the first memory slot
    self.assertEqual(tlm.matrix.shape, (N-1, N-1))
    self.assertEqual(tlm.precedence_weight.shape, (N-1,))
    
## Test Reset
def test_reset(self):
    N = 10
    tlm = TemporalLinkageMatrix(N)
    w = torch.ones(N) / N
    tlm.update(w)
    tlm.reset()
    self.assertTrue(torch.equal(tlm.matrix, torch.zeros(N, N)))
    self.assertTrue(torch.equal(tlm.precedence_weight, torch.zeros(N)))
    
    
if __name__ == '__main__':
    unittest.main()
