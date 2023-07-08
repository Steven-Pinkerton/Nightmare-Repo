def test_hierarchical_memory():
    N, M, controller_size, levels = 10, 20, 5, 3
    hier_mem = HierarchicalMemory(N, M, controller_size, levels)

    # Check initial state
    assert isinstance(hier_mem.memory_levels, nn.ModuleList)
    assert len(hier_mem.memory_levels) == levels
    for level in hier_mem.memory_levels:
        assert isinstance(level, DynamicMemory)
        assert level.N == N
        assert level.M == M

    assert isinstance(hier_mem.tlm_levels, nn.ModuleList)
    assert len(hier_mem.tlm_levels) == levels
    for level in hier_mem.tlm_levels:
        assert isinstance(level, TemporalLinkageMatrix)
        assert level.N == N

    # Define inputs
    controller_output = torch.randn((1, controller_size))
    read_ws = [torch.randn((N)) for _ in range(levels)]
    write_ws = [torch.randn((N)) for _ in range(levels)]
    erase_vectors = [torch.randn((M)) for _ in range(levels)]
    add_vectors = [torch.randn((M)) for _ in range(levels)]

    # Execute forward pass
    controller_output, read_vectors_list, forward_weights_list, backward_weights_list = hier_mem.forward(
        controller_output, read_ws, write_ws, erase_vectors, add_vectors
    )

    # Validate outputs
    assert controller_output.size() == (1, controller_size + M * levels)
    assert len(read_vectors_list) == levels
    assert len(forward_weights_list) == levels
    assert len(backward_weights_list) == levels

    for i in range(levels):
        assert read_vectors_list[i].size() == (1, M)
        assert forward_weights_list[i].size() == (1, N)
        assert backward_weights_list[i].size() == (1, N)