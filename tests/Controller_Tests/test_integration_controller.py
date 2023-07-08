def test_controller_to_interface_vectors():
    # Initialization parameters
    input_size = 100
    hidden_size = 50
    num_layers = 1
    output_size = 100
    memory_N = 20
    memory_M = 10
    num_heads = 2
    levels = 3
    controller_size = 100

    # Initialize Controller and InterfaceVectors modules
    controller = Controller(input_size, hidden_size, num_layers, output_size)
    memory = Memory(memory_N, memory_M, controller_size, levels)
    iv = InterfaceVectors(controller_size, memory_N, memory_M, num_heads, memory)

    # Simulate some input
    x = torch.randn(1, 10, input_size)

    # Run Controller
    controller_output, _ = controller(x)

    # Get the current memory state
    current_memory = memory.get_current_memory()

    # Run InterfaceVectors
    read_data_combined, write_ws, erase_vectors, add_vectors = iv(controller_output, current_memory)

    # Assert that the outputs of the InterfaceVectors are the correct size
    assert all([data.size() == (1, memory_N) for data in write_ws])  # write weights
    assert all([data.size() == (1, memory_M) for data in erase_vectors])  # erase vectors
    assert all([data.size() == (1, memory_M) for data in add_vectors])  # add vectors
    assert all([data.size() == (1, memory_M) for data in read_data_combined])  # read vectors

    # Assert that the weights are normalized (i.e., sum to 1)
    assert all([torch.isclose(torch.sum(w), torch.tensor(1.0), atol=1e-5) for w in write_ws])