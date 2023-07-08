import torch
from torch import nn

class Controller(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Controller, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True)
        
        # Add a fully connected layer that maps the LSTM's hidden states to the output size
        self.fc = nn.Linear(hidden_size, output_size)

        # You might need to adjust the output size if you're concatenating the read vectors to the output
        # If so, add another linear layer here to adjust the size of the updated output
        # This example assumes that the read vectors and controller output are both of size `output_size`
        self.fc_updated = nn.Linear(2*output_size, output_size)

    def forward(self, x):
        lstm_output, state = self.lstm(x)  # Return the LSTM state as well
        output = self.fc(lstm_output[:, -1, :])
        return output, state  # Return both the output and the state
    
    def update_controller_output(self, controller_output, read_vectors):
        # Flatten the read vectors into a single vector per batch
        read_vectors_flattened = torch.cat(read_vectors, dim=-1)
        # Concatenate the controller output and read vectors along the last dimension
        combined = torch.cat([controller_output, read_vectors_flattened], dim=-1)
        updated_output = self.fc_updated(combined)
        return updated_output
    
class Memory(nn.Module):
    def __init__(self, N, M, controller_size, levels=3):
        super(Memory, self).__init__()

        self.N = N  # Number of memory slots
        self.M = M  # Size of each memory slot

        # Dynamic memory typically represents a fast-access, quickly changing memory component
        self.dynamic_memory = DynamicMemory(N, M, controller_size)
        # Hierarchical memory usually represents a slower, more stable memory structure
        self.hierarchical_memory = HierarchicalMemory(N, M, controller_size, levels)
        # Temporal Linkage Matrix usually maintains information about temporal order of writes
        self.temporal_linkage_matrix = TemporalLinkageMatrix(N)

    # Read operation from the hierarchical memory
    def read(self, read_ws):
        return self.hierarchical_memory.read(read_ws)

    # Write operation into both the dynamic and hierarchical memories, and update the temporal linkage matrix
    def write(self, read_ws, write_ws, erase_vectors, add_vectors):
        self.dynamic_memory.write(write_ws, erase_vectors, add_vectors)
        self.hierarchical_memory(write_ws, erase_vectors, add_vectors, read_ws)
        self.temporal_linkage_matrix.update(ws)

    # Reset both types of memory and the temporal linkage matrix
    def reset(self):
        self.dynamic_memory.reset()
        self.hierarchical_memory.reset()
        self.temporal_linkage_matrix.reset()
        
    # Return the current state of the temporal linkage matrix
    def get_linkage_matrix(self):
        return self.temporal_linkage_matrix.get_linkage_matrix()

    # Read from the temporal linkage matrix given the read weights
    def read_linkage_matrix(self, read_weights):
        return self.temporal_linkage_matrix.read_linkage_matrix(read_weights)

    # Forward operation applies the controller output to the dynamic memory
    def forward(self, controller_output):
        return self.dynamic_memory(controller_output)

class InterfaceVectors(nn.Module):
    def __init__(self, controller_size, memory_N, memory_M, num_heads, memory):
        super(InterfaceVectors, self).__init__()
        
        # controller_size is the output size of the controller
        # memory_N is the number of memory locations
        # memory_M is the size of each memory location

        self.controller_size = controller_size
        self.memory_N = memory_N
        self.memory_M = memory_M

        # Initialize a linear layer to transform the controller's output into the interface vectors
        self.linear = nn.Linear(controller_size, self.calculate_interface_vector_size())

        self.num_heads = num_heads
        self.memory = memory  # Memory is now passed during initialization and stored as an attribut

    def calculate_interface_vector_size(self):
        # The size of the interface vectors is determined by the NTM's specifications.
        # It includes the read keys (k_read), write key (k_write), erase vector (e),
        # add vector (a), read strengths (beta_read), write strength (beta_write), 
        # read modes (pi), write gate (g_write), allocation gate (g_alloc), and free gates (f).
        return self.memory_M * 2 + self.memory_N * 3 + 5

    def forward(self, x, current_memory):
        iv = self.linear(x)
        read_vectors, write_vectors = self.split_interface_vector(iv, num_heads=self.num_heads)
        
        
        # Calculate write weights
        write_ws = [self.addressing(*w, current_memory) for w in write_vectors] # Now addressing needs current memory

        # Perform write operation: return the necessary parameters instead
        erase_vectors = [e for e, _ in write_vectors]
        add_vectors = [a for _, a in write_vectors]
        # return write_ws, erase_vectors, add_vectors

        # Calculate read weights
        read_ws = [self.addressing(*w, current_memory) for w in read_vectors]

        # Read operation
        read_data = self.memory.read(read_ws)
        
        # Get the forward weights from the Temporal Linkage Matrix
        forward_weights, _ = self.memory.read_linkage_matrix(read_ws)  # returns forward and backward weights

        # Now combine the read weights with the forward (and optionally, backward) weights from TLM
        alpha, beta = 0.5, 0.5  # These can be set manually or learned during training
        read_data_combined = [alpha * r + beta * f for r, f in zip(read_data, forward_weights)]  # Combine weights

        # Return both the combined read data and the parameters for the write operation
        return read_data_combined, write_ws, erase_vectors, add_vectors

    def split_interface_vector(self, iv, num_heads):
        # Split the interface vector into separate vectors for each head
        read_vectors = []
        write_vectors = []
        for _ in range(num_heads):
            k, beta, g, s, gamma, e, a = iv[:, :self.M], iv[:, self.M:2*self.M], iv[:, 2*self.M], iv[:, 2*self.M+1:2*self.M+3], iv[:, 2*self.M+3], iv[:, 2*self.M+4:3*self.M+4], iv[:, 3*self.M+4:]
            read_vectors.append((k, beta, g, s, gamma))
            write_vectors.append((e, a))
        return read_vectors, write_vectors  # Return a list of read vectors
    
    def addressing(self, k, beta, g, s, gamma):
        # Apply content-based addressing
        wc = self.content_addressing(k, beta)
        
        # Apply location-based addressing
        wg = self.location_addressing(g, s)
        
        # Interpolate between content and location-based addressing
        w = g * wc + (1 - g) * wg
        
        # Apply sharpening
        w = self.sharpen(w, gamma)
        
        return w
    
    def content_addressing(self, k, beta, current_memory):
        # Compute cosine similarity
        cos_sim = F.cosine_similarity(current_memory + 1e-16, k + 1e-16, dim=-1)
        # Apply softmax to get content addressing weights
        wc = F.softmax(beta * cos_sim, dim=-1)
        return wc

    def location_addressing(self, g, s):
        # Perform circular convolution to get location addressing weights
        wg = torch.conv1d(g.view(1, 1, -1), s.view(1, 1, -1), padding=s.size(1) // 2).view(-1)
        return wg

    def sharpen(self, w, gamma):
        # Perform sharpening
        w = w ** gamma
        # Re-normalize weights
        w = torch.div(w, torch.sum(w, dim=-1).unsqueeze(-1) + 1e-16)
        return w
    
class TemporalLinkageMatrix(nn.Module):
    def __init__(self, N):
        super(TemporalLinkageMatrix, self).__init__()

        # N is the number of memory locations
        self.N = N

        # Initialize the temporal linkage matrix and the precedence weight vector
        # Initialize the temporal linkage matrix and the precedence weight vector
        self.matrix = nn.Parameter(torch.zeros(N, N), requires_grad=False)
        self.precedence_weight = nn.Parameter(torch.zeros(N), requires_grad=False)

    def update(self, w):
        w = w.unsqueeze(1)  # reshape to column vector
        self.precedence_weight = (1 - torch.sum(w, dim=0)) * self.precedence_weight + w
        self.matrix = (1 - w - w.t()) * self.matrix + torch.matmul(w, self.precedence_weight.t())
        self.matrix *= 1 - torch.eye(self.N)

    def read(self, w):
        # `w` is the current read weight vector

        # Forward and backward read weights
        wf = torch.matmul(w, self.matrix)
        wb = torch.matmul(w, self.matrix.t())

        return wf, wb
    
    def allocate(self):
        # Add a row and a column to the linkage matrix with zeros
        self.matrix = nn.Parameter(torch.cat([self.matrix, torch.zeros(1, self.N)], dim=0))
        self.matrix = nn.Parameter(torch.cat([self.matrix, torch.zeros(self.N+1, 1)], dim=1))
        self.N += 1
        # Add an element to the precedence weight vector with zero
        self.precedence_weight = nn.Parameter(torch.cat([self.precedence_weight, torch.zeros(1)], dim=0))
    
    def deallocate(self, index):
        # Remove the row and the column from the linkage matrix
        self.matrix = nn.Parameter(torch.cat([self.matrix[:index], self.matrix[index+1:]], dim=0))
        self.matrix = nn.Parameter(torch.cat([self.matrix[:, :index], self.matrix[:, index+1:]], dim=1))
        self.N -= 1
        # Remove the element from the precedence weight vector
        self.precedence_weight = nn.Parameter(torch.cat([self.precedence_weight[:index], self.precedence_weight[index+1:]], dim=0))
    
    def reset(self):
        self.matrix.data.fill_(0)
        self.precedence_weight.data.fill_(0)
        
    def allocate_new_memory_slot(self):
        # Add a new entry to the precedence weight vector
        self.precedence_weight = nn.Parameter(torch.cat([self.precedence_weight, torch.zeros(1)]), requires_grad=False)
        
        N = self.matrix.shape[0]

        # Expand the linkage matrix with zeros for the new memory slot
        self.matrix = nn.Parameter(torch.cat([self.matrix, torch.zeros(N, 1)], dim=1), requires_grad=False)  # Add a column
        self.matrix = nn.Parameter(torch.cat([self.matrix, torch.zeros(1, N+1)], dim=0), requires_grad=False)  # Add a row

    def deallocate_memory_slot(self, index):
        # Remove the corresponding entry from the precedence weight vector
        self.precedence_weight = nn.Parameter(torch.cat([self.precedence_weight[:index], self.precedence_weight[index+1:]]), requires_grad=False)
        
        # Remove the corresponding row and column from the linkage matrix
        self.matrix = nn.Parameter(torch.cat([self.matrix[:index], self.matrix[index+1:]], dim=0), requires_grad=False)  # Remove a row
        self.matrix = nn.Parameter(torch.cat([self.matrix[:,:index], self.matrix[:,index+1:]], dim=1), requires_grad=False)  # Remove a column

class DynamicMemory(nn.Module):
    def __init__(self, N, M, controller_size):
        super(DynamicMemory, self).__init__()
        
        self.levels = levels
        self.controller = controller  # Add this line
        
        # Create a DynamicMemory instance for each level
        self.memory_levels = nn.ModuleList([DynamicMemory(N, M, controller.controller_size) for _ in range(levels)])

        self.N = N
        self.M = M
        self.controller_size = controller_size

        # Initialize memory matrix with small random values
        self.memory = nn.Parameter(torch.randn(N, M) * 0.01)

        # Initialize allocation gate
        self.alloc_gate = nn.Sequential(
            nn.Linear(controller_size, 1),
            nn.Sigmoid()
        )

        # Initialize deallocation gate
        self.dealloc_gate = nn.Sequential(
            nn.Linear(controller_size, 1),
            nn.Sigmoid()
        )
        
        # Initialize TemporalLinkageMatrix
        self.linkage_matrix = TemporalLinkageMatrix(N)

    def forward(self, controller_output):
        # Get the allocation and deallocation decisions
        alloc_decision = self.alloc_gate(controller_output)
        dealloc_decision = self.dealloc_gate(controller_output)
        
        # Update controller output with the read vectors from this level
        controller_output = self.controller.update_controller_output(controller_output, read_vectors)

        # Allocate or deallocate memory based on the decisions
        if alloc_decision > 0.5:
            self._allocate_memory()
        elif dealloc_decision > 0.5:
            self._deallocate_memory()

        # Update linkage matrix with the current write weights
        self.linkage_matrix.update(controller_output)

        # Return updated memory and linkage matrix
        return self.memory, self.linkage_matrix
        
    def read_linkage_matrix(self, read_weights):
        return self.linkage_matrix.read(read_weights)

    def update_linkage_matrix(self, write_weights):
        self.linkage_matrix.update(write_weights)

    def get_linkage_matrix(self):
        return self.linkage_matrix.matrix
    
    def write(self, ws, es, as_):
        self.memory = self.memory * (1 - ws.unsqueeze(-1) * es) + ws.unsqueeze(-1) * as_
        self.linkage_matrix.update(ws)


    def _allocate_memory(self):
         # Add a row to the memory matrix with small random values
         self.memory = nn.Parameter(torch.cat([self.memory, torch.randn(1, self.M) * 0.01], dim=0))
         self.N += 1
         # Update the linkage matrix accordingly
         self.linkage_matrix.allocate_new_memory_slot()

    def _deallocate_memory(self):
         # Remove the memory slot with the smallest norm
         norms = torch.norm(self.memory, dim=1)
         _, min_index = torch.min(norms, dim=0)
         self.memory = nn.Parameter(torch.cat([self.memory[:min_index], self.memory[min_index+1:]], dim=0))
         self.N -= 1
         # Update the linkage matrix accordingly
         self.linkage_matrix.deallocate_memory_slot(min_index)
         
        # Consider pre-allocating memory to improve computational efficiency and stability,
        # especially if using GPU-accelerated framework like PyTorch or CUDA.
            
    def read(self, ws):
        # `ws` is a list of read weight vectors
        return self.memory * ws.unsqueeze(-1), self.linkage_matrix.read(ws)

    def reset(self):
        # Resets memory to all zeros
        self.memory.data.fill_(0)
        # Reset TemporalLinkageMatrix
        self.linkage_matrix.reset()
        
class HierarchicalMemory(nn.Module):
    def __init__(self, N, M, controller_size, levels=3):
        super(HierarchicalMemory, self).__init__()

        self.levels = levels

        # Create a DynamicMemory instance for each level
        self.memory_levels = nn.ModuleList([DynamicMemory(N, M, controller_size) for _ in range(levels)])
        self.tlm_levels = nn.ModuleList([TemporalLinkageMatrix(N) for _ in range(levels)])

    def forward(self, controller_output, read_ws, write_ws, erase_vectors, add_vectors):
        # Initialize list to store read vectors and weights from each level
        read_vectors_list = []
        forward_weights_list = []
        backward_weights_list = []

        # Loop over each level in the hierarchy
        for i in range(self.levels):

            # Calculate the read and write weights for this level
            read_ws_level = read_ws[i]
            write_ws_level = write_ws[i]

            # Get the erase and add vectors for this level
            erase_vectors_level = erase_vectors[i]
            add_vectors_level = add_vectors[i]

            # Perform the write operation at this level
            self.memory_levels[i].write(write_ws_level, erase_vectors_level, add_vectors_level)

            # Update TLM after write operation at this level
            self.tlm_levels[i].update_linkage_matrix(write_ws_level)

            # Perform the read operation at this level
            read_vectors, _ = self.memory_levels[i].read(read_ws_level)

            # Now also get the forward and backward weights from the TLM at this level
            forward_weights, backward_weights = self.tlm_levels[i].get_forward_backward_weights()

            # Append the read vectors and weights to the lists
            read_vectors_list.append(read_vectors)
            forward_weights_list.append(forward_weights)
            backward_weights_list.append(backward_weights)

            # Update controller output with the read vectors from this level
            controller_output = self.update_controller_output(controller_output, read_vectors)

        # Return updated controller output, list of read vectors, and lists of forward and backward weights
        return controller_output, read_vectors_list, forward_weights_list, backward_weights_list

    def update_controller_output(self, controller_output, read_vectors):
        # Concatenate read vectors to controller output along dimension 1
        updated_output = torch.cat([controller_output, read_vectors], dim=1)
        return updated_output

    def reset(self):
        # Reset each level of memory and TLM
        for i in range(self.levels):
            self.memory_levels[i].reset()
            self.tlm_levels[i].reset()  # Also reset TLMs