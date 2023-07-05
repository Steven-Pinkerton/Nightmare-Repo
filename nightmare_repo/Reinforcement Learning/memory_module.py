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

    def forward(self, x):
        lstm_output, state = self.lstm(x)  # Return the LSTM state as well
        output = self.fc(lstm_output[:, -1, :])
        return output, state  # Return both the output and the state
    
class Memory(nn.Module):
    def __init__(self, N, M):
        super(Memory, self).__init__()

        # N is the number of memory locations (also known as the number of rows of the memory matrix)
        # M is the size of each memory location (also known as the number of columns of the memory matrix)
        self.N = N
        self.M = M

        # Initialize memory matrix with zeros
        self.memory = nn.Parameter(torch.zeros(N, M))

        def read(self, ws):
            # `ws` is a list of read weight vectors
            return [torch.matmul(w.unsqueeze(1), self.memory).squeeze(1) for w in ws]

    def write(self, ws, es, as):
        # `ws`, `es`, and `as` are lists of write weight, erase, and add vectors respectively
        for w, e, a in zip (ws, es, as):
            self.memory = self.memory * (1 - torch.matmul(w.unsqueeze(-1), e.unsqueeze(1)))
            self.memory = self.memory + torch.matmul(w.unsqueeze(-1), a.unsqueeze(1))

    def reset(self):
        # Resets memory to all zeros
        self.memory.data.fill_(0)
        
    def update_linkage_matrix(self, write_weights):
        # Update the Temporal Linkage Matrix after a write operation
        self.linkage_matrix.update(write_weights)
        
    def get_linkage_matrix(self):
        # Get the current state of the Temporal Linkage Matrix
        return self.linkage_matrix.matrix

    def read_linkage_matrix(self, read_weights):
        # Read from the Temporal Linkage Matrix
        return self.linkage_matrix.read(read_weights)
        
class InterfaceVectors(nn.Module):
    def __init__(self, controller_size, memory_N, memory_M):
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

    def calculate_interface_vector_size(self):
        # The size of the interface vectors is determined by the NTM's specifications.
        # It includes the read keys (k_read), write key (k_write), erase vector (e),
        # add vector (a), read strengths (beta_read), write strength (beta_write), 
        # read modes (pi), write gate (g_write), allocation gate (g_alloc), and free gates (f).
        return self.memory_M * 2 + self.memory_N * 3 + 5

    def forward(self, x):
        iv = self.linear(x)
        read_vectors, write_vectors = self.split_interface_vector(iv, num_heads=self.num_heads)
        read_ws = [self.addressing(w) for w in read_vectors]
        write_ws = [self.addressing(w) for w in write_vectors]
        
        self.write(write_ws, [e for e, _ in write_vectors], [a for _, a in write_vectors])  # Perform write operation
        self.memory.update_linkage_matrix(write_ws)  # Update the Temporal Linkage Matrix

        # Add TLM's contribution in read operation
        for w in read_ws:
            forward_weights, backward_weights = self.memory.read_linkage_matrix(w)  # Read from the Temporal Linkage Matrix
            # You might need to combine these weights with the content and location-based addressing weights
            
        read_data = [self.read(w) for w in read_ws]  # Read data for each head
        return read_data  # Return the read data as a list

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
    
    def content_addressing(self, k, beta):
        # Compute cosine similarity
        cos_sim = F.cosine_similarity(self.memory + 1e-16, k + 1e-16, dim=-1)
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
        self.matrix = nn.Parameter(torch.zeros(N, N))
        self.precedence_weight = nn.Parameter(torch.zeros(N))

    def update(self, w):
        # `w` is the current write weight vector
        w = w.unsqueeze(1)  # reshape to column vector

        # Update the precedence weight
        self.precedence_weight = (1 - torch.sum(w, dim=0)) * self.precedence_weight + w

        # Update the temporal linkage matrix
        self.matrix = (1 - w - w.t()) * self.matrix + torch.matmul(w, self.precedence_weight.t())

        # Make sure that self-loops in the matrix are zero
        self.matrix *= 1 - torch.eye(self.N)

    def read(self, w):
        # `w` is the current read weight vector

        # Forward and backward read weights
        wf = torch.matmul(w, self.matrix)
        wb = torch.matmul(w, self.matrix.t())

        return wf, wb