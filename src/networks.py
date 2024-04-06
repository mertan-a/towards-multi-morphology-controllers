import torch
from torch import nn
from torch.nn.utils import parameters_to_vector
from torch import sigmoid
import numpy as np
from copy import deepcopy
import math

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.relu_stack = nn.Sequential(
            nn.Linear(self.input_size, 32),
            nn.ReLU(),
            #nn.Linear(32, 64),
            #nn.ReLU(),
            #nn.Linear(64, 32),
            #nn.ReLU(),
            nn.Linear(32, self.output_size)
        )

    def forward(self, x):
        logits = self.relu_stack(x)
        return sigmoid(logits) - 0.4

class NeuralNetworkModular(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetworkModular, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.relu_stack = nn.Sequential(
            nn.Linear(self.input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.output_size)
        )

    def forward(self, x):
        logits = self.relu_stack(x)
        return sigmoid(logits) - 0.4


#class TransformerEncoder(nn.Module):
#    def __init__(self, embedding_size, num_attention_heads=4):
#        super(TransformerEncoder, self).__init__()
#        # query, key, value weight matrices for each attention head
#        self.query_weight_matrices = nn.ModuleList([nn.Linear(embedding_size, embedding_size) for _ in range(num_attention_heads)])
#        self.key_weight_matrices = nn.ModuleList([nn.Linear(embedding_size, embedding_size) for _ in range(num_attention_heads)])
#        self.value_weight_matrices = nn.ModuleList([nn.Linear(embedding_size, embedding_size) for _ in range(num_attention_heads)])
#        # output weight matrix
#        self.output_weight_matrix = nn.Linear(embedding_size*num_attention_heads, embedding_size)
#        # encoder mlp
#        self.encoder_mlp = nn.Sequential(
#            nn.Linear(embedding_size, embedding_size),
#            nn.Tanh(),
#            nn.Linear(embedding_size, embedding_size)
#        )
#        # layer normalization
#        self.layer_norm_1 = nn.LayerNorm(embedding_size)
#        self.layer_norm_2 = nn.LayerNorm(embedding_size)
#
#    def forward(self, x):
#        # x: (num_existing_voxels, embedding_size)
#        # make copy of x for residual connection
#        x_copy = deepcopy(x)
#        # calculate attention for each attention head
#        attention_heads = []
#        for i in range(len(self.query_weight_matrices)):
#            # calculate attention
#            attention_heads.append(self.calculate_attention(x, self.query_weight_matrices[i], self.key_weight_matrices[i], self.value_weight_matrices[i]))
#        # concatenate attention heads
#        x = torch.cat(attention_heads, dim=1)
#        # output weight matrix
#        x = self.output_weight_matrix(x)
#        # add residual connection
#        x = x + x_copy
#        # normalize
#        x = self.layer_norm_1(x)
#        # make copy of x for residual connection
#        x_copy = deepcopy(x)
#        # encoder mlp
#        x = self.encoder_mlp(x)
#        # add residual connection
#        x = x + x_copy
#        # normalize
#        x = self.layer_norm_2(x)
#        return x
#
#    def calculate_attention(self, x, query_weight_matrix, key_weight_matrix, value_weight_matrix):
#        # x: (num_existing_voxels, embedding_size)
#        # query: (embedding_size, embedding_size)
#        # key: (embedding_size, embedding_size)
#        # value: (embedding_size, embedding_size)
#        # calculate query, key, value
#        query = query_weight_matrix(x)
#        key = key_weight_matrix(x)
#        value = value_weight_matrix(x)
#        # calculate attention
#        qXk = torch.matmul(query, key.transpose(0,1))
#        attention = torch.softmax(qXk/math.sqrt(x.shape[1]), dim=1)
#        # calculate output
#        output = torch.matmul(attention, value)
#        return output
#
#class TransformerDecoder(nn.Module):
#    def __init__(self, embedding_size, num_output_heads):
#        super(TransformerDecoder, self).__init__()
#        # an mlp for each output head
#        self.mlps = nn.ModuleList([nn.Sequential(
#            nn.Linear(embedding_size, 8),
#            nn.Tanh(),
#            nn.Linear(8, 1)
#        ) for _ in range(num_output_heads)])
#
#    def forward(self, x, mask):
#        # x: (num_existing_voxels, embedding_size)
#        outs = []
#        pointer_to_x = 0
#        for i, mlp in enumerate(self.mlps):
#            if mask[i]:
#                outs.append(mlp(x[pointer_to_x, :]))
#                pointer_to_x += 1
#        outs = torch.cat(outs, dim=0)
#        return sigmoid(outs) - 0.4

#class TransformerNetwork(nn.Module):
#    def __init__(self, robot_bounding_box, observation_per_voxel_size, 
#                 embedding_size, num_attention_heads=4, num_encoder_layers=1):
#        super(TransformerNetwork, self).__init__()
#        # transformer encoder
#        self.encoders = nn.ModuleList([TransformerEncoder(embedding_size, num_attention_heads) for _ in range(num_encoder_layers)])
#        self.decoder = TransformerDecoder(embedding_size, robot_bounding_box[0]*robot_bounding_box[1])
#        # positional encoding
#        self.positional_encoding = self.get_positional_encoding(robot_bounding_box, embedding_size)
#        # learnable linear projection for each voxel
#        self.linear_projection = nn.Linear(observation_per_voxel_size, embedding_size)
#    
#    def forward(self, x, mask):
#        # x: (num_existing_voxels, observation_per_voxel_size)
#        # linear projection
#        x = self.linear_projection(x)
#        # add positional encoding
#        x = x + self.positional_encoding[mask, :]
#        # transformer encoder
#        for encoder in self.encoders:
#            x = encoder(x)
#        # transformer decoder
#        x = self.decoder(x, mask)
#        return x
#
#    def get_positional_encoding(self, robot_bounding_box, embedding_size):
#        # positional encoding
#        pe = torch.zeros(robot_bounding_box[0]*robot_bounding_box[1], embedding_size)
#        position = torch.arange(0, robot_bounding_box[0]*robot_bounding_box[1], dtype=torch.float).unsqueeze(1)
#        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-math.log(10000.0) / embedding_size))
#        pe[:, 0::2] = torch.sin(position * div_term)
#        pe[:, 1::2] = torch.cos(position * div_term)
#        return pe.double()

class TransformerNetwork(nn.Module):
    def __init__(self, robot_bounding_box, observation_per_voxel_size,
                 embedding_size, num_attention_heads=4, num_encoder_layers=1):
        super(TransformerNetwork, self).__init__()
        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=num_attention_heads,
                                                    dim_feedforward=embedding_size*4, dropout=0,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        # embedding
        self.embedding = nn.Linear(observation_per_voxel_size, embedding_size)
        # positional encoding
        self.positional_encoding = self.get_positional_encoding(robot_bounding_box, embedding_size)
        # output layer
        self.output_layer = nn.Linear(embedding_size, 1)
        # save embedding size
        self.embedding_size = embedding_size

    def get_positional_encoding(self, robot_bounding_box, embedding_size):
        # positional encoding
        pe = torch.zeros(robot_bounding_box[0]*robot_bounding_box[1], embedding_size)
        position = torch.arange(0, robot_bounding_box[0]*robot_bounding_box[1], dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-math.log(10000.0) / embedding_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe.double()

    def forward(self, x):
        #print(x.shape) # (batchsize, 25, 9)
        # linear projection
        x = self.embedding(x) * math.sqrt(self.embedding_size)
        # add positional encoding
        x = x + self.positional_encoding
        # transformer encoder
        x = self.encoder(x)
        # linear output layer
        outputs = self.output_layer(x)
        # activation
        outputs = sigmoid(outputs) - 0.4
        outputs = outputs.squeeze()
        return outputs

# test code
if __name__ == '__main__':
    mlp = NeuralNetwork(input_size=25*9, output_size=25)
    print(f'num params: {parameters_to_vector(mlp.parameters()).shape}')
    transformer = TransformerNetwork(robot_bounding_box=[5,5], observation_per_voxel_size=9, embedding_size=64, 
                              num_attention_heads=8, num_encoder_layers=4)
    print(f'num params: {parameters_to_vector(transformer.parameters()).shape}')
    modular = NeuralNetworkModular(input_size=9*9, output_size=1)
    print(f'num params: {parameters_to_vector(modular.parameters()).shape}')
    exit()
    ####
    robot_bounding_box = (10,10)
    mask = torch.rand(robot_bounding_box[0]*robot_bounding_box[1]) > 0.5
    num_existing_voxels = torch.sum(mask)
    print(f'num_existing_voxels: {num_existing_voxels}')
    transformer = TransformerNetwork(robot_bounding_box=robot_bounding_box, observation_per_voxel_size=9, embedding_size=8, 
                              num_attention_heads=4, num_encoder_layers=1)
    print(f'num params: {parameters_to_vector(transformer.parameters()).shape}')
    exit()
    import matplotlib.pyplot as plt
    cax = plt.matshow(transformer.positional_encoding.detach().numpy().squeeze(0))
    plt.gcf().colorbar(cax)
    plt.show()
    plt.close()

    transformer.double()
    inp = torch.rand(1, num_existing_voxels, 9).double()
    out = transformer(inp, mask)
    #print(f'out: {out.shape}')





