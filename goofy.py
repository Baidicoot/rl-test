import torch
from torch import nn

class Feedforward(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.GELU()
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        
    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        return output

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    def attention(self, query, key, value):
        score = torch.matmul(query, key.transpose(-1, -2))
        dim_key = torch.tensor(key.shape[-1], dtype=torch.float32)
        scaled_score = score / torch.sqrt(dim_key)
        weights = nn.functional.softmax(scaled_score, dim=-1)
        output = torch.matmul(weights, value)
        return output

    def separate_heads(self, x, batch_size):
        x = x.reshape(batch_size, -1, self.num_heads, self.projection_dim)
        return x.transpose(1, 2)

    def forward(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = inputs.shape[0]
        query = self.query(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )
        key = self.separate_heads(
            key, batch_size
        )
        value = self.separate_heads(
            value, batch_size
        )
        attention = self.attention(query, key, value)
        attention = attention.transpose(1, 2)
        concat_attention = attention.reshape(batch_size, -1, self.embed_dim)
        output = self.combine_heads(concat_attention)
        return output

class DecoderOnlyLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff=None):
        super().__init__()

        if d_ff is None:
            d_ff = 4*d_model

        self.attn = MultiHeadSelfAttention(d_model, n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = Feedforward(d_model, d_ff, d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.norm1(self.attn(x))
        x = x + self.norm2(self.ff(x))
        return x

class DecoderOnly(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_layers, d_model, n_head):
        super().__init__()

        self.embed = nn.Linear(n_inputs, d_model)
        self.layers = [DecoderOnlyLayer(d_model, n_head) for _ in range(n_layers)]
        self.out = nn.Linear(d_model, n_outputs)
    
    def forward(self, x):
        x = self.embed(x)
        pos = torch.linspace(0, x.shape[1], x.shape[1]).to(x.device)
        x[:,:,0] += pos
        for l in self.layers:
            x = l(x)
        logits = self.out(x)
        return logits

class TestModel(nn.Module):
    def __init__(self, n_inputs, d_model, n_outputs):
        super().__init__()

        self.ff1 = Feedforward(n_inputs, d_model, n_outputs)
        #self.gelu = nn.GELU()
        #self.ff2 = Feedforward(d_model, d_model, n_outputs)

    def forward(self, x):
        return self.ff1(x)
        #return self.ff2(self.gelu(self.ff1(x)))