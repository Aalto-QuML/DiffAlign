import torch
import torch.nn.functional as F

class GraphAttentionLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.5):
        super(GraphAttentionLayer, self).__init__()
        self.W = torch.nn.Linear(input_dim, output_dim)
        self.a = torch.nn.Linear(2*output_dim, 1)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x, adj):
        h = self.W(x)
        a_input = self._prepare_attentional_mechanism_input(h)
        e = self._calculate_attention_coefficients(a_input, adj)
        alpha = F.softmax(e, dim=1)
        alpha = self.dropout(alpha)
        h_prime = torch.matmul(alpha, h)
        return h_prime

    def _prepare_attentional_mechanism_input(self, h):
        N = h.size()[0]
        h_rep = h.repeat_interleave(N, dim=0)
        h_broadcast = h.repeat(N, 1)
        combined_h = torch.cat([h_rep, h_broadcast], dim=1)
        return combined_h.view(N, -1, 2*h.size()[1])

    def _calculate_attention_coefficients(self, a_input, adj):
        e = F.leaky_relu(self.a(a_input))
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        return attention

class GraphTransformerNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, dropout_rate=0.5):
        super(GraphTransformerNetwork, self).__init__()
        self.W1 = torch.nn.Linear(input_dim, hidden_dim)
        self.W2 = torch.nn.Linear(hidden_dim*num_heads, output_dim)
        self.multihead_attention = torch.nn.ModuleList([GraphAttentionLayer(hidden_dim, hidden_dim, dropout_rate) for _ in range(num_heads)])
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x, adj):
        h = F.relu(self.W1(x))
        h_list = [att(h, adj) for att in self.multihead_attention]
        h_prime = torch.cat(h_list, dim=2)
        h_prime = self.dropout(h_prime)
        out = self.W2(h_prime)
        return out
