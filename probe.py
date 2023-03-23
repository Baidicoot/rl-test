import torch
from torch import nn
import torch.nn.functional as F
from goofy import MultiLayerFeedForward

import matplotlib.pyplot as plt

class ProbedFeedForward(nn.Module):
    def __init__(self, n_layers, n_inputs, d_model, n_outputs):
        super().__init__()
        self.in_layer = nn.Linear(n_inputs, d_model)
        self.layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layers)])
        self.out_layer = nn.Linear(d_model, n_outputs)
        
        self.register_buffer("probe", torch.zeros(n_layers+1, d_model))

    def forward(self, x):
        x = F.gelu(self.in_layer(x))
        self.probe[0,:] = x
        for i, l in enumerate(self.layers):
            x = F.gelu(l(x))
            self.probe[i+1,:] = x
        x = self.out_layer(x)
        return x

    def from_mlff(net):
        """
        Create a ProbedFeedForward from a MultiLayerFeedForward
        """
        pff = ProbedFeedForward(len(net.layers), net.in_layer.in_features, net.layers[0].out_features, net.out_layer.out_features)
        pff.in_layer.weight = net.in_layer.weight
        pff.in_layer.bias = net.in_layer.bias
        for i, l in enumerate(net.layers):
            pff.layers[i].weight = l.weight
            pff.layers[i].bias = l.bias
        pff.out_layer.weight = net.out_layer.weight
        pff.out_layer.bias = net.out_layer.bias
        return pff

def show_act(inp, probe, out):
    """
    Show the activations stored in probe, with output in out, on the plot plt
    The activations should be plotted as a 2D black and white image
    """
    fig, ax = plt.subplots(3, 1)
    fig.suptitle("activations")

    ax[0].imshow(inp.unsqueeze(0).detach().numpy(), cmap="gray")
    ax[0].set_xlabel("neuron")
    ax[0].set_yticks([])

    ax[1].imshow(probe.flip(0).detach().numpy(), cmap="gray")
    ax[1].set_xlabel("neuron")
    ax[1].set_ylabel("layer")
    ax[1].set_yticks([])

    ax[2].imshow(out.unsqueeze(0).detach().numpy(), cmap="gray")
    ax[2].set_xlabel("neuron")
    ax[2].set_yticks([])

def show_weights(net):
    """
    Show the weights of the network net
    The weights shoud be presented as a 2D black and white image per layer
    All images should be shown on the same figure
    Each image should have axis titles indicating to and from neurons
    """
    fig, ax = plt.subplots(1, len(net.layers)+2)
    fig.suptitle("weights")

    ax[0].imshow(net.in_layer.weight.detach().numpy(), cmap="gray")
    ax[0].set_xlabel("from")
    ax[0].set_ylabel("to")
    ax[0].set_title("input layer")

    for i, l in enumerate(net.layers):
        ax[i+1].imshow(l.weight.detach().numpy(), cmap="gray")
        ax[i+1].set_xlabel("from")
        ax[i+1].set_ylabel("to")
        ax[i+1].set_title(f"layer {i}")

    ax[-1].imshow(net.out_layer.weight.transpose(0, 1).detach().numpy(), cmap="gray")
    ax[-1].set_xlabel("to")
    ax[-1].set_ylabel("from")
    ax[-1].set_title("output layer")

model = MultiLayerFeedForward(2, 8, 4, 4)
model.load_state_dict(torch.load("policy_10K.pt"))
model = ProbedFeedForward.from_mlff(model)

inp = torch.rand(8)
out = model(inp)

show_act(inp, model.probe, out)
show_weights(model)

plt.show()