import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import (
    alexnet,
    squeezenet1_1,
    resnet18,
    vgg11,
    vit_b_16,
    AlexNet_Weights,
    SqueezeNet1_1_Weights,
    ResNet18_Weights,
    VGG11_Weights,
    ViT_B_16_Weights
)
from typing import List,Tuple

class SimpleMLP(nn.Module):
    def __init__(self, input_size:int, hidden_sizes:List[int],
                 output_size:int,
                 active_end=False, dropout=0.2):
        """
        Initialize a simple MLP model.

        Args:
            input_size:     Input size.
            hidden_sizes:   List of hidden layer sizes.
            output_size:    Output size.
            active_end:     Activation function at the end.
            dropout:        Dropout rate.

        Variables:
            networks:       List of neural networks.
        """
        # Initialize the parent class
        super(SimpleMLP, self).__init__()

        # Populate the layers
        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = size
        layers.append(nn.Linear(prev_size, output_size))

        # Add final activation function if required
        if active_end:
            layers.append(nn.ReLU())

        # Define the model
        self.networks = nn.Sequential(*layers)
    
    def forward(self, xnn:torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            xnn:  Input tensor.

        Returns:
            ynn:  Output tensor.
        """

        # Simple MLP
        ynn = self.networks(xnn)         

        return ynn
    
class SimpleEncoder(nn.Module):
    def __init__(self, input_size:int, hidden_sizes:List[int],
                 encoder_output_size:int, decoder_output_size:int,
                 active_end=False, dropout=0.2):
        """
        Initialize a simple encoder model (with a decoder for training).

        Args:
            input_size:         Input size.
            hidden_sizes:       List of hidden layer sizes.
            encoder_output_size: Encoder output size.
            decoder_output_size: Decoder output size.
            active_end:         Activation function at the end.
            dropout:            Dropout rate.

        Variables:
            networks:           List of neural networks.
            final_layer:        Final layer.
        """
        # Initialize the parent class
        super(SimpleEncoder, self).__init__()

        # Populate the layers
        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = size
        layers.append(nn.Linear(prev_size, encoder_output_size))
        layers.append(nn.ReLU())

        # Add final activation function if required
        if active_end:
            layers.append(nn.ReLU())

        # Define the model
        self.networks = nn.Sequential(*layers)
        self.final_layer = nn.Linear(encoder_output_size, decoder_output_size)

    def forward(self, xnn:torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            xnn:  Input tensor.

        Returns:
            ynn:  Decoder output tensor.
            znn:  Encoder output tensor.
        """

        # Simple Encoder
        znn = self.networks(xnn)            # encoder output
        ynn = self.final_layer(znn)         # decoder output
        
        return ynn,znn

class SharpEncoder(nn.Module):
    def __init__(self, input_size:int, hidden_sizes:List[int],
                 encoder_output_size:int, decoder_output_size:int,
                 active_end=False, base_dropout=0.1, final_dropout=0.4):
        """
        Initialize a simple encoder model (with a decoder for training).

        Args:
            input_size:         Input size.
            hidden_sizes:       List of hidden layer sizes.
            encoder_output_size: Encoder output size.
            decoder_output_size: Decoder output size.
            active_end:         Activation function at the end.
            dropout:            Dropout rate.

        Variables:
            networks:           List of neural networks.
            final_layer:        Final layer.
        """
        # Initialize the parent class
        super(SharpEncoder, self).__init__()

        # Populate the layers
        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(base_dropout))
            prev_size = size
        layers.append(nn.Linear(prev_size, encoder_output_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(final_dropout))

        # Add final activation function if required
        if active_end:
            layers.append(nn.ReLU())

        # Define the model
        self.networks = nn.Sequential(*layers)
        self.final_layer = nn.Linear(encoder_output_size, decoder_output_size)

    def forward(self, xnn:torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            xnn:  Input tensor.

        Returns:
            ynn:  Decoder output tensor.
            znn:  Encoder output tensor.
        """

        # Simple Encoder
        znn = self.networks(xnn)            # encoder output
        ynn = self.final_layer(znn)         # decoder output
        
        return ynn,znn

class DirectEncoder(nn.Module):
    def __init__(self, input_size:int, hidden_sizes:List[int],
                 encoder_output_size:int,
                 active_end=False, dropout=0.2):
        """
        Initialize a simple encoder model (with a decoder for training).

        Args:
            input_size:         Input size.
            hidden_sizes:       List of hidden layer sizes.
            decoder_output_size: Decoder output size.
            active_end:         Activation function at the end.
            dropout:            Dropout rate.

        Variables:
            networks:           List of neural networks.
            final_layer:        Final layer.
        """
        # Initialize the parent class
        super(DirectEncoder, self).__init__()

        # Populate the layers
        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = size
        layers.append(nn.Linear(prev_size, encoder_output_size))
        layers.append(nn.ReLU())

        # Add final activation function if required
        if active_end:
            layers.append(nn.ReLU())

        # Define the model
        self.networks = nn.Sequential(*layers)

    def forward(self, xnn:torch.Tensor) -> Tuple[torch.Tensor,None]:
        """
        Forward pass of the model.

        Args:
            xnn:  Input tensor.

        Returns:
            ynn:  Decoder output tensor.
            znn:  Encoder output tensor.
        """

        # Simple Encoder
        ynn = self.networks(xnn)            # encoder output
        
        return ynn,None

class WideShallowCNN(nn.Module):
    def __init__(self, input_channels=3, output_dim=1000):
        super(WideShallowCNN, self).__init__()

        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)  # 224x224x3 -> 224x224x64
        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)  # 112x112x64 -> 112x112x128
        # Convolutional Layer 3
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)  # 56x56x128 -> 56x56x256

        # Fully Connected Layer
        # Flatten the features before passing to this layer
        # Assuming input image size is 224x224, this will need to be adjusted if input size changes
        self.fc1 = nn.Linear(256 * 28 * 28, 1000)

        # Max Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        # Apply Conv1 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv1(x)))  # 224x224x64 -> 112x112x64
        # Apply Conv2 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))  # 112x112x128 -> 56x56x128
        # Apply Conv3 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv3(x)))  # 56x56x256 -> 28x28x256

        # Flatten the feature map
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 256*28*28)

        # Fully Connected Layer
        x = self.fc1(x)  # Output 1000 features

        return x
    
class VisionCNN(nn.Module):
    def __init__(self, visual_type:str,
                 Nout:int=1000):
        """
        Initialize a vision CNN model.

        Args:
            visual_type:    Type of visual network.
            Nout:           Output size.

        Variables:
            networks:       Vision network.
            Nout:           Output size.
        """
        # Initialize the parent class
        super(VisionCNN, self).__init__()

        # Instantiate Visual Network
        if visual_type == "AlexNet":
            networks = alexnet(weights=AlexNet_Weights.DEFAULT)
        elif visual_type == "SqueezeNet1_1":
            networks = squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT)
        elif visual_type == "ResNet18":
            networks = resnet18(weights=ResNet18_Weights.DEFAULT)()
        elif visual_type == "VGG11":
            networks = vgg11(weights=VGG11_Weights.DEFAULT)
        elif visual_type == "ViT_B_16":
            networks = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        elif visual_type == "WideShallowCNN":
            networks = WideShallowCNN()
        else:
            raise ValueError(f"Invalid visual_type: {visual_type}")
        
        # Define the model
        self.networks = networks
        self.Nout = Nout

    def forward(self, xnn:torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            xnn:  Input tensor.
        
        Returns:
            ynn:  Output tensor.
        """

        # Vision CNN
        ynn = self.networks(xnn)

        return ynn