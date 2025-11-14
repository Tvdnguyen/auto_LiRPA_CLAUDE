"""
Traffic Sign Recognition Network for GTSRB
A CNN architecture designed for traffic sign classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TrafficSignNet(nn.Module):
    """
    Traffic Sign Classification Network
    Architecture: Conv -> Conv -> Pool -> Conv -> Conv -> Pool -> FC -> FC -> Output
    Designed to achieve >90% accuracy on GTSRB dataset
    """

    def __init__(self, num_classes: int = 43):
        super(TrafficSignNet, self).__init__()

        # First convolutional block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.2)

        # Second convolutional block
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.2)

        # Third convolutional block
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.3)

        # Fully connected layers
        # After 3 pooling layers: 32 -> 16 -> 8 -> 4
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout5 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        # First block
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout1(x)

        # Second block
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        # Third block
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)
        x = self.dropout3(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = F.relu(self.fc2(x))
        x = self.dropout5(x)
        x = self.fc3(x)

        return x

    def get_layer_info(self):
        """
        Return information about all Conv and FC layers for perturbation selection

        Returns:
            List of tuples: (layer_name, layer_type, layer_object, output_shape_info)
        """
        layers = []

        # Convolutional layers
        layers.append(('conv1', 'Conv2d', self.conv1, '32x32x32'))
        layers.append(('conv2', 'Conv2d', self.conv2, '32x32x32'))
        layers.append(('conv3', 'Conv2d', self.conv3, '16x16x64'))
        layers.append(('conv4', 'Conv2d', self.conv4, '16x16x64'))
        layers.append(('conv5', 'Conv2d', self.conv5, '8x8x128'))
        layers.append(('conv6', 'Conv2d', self.conv6, '8x8x128'))

        # Fully connected layers
        layers.append(('fc1', 'Linear', self.fc1, '512'))
        layers.append(('fc2', 'Linear', self.fc2, '256'))
        layers.append(('fc3', 'Linear', self.fc3, '43'))

        return layers


class TrafficSignNetNoDropout(nn.Module):
    """
    Traffic Sign Network WITHOUT Dropout layers
    Used for verification with auto_LiRPA intermediate perturbations
    Dropout causes issues with Patches in backward propagation
    """

    def __init__(self, num_classes: int = 43):
        super(TrafficSignNetNoDropout, self).__init__()

        # First convolutional block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # No dropout1

        # Second convolutional block
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # No dropout2

        # Third convolutional block
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # No dropout3

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        # No dropout4
        self.fc2 = nn.Linear(512, 256)
        # No dropout5
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        # First block (no dropout)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        # Second block (no dropout)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        # Third block (no dropout)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers (no dropout)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def get_layer_info(self):
        """Return layer information"""
        layers = []
        layers.append(('conv1', 'Conv2d', self.conv1, '32x32x32'))
        layers.append(('conv2', 'Conv2d', self.conv2, '32x32x32'))
        layers.append(('conv3', 'Conv2d', self.conv3, '16x16x64'))
        layers.append(('conv4', 'Conv2d', self.conv4, '16x16x64'))
        layers.append(('conv5', 'Conv2d', self.conv5, '8x8x128'))
        layers.append(('conv6', 'Conv2d', self.conv6, '8x8x128'))
        layers.append(('fc1', 'Linear', self.fc1, '512'))
        layers.append(('fc2', 'Linear', self.fc2, '256'))
        layers.append(('fc3', 'Linear', self.fc3, '43'))
        return layers

    def load_from_dropout_checkpoint(self, checkpoint_path):
        """
        Load weights from a checkpoint trained with dropout model
        Automatically handles missing dropout layer keys

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            self (for chaining)
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict']

        # Filter out dropout keys
        filtered_state_dict = {
            k: v for k, v in state_dict.items()
            if 'dropout' not in k
        }

        # Load (with strict=False to ignore missing dropout keys)
        self.load_state_dict(filtered_state_dict, strict=False)

        print(f"Loaded weights from {checkpoint_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Test Accuracy: {checkpoint.get('test_acc', 'N/A'):.2f}%")
        print("  Dropout layers: Skipped")

        return self


class TrafficSignNetSimple(nn.Module):
    """
    Simplified Traffic Sign Network (faster training for testing)
    """

    def __init__(self, num_classes: int = 43):
        super(TrafficSignNetSimple, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def get_layer_info(self):
        """Return layer information"""
        layers = []
        layers.append(('conv1', 'Conv2d', self.conv1, '32x32x32'))
        layers.append(('conv2', 'Conv2d', self.conv2, '16x16x64'))
        layers.append(('conv3', 'Conv2d', self.conv3, '8x8x128'))
        layers.append(('fc1', 'Linear', self.fc1, '256'))
        layers.append(('fc2', 'Linear', self.fc2, '43'))
        return layers


def test_model():
    """Test model architecture"""
    model = TrafficSignNet(num_classes=43)
    print("TrafficSignNet Architecture:")
    print(model)
    print("\n" + "="*50)

    # Test forward pass
    dummy_input = torch.randn(1, 3, 32, 32)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # Print layer info
    print("\n" + "="*50)
    print("Available layers for perturbation:")
    for i, (name, ltype, layer, shape) in enumerate(model.get_layer_info()):
        print(f"{i}: {name:10s} | Type: {ltype:10s} | Output shape: {shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


if __name__ == '__main__':
    test_model()
