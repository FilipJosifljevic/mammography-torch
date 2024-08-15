import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from keras.models import load_model
import numpy as np

from torch_mammo import CustomCNN

# Create a torch model instance
torch_model = CustomCNN()
print(torch_model)

# Load weights from keras model
def extract_weights(model, weights_dict, prefix=''):
    for layer in model.layers:
        if hasattr(layer, 'layers'):  # Check if the layer is a nested model
            extract_weights(layer, weights_dict, prefix=prefix + layer.name + '_')
        else:
            weights = layer.get_weights()
            if weights:
                # Print layer name and weights shapes
                print(f"Extracting weights for layer: {prefix + layer.name}")
                for i, w in enumerate(weights):
                    print(f"  Weight {i} shape: {w.shape}")


                # Verify expected shapes based on layer type
                if 'conv' in layer.name:
                    expected_shape = (layer.filters, layer.input_shape[-1], layer.kernel_size[0], layer.kernel_size[1])
                    if weights[0].shape != (expected_shape[2], expected_shape[3], expected_shape[1], expected_shape[0]):
                        print(
                            f"Warning: Unexpected shape for conv weights in {layer.name}. Expected: {expected_shape}, Got: {weights[0].shape}")
                elif 'dense' in layer.name:
                    expected_shape = (layer.input_shape[-1], layer.units)
                    if weights[0].shape != expected_shape:
                        print(
                            f"Warning: Unexpected shape for dense weights in {layer.name}. Expected: {expected_shape}, Got: {weights[0].shape}")
                weights_dict[prefix + layer.name] = weights

                # print(prefix + layer.name)

keras_model = load_model("inbreast_vgg16_512x1.h5")
keras_weights = {}
extract_weights(keras_model, keras_weights)

# Mapping of Keras layer names to PyTorch layers
name_mapping = {
    'model_1_block1_conv1': ('model1', 'conv1_1'),
    'model_1_block1_conv2': ('model1', 'conv1_2'),
    'model_1_block2_conv1': ('model1', 'conv2_1'),
    'model_1_block2_conv2': ('model1', 'conv2_2'),
    'model_1_block3_conv1': ('model1', 'conv3_1'),
    'model_1_block3_conv2': ('model1', 'conv3_2'),
    'model_1_block3_conv3': ('model1', 'conv3_3'),
    'model_1_block4_conv1': ('model1', 'conv4_1'),
    'model_1_block4_conv2': ('model1', 'conv4_2'),
    'model_1_block4_conv3': ('model1', 'conv4_3'),
    'model_1_block5_conv1': ('model1', 'conv5_1'),
    'model_1_block5_conv2': ('model1', 'conv5_2'),
    'model_1_block5_conv3': ('model1', 'conv5_3'),
    'conv2d_1': 'conv1',
    'batch_normalization_1': 'bn1',
    'conv2d_2': 'conv2',
    'batch_normalization_2': 'bn2',
    'dense_1': 'fc'
}

# Function to convert Keras weights to PyTorch format
def convert_weights(keras_weights):
    if keras_weights.ndim == 4:  # Convolutional layer
        # Transpose the weights from (H, W, C_in, C_out) to (C_out, C_in, H, W)
        return torch.from_numpy(np.transpose(keras_weights, (3, 2, 0, 1)))
    elif keras_weights.ndim == 2:  # Dense layer
        # Transpose the weights from (input_features, output_features) to (output_features, input_features)
        return torch.from_numpy(np.transpose(keras_weights))
    else:
        return torch.from_numpy(keras_weights)

# # Add weights to torch model
for keras_name, torch_names in name_mapping.items():
    if isinstance(torch_names, tuple):  # submodules in layer model1
        module, layer = torch_names
        target_layer = getattr(getattr(torch_model, module), layer)
    else:
        target_layer = getattr(torch_model, torch_names)

    layer_weights = keras_weights[keras_name]
    conv_weights, conv_bias = layer_weights[0], layer_weights[1]

    # Depending on the layer type, different actions might be needed
    if 'conv' in torch_names[-1] or 'conv' in torch_names or 'fc' in torch_names:
        target_layer.weight.data = convert_weights(conv_weights)
        target_layer.bias.data = torch.from_numpy(conv_bias)
    elif 'bn' in torch_names:
        target_layer.weight.data = torch.from_numpy(layer_weights[0])
        target_layer.bias.data = torch.from_numpy(layer_weights[1])
        target_layer.running_mean = torch.from_numpy(layer_weights[2])
        target_layer.running_var = torch.from_numpy(layer_weights[3])


#not saving the whole model but the weights
torch.save(torch_model.state_dict(), 'inbreast_vgg16_512x1.pth')
print(torch_model.state_dict())
