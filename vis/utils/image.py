"""Core module for image related operations"""
from PIL import Image
import numpy as np
import torch

def preprocess_input(path):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    image = Image.open(path).convert('RGB')
    image = image.resize((224, 224), Image.ANTIALIAS)

    image = np.float32(image)
    image = image.transpose(2, 0, 1)

    for channel, _ in enumerate(image):
        image[channel] /= 255
        image[channel] -= mean[channel]
        image[channel] /= std[channel]

    tensor = torch.from_numpy(image).float()
    tensor.unsqueeze_(0) # https://pytorch.org/docs/stable/torch.html#torch.unsqueeze -> add a dimension of size 1 in axis 0

    return torch.autograd.Variable(tensor, requires_grad=True)

def format_output(array):
    """Utility function to format an array to output it

    Args:
        array (numpy.ndarray): 3D-numpy of shape (3, 224, 224)

    Returns:
        numpy.ndarray of shape (224, 224, 3)
    """

    array = array.transpose(1, 2, 0)

    if np.max(array) <= 1:
        array *= 255
        array = array.astype(np.uint8)

    return array