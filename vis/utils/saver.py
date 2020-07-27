"""Core module for saving related operations"""
from vis.utils import image
from PIL import Image
import numpy as np
import os

def save_gradient(gradient, folder, filename):
    """Utility class to save a gradient as an image

    Args:
        gradient (numpy.ndarray): 3D-numpy of shape (3, 224, 224)
        folder (str): Output folder
        filename (str): String to name output file
    """

    if not os.path.exists(folder):
        os.makedirs(folder)

    # normalize
    gradient -= gradient.min()
    gradient /= gradient.max()

    img = image.format_output(gradient)
    img = Image.fromarray(img)

    img.save(os.path.join(folder, filename))