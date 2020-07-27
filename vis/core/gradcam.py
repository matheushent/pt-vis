"""
Core module for Grad-CAM algorithm
"""
import numpy as np
import torch
import cv2

class GradCam:
    """
    Utility class to perform Grad-CAM for a given input

    Paper: [Grad-CAM: Visual Explanations from Deep Networks
            via Gradient-based Localization](https://arxiv.org/abs/1610.02391)
    """

    def explain(self, img, model, class_index, layer=None, guided=True):

        if not isinstance(layer, int):
            raise ValueError("'layer' input is {} while an int required.".format(type(layer)))
        
        if not layer:
            layer = 11 # select 11th layer by default

        outputs, grads = self.get_gradients_and_filters(model, img, class_index, layer)

        cam = GradCam.generate_output(outputs, grads, img)

        if guided:
            guided_grads = self.get_guided_gradients(model, img, class_index)

            return np.multiply(cam, guided_grads)

        return cam

    def get_gradients_and_filters(self, model, inputs, class_index, layer):

        x = inputs

        for i, module in model.features._modules.items():
            x = module(x) # foward

            if int(i) == layer:
                x.register_hook(self.save_gradient)
                output = x

        x = x.view(x.size(0), -1) # flatten

        x = model.classifier(x)

        one_hot_output = torch.FloatTensor(1, x.size()[-1]).zero_()
        one_hot_output[0][class_index] = 1

        # zero gradients
        model.features().zero_grad()
        model.classifier().zero_grad()

        # backward pass
        x.backward(gradient=one_hot_output, retain_graph=True)

        return self.gradient.data.numpy()[0], output.data.numpy()[0]

    def save_gradient(self, grad):
        self.gradient = grad

    @staticmethod
    def generate_output(outputs, grads, img):

        weights = np.mean(grads, axis=(1, 2))

        cam = np.zeros(outputs.shape[:1], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * outputs[i, :, :]

        cam = np.maximum(cam, 0) # return max element-wise
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)
        cam = np.uint8(cam, tuple(img.shape[2:]))
        
        return cam

    def get_guided_gradients(self, model, inputs, class_index):

        self.forward_relu_outputs = []

        def relu_backward_hook_function(module, grad_in, grad_out):

            forward_output = self.forward_relu_outputs[-1]
            forward_output[forward_output > 0] = 1
            modified_grad_out = forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]
            return (modified_grad_out,)
        
        def relu_forward_hook_function(module, in_tensor, out_tensor):

            self.forward_relu_outputs.append(out_tensor)

        for i, module in model.features()._modules.items():

            if isinstance(module, torch.nn.ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        
        first_layer = list(model.features._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

        output = model(inputs)
        model.zero_grad()

        one_hot_output = torch.FloatTensor(1, output.size()[-1]).zero_() # Flatten
        one_hot_output[0][class_index] = 1

        output.backward(gradient=one_hot_output)

        return self.gradients.data.numpy()[0]