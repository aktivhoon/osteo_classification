import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

class CAM():
    def __init__(self, model):
        self.gradient = []
        self.h = model.module.fc.register_backward_hook(self.save_gradient)

    def save_gradient(self, *args):
        grad_input = args[1]
        grad_output = args[2]
        self.gradient.append(grad_output[0])

    def get_gradient(self, idx):
        return self.gradient[idx]

    def remove_hook(self):
        self.h.remove()

    def normalize_cam(self, x):
        x = 2 * (x-torch.min(x))/(torch.max(x) - torch.min(x) + 1e-8) - 1
        x[x<torch.max(x)]=-1
        return x

    def visualize(self, cam_img, img_var):
        x = img_var[0, :, :].cpu().data.numpy()
        output = np.concatenate((x, cam_img, x+cam_img), axis = 1)
        print(output.shape)

    def get_cam(self, idx):
        grad = self.get_gradient(idx)
        print(grad.shape)
        alpha = torch.sum(grad, dim = -1, keepdim = True)
        alpha = torch.sum(alpha, dim = -2, keepdim = True)

        cam = alpha * grad
        cam = torch.sum(cam, dim = 0)
        cam = self.normalize_cam(cam)
        self.remove_hook()
        return cam    
