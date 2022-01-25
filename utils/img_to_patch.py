'''
An accidental chance, I know the the patches means a special convlution. 
So when I decide to implentment the img_to_patch, 
it occured me that I could us the im2col function to solve this problem.
'''

import numpy as np
import sys,os
sys.path.append(os.getcwd())
print(os.getcwd())

class Img2Patch():

    def __init__(self, kernel_size: int=3, stride: int=1, padding: int=0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = padding
    
    def im2col(self, x):
        '''
        x's shape: [B, C, H, W]
        2d convolution using im2col method.
        '''
        # 2d convolution module using im2col method.
        input_data = x
        filter_h, filter_w = self.kernel_size, self.kernel_size
        stride = self.stride
        pad = self.pad
        N, C, H, W = input_data.shape
        out_h = (H + 2 * pad - filter_h) // stride + 1
        out_w = (W + 2 * pad - filter_w) // stride + 1

        img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
        col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

        for y in range(filter_h):
            y_max = y + stride * out_h
            for x in range(filter_w):
                x_max = x + stride * out_w
                col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
        return col.reshape(N, out_h*out_w, -1)

    def __call__(self, x):
        col = self.im2col(x)
        return col
        

if __name__ == "__main__":
    import torchvision
    import matplotlib.pyplot as plt
    import torch
    # from utils.dot_product import scaled_dot_product

    # import the dataset
    from data.dataset import get_dataset
    # from torchvision.datasets import CIFAR10
    train_set, test_set = get_dataset("MNIST", "MNIST", "~/data/MNIST")

    # Visualize some examples
    NUM_IMAGES = 4
    CIFAR_images = torch.stack([test_set[idx][0] for idx in range(NUM_IMAGES)], dim=0)
    print(CIFAR_images.shape)
    im2patch = Img2Patch(kernel_size=4, stride=4)
    img_patches = im2patch(CIFAR_images)
    print(img_patches.shape)


