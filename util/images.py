import numpy as np
import torch


# Inverse Linear Transformation and convert it to numpy
def img2photo(imgs, v_range):
    return (img_depreprocess(imgs, v_range)).transpose(1,2).transpose(2,3).detach().cpu().numpy().astype(np.uint8)

# Linear Transformation given range
def img_preprocess(img, v_range):
    if v_range == [0,1]:
        return img / 255
    elif v_range == [-1, 1]:
        return img / 127.5 - 1
    else:
        return img

# Inverse Linear Transformation given range
def img_depreprocess(img, v_range):
    if v_range == [0,1]:
        return img * 255
    elif v_range == [-1, 1]:
        return (img + 1) * 127.5
    else:
        return img
