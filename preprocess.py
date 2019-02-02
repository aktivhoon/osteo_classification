import random
import numpy as np
from scipy.ndimage import gaussian_filter, zoom, rotate, map_coordinates

import warnings
warnings.filterwarnings("ignore")

def cut_out(input_, target_):
    dx = 20
    dy = 20

    cx = random.randint(0, input_.shape[1] - dx)
    cy = random.randint(0, input_.shape[0] - dy)

    input_[cy:cy + dy, cx:cx + dx] = 0

    return input_, target_


def random_crop2d(input_, target_):
    zoom_rates = [1.2, 1.3, 1.4]
    zoom_rate  = random.choice(zoom_rates)


    zoom_input  = zoom(input_,  zoom_rate)
    zoom_target = zoom(target_, zoom_rate)

    zoom_shape, img_shape = zoom_input.shape, input_.shape
    dx = random.randint(0, zoom_shape[0] - img_shape[0])
    dy = random.randint(0, zoom_shape[1] - img_shape[1])

    zoom_input  = zoom_input[dx:dx + img_shape[0], dy:dy + img_shape[1]]
    zoom_target = zoom_target[dx:dx + img_shape[0], dy:dy + img_shape[1]]
    return zoom_input, zoom_target


def random_flip2d(input_, target_):
    flip = random.randint(0, 2) # 0, 1, 2

    if   flip == 0:
        return input_[:, ::-1],    target_[:, ::-1]
    elif flip == 1:
        return input_[::-1, :],    target_[::-1, :]
    elif flip == 2:
        return input_[::-1, ::-1], target_[::-1, ::-1]
    elif flip == 3:
        return input_, target_

def image_resize(img, zoom=1):
    pass

def random_rotate2d(input_, target_):
    angle = random.randint(10, 350)
    rotate_input  = rotate(input_, angle,
                           reshape=False)
    rotate_target = rotate(target_, angle,
                           reshape=False)
    return rotate_input, rotate_target

# elastic transformation applied to 2d image (will be called by --augment elastic)
def elastic_transform(input_, param_list=None, random_state=None):    
    if param_list is None:
        param_list = [(1, 1), (5, 2), (1, 0.5), (1, 3)]
    alpha, sigma = random.choice(param_list)

    assert len(input_.shape)==2
    shape = input_.shape

    if random_state is None:
       random_state = np.random.RandomState(None)    

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    #print(np.mean(dx), np.std(dx), np.min(dx), np.max(dx))

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
    
    transformed = []
    print(shape)
    for image in input_:
        new = np.zeros(shape)
        if len(shape) == 3:
            for i in range(image.shape[2]):
                new[:, :, i] = map_coordinates(image[:, :, i], indices, order=1, mode="reflect").reshape(shape)
        else:
            new[:, :] = map_coordinates(image[:, :], indices, order=1, mode="reflect").reshape(shape)
        transformed.append(new)
    return transformed

# elastic transformation applied to concatenated 2d images (will be called by --augment cat_elastic)
def elastic_transform_2(input_, param_list=None, random_state=None):
    if param_list is None:
        param_list = [(1, 1), (5, 2), (1, 0.5), (1, 3)]
    alpha, sigma = random.choice(param_list)

    assert len(input_.shape)==3
    shape_2d = input_.shape[1:3]
    shape = input_.shape
    if random_state is None:
        random_state = np.random.RandomState(None)

    dx = gaussian_filter((random_state.rand(*shape_2d) * 2 - 1), sigma, mode="constant", cval = 0) * alpha
    dy = gaussian_filter((random_state.rand(*shape_2d) * 2 - 1), sigma, mode="constant", cval = 0) * alpha

    x, y = np.meshgrid(np.arange(shape_2d[0]), np.arange(shape_2d[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

    channel_count = 0
    new = np.zeros(shape)
    channel = 0
    for image in input_:
        if len(shape) == 4:
            for i in range(image.shape[3]):
                new[:, channel, :, i] = map_coordinates(image[:, channel, :, i], indices, order=1, mode="reflect").reshape(shape_2d)
        else:
            if channel == 0:
                new[channel, :, :] = map_coordinates(image[:, :], indices, order=1, mode="reflect").reshape(shape_2d)
            else:
                new[channel, :, :] = image[:, :]
        channel += 1
    return new

ARG_TO_DICT = {
        "crop":random_crop2d,
        "flip":random_flip2d,        
        "elastic":elastic_transform,
        "rotate":random_rotate2d,
        "cut":cut_out,
        "cat_elastic":elastic_transform_2
        }

def get_preprocess(preprocess_list):
    if not preprocess_list:
        return []
    return [ARG_TO_DICT[p] for p in preprocess_list.split(",")]
