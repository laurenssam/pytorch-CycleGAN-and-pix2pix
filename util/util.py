"""This module contains simple helper functions """
from __future__ import print_function

from pathlib import Path

import torch
import numpy as np
from PIL import Image
import os
from torchvision.datasets import Cityscapes
import pickle
import torch.functional as F
from tqdm import tqdm


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def fakenize_real(label, fake, num_classes):
    batch_size, _, height, width = fake.shape
    max_fake = torch.max(fake, dim=1)[0].unsqueeze(dim=1) # N * 1 * H *  W
    max_fake = max_fake.repeat(1, num_classes, 1, 1) # N * C * H * W
    ignore_mask = label == 255
    label[ignore_mask] = 0
    label_one_hot = torch.nn.functional.one_hot(label, num_classes).permute(0, 3, 1, 2) # N * C * H * W
    fake_label_permuted = torch.zeros_like(fake)
    for i in range(batch_size):
        fake_label_permuted[i, :] = fake[i, torch.randperm(num_classes)]
    values_to_be_swapped = torch.max(fake_label_permuted * label_one_hot, dim=1)[0]
    values_to_be_swapped = values_to_be_swapped.unsqueeze(dim=1).repeat(1, num_classes, 1, 1)
    current_max_indices = fake_label_permuted == max_fake
    fake_label_permuted[current_max_indices] = 0
    fake_label_permuted += current_max_indices * values_to_be_swapped

    fake_label_permuted *= (1 - label_one_hot)
    fake_label_permuted += label_one_hot * max_fake

    return fake_label_permuted

# def calculate_class_weights(val_loader, num_classes):
#     file_path = Path("classweights.p")
#     if file_path.exists():
#         return pickle.load(open(file_path, "rb"))
#     counter = {cls_idx: 0 for cls_idx in range(num_classes)}
#     pixel_count = 0
#     count = 0
#     for _, mask in val_loader:
#         for cls_idx in range(num_classes):
#             count_cls = (mask == cls_idx).sum().item()
#             counter[cls_idx] += count_cls
#             pixel_count += count_cls
#         count += 1
#     weights = [(1./(float(count)/pixel_count)) for count in counter.values()]
#     weights = torch.FloatTensor(weights)
#     pickle.dump(weights, open(file_path, "wb"))
#     return weights

def calculate_class_weights(dataloader, num_classes):
    # Create an instance from the data loader
    file_path = Path("classweights.p")
    if file_path.exists():
        return pickle.load(open(file_path, "rb"))
    print("Calculating class weights")
    z = np.zeros((num_classes,))
    # Initialize tqdm
    tqdm_batch = tqdm(dataloader)
    print('Calculating classes weights')
    for _, y in tqdm_batch:
        y = y.detach().cpu().numpy()
        mask = (y >= 0) & (y < num_classes)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)
        z += count_l
    tqdm_batch.close()
    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
        class_weights.append(class_weight)
    ret = np.array(class_weights)
    weights = torch.FloatTensor(ret)
    pickle.dump(weights, open(file_path, "wb"))
    return weights

def mask_to_img(mask):
    height, width = mask.squeeze().shape
    new_img = torch.zeros(height , width, 3)
    for cls in Cityscapes.classes:
        new_img[mask == cls[2]] = torch.FloatTensor(list(cls[-1]))
    return new_img.numpy().astype(np.uint8)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
