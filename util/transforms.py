import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.datasets import Cityscapes
import random
from copy import deepcopy

def joint_transform_train(crop_size, context_info):
    def augmentations(input_image, mask):
        height, width = input_image.size
        new_height, new_width = height // 2, width // 2
        image = TF.resize(input_image, new_width)
        mask = TF.resize(mask, (new_width, new_height), Image.NEAREST)
        flip = random.random() > 0.5
        if flip:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        # angle = random.randint(-5, 5)
        # input_image = TF.rotate(input_image, angle)
        # mask = TF.rotate(mask, angle)

        left = random.randint(0, new_height - crop_size)
        top = random.randint(0, new_width - crop_size)
        image = TF.crop(image, top, left, crop_size, crop_size)
        mask = TF.crop(mask, top, left, crop_size, crop_size)
        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.28689554, 0.32513303, 0.28389177], [0.18696375, 0.19017339, 0.18720214])

        mask = torch.from_numpy(np.array(mask, dtype=np.int32)).long()
        for cls in Cityscapes.classes:
            mask[mask == cls[1]] = cls[2] ## map id to train_id
        return image, mask
    return augmentations

def joint_transform_val(input_image, mask, context_info):
    height, width = input_image.size
    input_image = TF.resize(input_image, width // 2)
    mask = TF.resize(mask, (width // 2, height // 2), Image.NEAREST)
    input_image = TF.to_tensor(input_image)
    input_image = TF.normalize(input_image, [0.28689554, 0.32513303, 0.28389177], [0.18696375, 0.19017339, 0.18720214])

    mask = torch.from_numpy(np.array(mask, dtype=np.int32)).long()

    for cls in Cityscapes.classes:
        mask[mask == cls[1]] = cls[2]  ## map id to train_id

    return input_image, mask

def denormalize(input_image):
    mean = [0.28689554, 0.32513303, 0.28389177]
    std = [0.18696375, 0.19017339, 0.18720214]
    return TF.normalize(input_image, [-mean[i]/std[i] for i in range(len(mean))], [1/value for value in std])
