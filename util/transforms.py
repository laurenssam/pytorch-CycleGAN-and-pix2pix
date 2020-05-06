import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.datasets import Cityscapes
import random

def joint_transform_train(crop_size):
    def augmentations(input_image, mask):
        height, width = input_image.size
        if random.random() > 0.5:
            input_image = TF.hflip(input_image)
            mask = TF.hflip(mask)
        # angle = random.randint(-5, 5)
        # input_image = TF.rotate(input_image, angle)
        # mask = TF.rotate(mask, angle)

        left = random.randint(0, height - crop_size)
        top = random.randint(0, width - crop_size)
        input_image = TF.crop(input_image, top, left, crop_size, crop_size)
        mask = TF.crop(mask, top, left, crop_size, crop_size)

        input_image = TF.to_tensor(input_image)
        input_image = TF.normalize(input_image, [0.28689554, 0.32513303, 0.28389177], [0.18696375, 0.19017339, 0.18720214])

        mask = torch.from_numpy(np.array(mask, dtype=np.int32)).long()
        for cls in Cityscapes.classes:
            mask[mask == cls[1]] = cls[2] ## map id to train_id
        return input_image, mask
    return augmentations

def joint_transform_val(input_image, mask):
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
