from torchvision.datasets import Cityscapes
from torch.utils.data import DataLoader
from util.transforms import joint_transform_train, joint_transform_val


def get_loaders_cityscapes(root_path, opt):
    training_dataset = Cityscapes(root_path, split='train', mode='fine',
                         target_type='semantic',
                         transforms=joint_transform_train(opt.crop_size))
    opt.output_nc = len(Cityscapes.train_id_to_name)
    training_loader = DataLoader(training_dataset, batch_size=opt.batch_size, shuffle=True)
    train_dataset_size = len(training_dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % train_dataset_size)

    validation_dataset = Cityscapes(root_path, split='val', mode='fine',
                         target_type='semantic',
                         transforms=joint_transform_val)
    opt.output_nc = len(Cityscapes.train_id_to_name)
    val_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
    val_dataset_size = len(validation_dataset)    # get the number of images in the dataset.
    print('The number of validation images = %d' % val_dataset_size)

    train_eval_set = Cityscapes(root_path, split='train', mode='fine',
                         target_type='semantic',
                         transforms=joint_transform_val)
    train_eval_set.images = train_eval_set.images[:500]
    train_eval_loader = DataLoader(train_eval_set, batch_size=1, shuffle=False)
    return training_loader, val_loader, train_eval_loader
