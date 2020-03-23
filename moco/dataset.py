from __future__ import print_function

import zipfile
import io
import torch
import torchvision.datasets as datasets
from moco.utils.zipdataset import ImageZipFolder
from moco.utils.zipdatasetv2 import ImageZipFolder as ImageZipFolderV2


class ImageFolderInstance(datasets.ImageFolder):
    """Folder dataset which returns the index of the image as well
    """

    def __init__(self, root, transform=None, target_transform=None, two_crop=False):
        super(ImageFolderInstance, self).__init__(root, transform, target_transform)
        self.two_crop = two_crop

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        image = self.loader(path)
        if self.transform is not None:
            img = self.transform(image)
        else:
            img = image
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.two_crop:
            img2 = self.transform(image)
            img = torch.cat([img, img2], dim=0)

        return img, target


class ImageZipInstance(ImageZipFolder):
    """Folder dataset which returns the index of the image as well
    """

    def __init__(self, root, transform=None, target_transform=None, two_crop=False):
        super(ImageZipInstance, self).__init__(root, transform, target_transform)
        self.two_crop = two_crop

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        image = self.loader(path)
        if self.transform is not None:
            img = self.transform(image)
        else:
            img = image
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.two_crop:
            img2 = self.transform(image)
            img = torch.cat([img, img2], dim=0)

        return img, target


class ImageZipInstanceV2(ImageZipFolderV2):
    """Folder dataset which returns the index of the image as well
    """

    def __init__(self, root, transform=None, target_transform=None, two_crop=False):
        super(ImageZipInstanceV2, self).__init__(root, transform, target_transform)
        self.two_crop = two_crop
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is class_index of the target class.
        """
        # path, target = self.imgs[index]
        # image = self.loader(path)

        # ori_image, image, target = super(ImageZipInstanceV2, self).__getitem__(index)
        # print(self.zip_file)
        buffer_name, target = self.samples[index]
        zip_file = zipfile.ZipFile(self.zip_file_name, 'r')
        buffer = zip_file.read(buffer_name)
        image = self.loader(io.BytesIO(buffer))

        if self.transform is not None:
            img = self.transform(image)
        else:
            img = image
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.two_crop:
            img2 = self.transform(image)
            img = torch.cat([img, img2], dim=0)

        return img, target
