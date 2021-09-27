from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
import csv

class Fair_Face(data.Dataset):
    """ Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs_train, selected_attrs_val, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs_val
        # self.selected_attrs_val = selected_attrs_val
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()
        self.imageX = None
        # exit()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        lines = []
        """Preprocess the Fair_Face attribute file."""
        with open(self.attr_path) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for row in readCSV:
                lines.append(row)
                # print(row[1:4])
        # lines = readCSV

        # lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1:]
        # print(all_attr_names)

        # for i, attr_name in enumerate(all_attr_names):
        #     print(attr_name)
        #     # self.attr2idx[attr_name] = i
        #     exit()
        #     self.idx2attr[i] = attr_name
        # lines = lines[2:]
        # random.seed(1234)
        # random.shuffle(lines)
        for i, line in enumerate(all_attr_names):
            # print(line)
            filename = line[0]
            values = line[1:4]
            # print(values)
            # print(filename)
            label = values
            # exit()
            # for attr_name in self.selected_attrs:
            #     # idx = self.attr2idx[attr_name]
            #     print( self.selected_attrs)
            #     print(attr_name)
            #     label.append(attr_name)
            #     print([filename, label])

            if (i+1) < 2000:
                # print(label)
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])
        print("finito")
        print(self.train_dataset)
        # exit()
        exit()
        print('Finished preprocessing the Fair_Face dataset...')

    def get_dataX(self):
        dataset = self.train_dataset
        return dataset

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image  = Image.open(os.path.join(self.image_dir, filename)).convert('RGB')

        self.imageX = get_2nd_dir(self.image_dir)
        self.imageX = Image.open(os.path.join(self.imageX, filename)).convert('RGB')

        return self.transform(image), self.transform(self.imageX), torch.FloatTensor(label), filename

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_2nd_dir(dir):
    """Select the second directory ."""
    x = "/media/electronicshelf/3F8C28A65EBC23F1/CelebAMask-HQ"
    A = "CelebA-HQ-White-Background"
    B = "CelebA-HQ-White-Background-Sketch"
    dir_domain = str(dir).split("/")
    # print(dir_domain)
    # exit()
    dir_domain = dir_domain[5]


    if dir_domain == A:
        # print(os.path.join(x, B))
        return os.path.join(x, B)
    else:
        # print(os.path.join(x, A))
        return os.path.join(x, A)


def get_loader(image_dir, attr_path, selected_attrs, selected_attrs_val, crop_size, image_size,
               batch_size=16, dataset='Fair_Face', mode='train', num_workers=1):
    """Build and return a data loader."""
    dataX = None
    transform = []
    # print("try")
    # if mode == 'train':
        # transform.append(T.RandomHorizontalFlip())
    # transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    if dataset == 'Fair_Face':
        print(dataset)
        dataset = Fair_Face(image_dir, attr_path, selected_attrs, selected_attrs_val, transform, mode)
        dataX = Fair_Face(image_dir, attr_path, selected_attrs,selected_attrs_val,  transform, mode).get_dataX()

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode == 'train'),
                                  num_workers=num_workers)
    return data_loader, dataX
