import cv2
import glob
import torch
import random
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import ImageEnhance


def cv_random_flip(img, edge, mask):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        edge = edge.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, edge, mask


def randomCrop(image, edge, mask):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), edge.crop(random_region), mask.crop(random_region)


def randomRotation(image, edge, mask):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        edge = edge.rotate(random_angle, mode)
        mask = mask.rotate(random_angle, mode)
    return image, edge, mask


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)


class DatasetGenerate(Dataset):
    def __init__(self, img_folder, gt_folder, edge_folder, train_size, phase: str = 'train', augmentation=False, seed=None):
        print(img_folder)
        self.images = sorted(glob.glob(img_folder + '/*'))
        self.gts = sorted(glob.glob(gt_folder + '/*'))
        self.edges = sorted(glob.glob(edge_folder + '/*'))
        self.augmentation = augmentation
        self.train_size = train_size
        self.onechannel_transform = transforms.Compose([
            transforms.Resize((self.train_size, self.train_size)),
            transforms.ToTensor()])
        self.train_transform = transforms.Compose([
            transforms.Resize((self.train_size, self.train_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225]),
        ])
        print(len(self.images))
        train_images, val_images, train_gts, val_gts, train_edges, val_edges = train_test_split(self.images, self.gts,
                                                                                                self.edges,
                                                                                                test_size=0.05,
                                                                                                random_state=seed)
        if phase == 'train':
            self.images = train_images
            self.gts = train_gts
            self.edges = train_edges
        elif phase == 'val':
            self.images = val_images
            self.gts = val_gts
            self.edges = val_edges
        else:  # Testset
            pass

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        mask = cv2.imread(self.gts[idx])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = Image.fromarray(mask)
        edge = cv2.imread(self.edges[idx])
        edge = cv2.cvtColor(edge, cv2.COLOR_BGR2GRAY)
        edge = Image.fromarray(edge)

        if self.augmentation is True:
            image, edge, mask = cv_random_flip(image, edge, mask)
            image, edge, mask = randomCrop(image, edge, mask)
            image, edge, mask = randomRotation(image, edge, mask)

            image = colorEnhance(image)
            mask = randomPeper(mask)

        image = self.train_transform(image)
        mask = self.onechannel_transform(mask)
        edge = self.onechannel_transform(edge)
        return image, mask, edge

    def __len__(self):
        return len(self.images)


class Test_DatasetGenerate(Dataset):
    def __init__(self, img_folder, gt_folder, train_size):
        self.images = sorted(glob.glob(img_folder + '/*'))
        self.gts = sorted(glob.glob(gt_folder + '/*'))
        self.train_size = train_size
        self.test_transform = transforms.Compose([
            transforms.Resize((self.train_size, self.train_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
        ])
    def __getitem__(self, idx):
        image_name = Path(self.images[idx]).stem
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]

        image = self.test_transform(Image.fromarray(image))

        return image, self.gts[idx], original_size, image_name

    def __len__(self):
        return len(self.images)


def get_loader(img_folder, gt_folder, edge_folder, train_size, phase: str, batch_size, shuffle,
               num_workers, augmentation, seed=None):
    if phase == 'test':
        dataset = Test_DatasetGenerate(img_folder, gt_folder, train_size)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    else:
        dataset = DatasetGenerate(img_folder, gt_folder, edge_folder, train_size, phase, augmentation, seed)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                 drop_last=True)

    print(f'{phase} length : {len(dataset)}')

    return data_loader


def gt_to_tensor(gt):
    gt = cv2.imread(gt)
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY) / 255.0
    gt = np.where(gt > 0.5, 1.0, 0.0)
    gt = torch.tensor(gt, device='cuda', dtype=torch.float32)
    gt = gt.unsqueeze(0).unsqueeze(1)

    return gt
