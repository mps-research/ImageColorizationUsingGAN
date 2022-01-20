from pathlib import Path
from PIL import Image, ImageOps
from torch.utils.data import Dataset


class Places365(Dataset):
    def __init__(self, root_dir, train=True, transform=None, target_transform=None):
        self._image_paths = list((Path(root_dir) / ('train' if train else 'val')).glob('*/*.jpg'))
        self._transform = transform
        self._target_transform = target_transform

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, idx):
        rgb_image = Image.open(self._image_paths[idx])
        gray_image = ImageOps.grayscale(rgb_image)
        if self._transform:
            gray_image = self._transform(gray_image)
        if self._target_transform:
            rgb_image = self._target_transform(rgb_image)
        return gray_image, rgb_image
