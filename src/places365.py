import shutil
import random
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageOps
from torch.utils.data import Dataset


def sample(x, n):
    return random.sample(x, min(n, len(x)))


def copy_class_images(src_image_paths, dst_class_dir):
    if not dst_class_dir.exists():
        dst_class_dir.mkdir()
    for src_image_path in src_image_paths:
        dst_image_path = dst_class_dir / src_image_path.name
        shutil.copyfile(src_image_path, dst_image_path)


def create_dataset(src_dir, dst_dir, n_classes, n_train_samples_per_class, n_val_samples_per_class):
    src_dir = Path(src_dir)
    if not src_dir.exists():
        print(f'{src_dir} does not exist.')
        raise FileNotFoundError

    dst_dir = Path(dst_dir)
    if not dst_dir.exists():
        dst_dir.mkdir()
    else:
        print(f'{dst_dir} already exists.')
        raise FileExistsError

    src_train_dir = Path(src_dir) / 'train'
    if not src_train_dir.exists():
        raise FileNotFoundError

    dst_train_dir = Path(dst_dir) / 'train'
    if not dst_train_dir.exists():
        dst_train_dir.mkdir()
    else:
        print(f'{dst_train_dir} already exists.')
        raise FileExistsError

    src_val_dir = src_dir / 'val'
    if not src_val_dir.exists():
        raise FileNotFoundError

    dst_val_dir = dst_dir / 'val'
    if not dst_val_dir.exists():
        dst_val_dir.mkdir()
    else:
        print(f'{dst_val_dir} already exists.')
        raise FileExistsError

    classes = random.sample([p.parts[-1] for p in src_train_dir.glob('*')], n_classes)

    for clazz in tqdm(classes):
        src_train_image_paths = list(src_train_dir.glob(f'{clazz}/*.jpg'))
        copy_class_images(sample(src_train_image_paths, n_train_samples_per_class), dst_train_dir / clazz)

        src_val_image_paths = list(src_val_dir.glob(f'{clazz}/*.jpg'))
        copy_class_images(sample(src_val_image_paths, n_val_samples_per_class), dst_val_dir / clazz)


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
