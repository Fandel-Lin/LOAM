import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, training_map: np, auxiliary_dict: dict, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.auxiliary_dict = auxiliary_dict.copy()

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.') and (('_'.join(splitext(file)[0].split('_')[:-4])) in training_map)]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])
        
        img_sup_0 = load_image(join(self.images_dir, 'sup', name+'_sup_0.png'))
        img_sup_1 = load_image(join(self.images_dir, 'sup', name+'_sup_1.png'))
        img_sup_2 = load_image(join(self.images_dir, 'sup', name+'_sup_2.png'))
        img_sup_3 = load_image(join(self.images_dir, 'sup', name+'_sup_3.png'))
        img_sup_4 = load_image(join(self.images_dir, 'sup', name+'_sup_4.png'))
        img_sup_5 = load_image(join(self.images_dir, 'sup', name+'_sup_5.png'))

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        img_sup_0 = self.preprocess(self.mask_values, img_sup_0, self.scale, is_mask=False)
        img_sup_1 = self.preprocess(self.mask_values, img_sup_1, self.scale, is_mask=False)
        img_sup_2 = self.preprocess(self.mask_values, img_sup_2, self.scale, is_mask=False)
        img_sup_3 = self.preprocess(self.mask_values, img_sup_3, self.scale, is_mask=False)
        img_sup_4 = self.preprocess(self.mask_values, img_sup_4, self.scale, is_mask=False)
        img_sup_5 = self.preprocess(self.mask_values, img_sup_5, self.scale, is_mask=False)

        img_combined = np.zeros((7, img.shape[1], img.shape[2]), dtype='uint8').astype(float)
        img_combined[0] = img
        img_combined[1] = img_sup_0
        img_combined[2] = img_sup_1
        img_combined[3] = img_sup_2
        img_combined[4] = img_sup_3
        img_combined[5] = img_sup_4
        img_combined[6] = img_sup_5

        #try:
            #fetching_auxi = self.auxiliary_dict[str('_'.join(name.split('_')[:-2]))][0]
        #except:
            #print(name, str('_'.join(name.split('_')[:-2])))

        return {
            #'image': torch.as_tensor(img.copy()).float().contiguous(),
            'image': torch.as_tensor(img_combined.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),
            'auxiliary_1': self.auxiliary_dict[str('_'.join(name.split('_')[:-2]))][0],
            'auxiliary_2': self.auxiliary_dict[str('_'.join(name.split('_')[:-2]))][1]
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, training_map, auxiliary_dict, scale=1):
        super().__init__(images_dir, mask_dir, training_map,auxiliary_dict, scale, mask_suffix='_mask')