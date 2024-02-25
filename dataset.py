#DATASET.PY


from glob import glob
from PIL import Image
import random
from torch.utils.data import Dataset


class Places2(Dataset):
    def __init__(self, data_root, img_transform, mask_transform, data='train'):
        super(Places2, self).__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform

        # get the list of image paths
        if data == 'train':
            self.paths = glob('{}/train/image/*.*'.format(data_root),
                              recursive=True)
            self.mask_paths = glob('{}/train/binary_mask/*.*'.format(data_root))
        else:
            self.paths = glob('{}/val_256/*.jpg'.format(data_root, data))
            self.mask_paths = glob('{}/val_mask/*.png'.format(data_root))

        self.N_mask = len(self.mask_paths)
        print(len(self.mask_paths,), len(self.paths))

    def __len__(self):
        if self.paths>self.mask_paths:
          return len(self.mask_paths)
        else:
          return len(self.paths)

    def __getitem__(self, index):
        print(index)
        img = self._load_img(self.paths[index])
        img = self.img_transform(img.convert('RGB'))
        mask = Image.open(self.mask_paths[random.randint(0, self.N_mask - 1)])
        mask = self.mask_transform(mask.convert('RGB'))
        return img * mask, mask, img

    def _load_img(self, path):
        img = Image.open(path)
        return img
