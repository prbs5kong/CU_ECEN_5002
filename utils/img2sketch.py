import numpy as np
import cv2
import os
from glob import glob
from tqdm import tqdm

if __name__ == "__main__":
    # img_list = glob('CelebA-HQ/CelebA-HQ-img/*.jpg') # img number from 0 to 29999
    mask_dir = '../CelebA-HQ/CelebAMask-HQ-mask-anno' # masks of 2000 images per directory (0, 1, 2, ..., 14)
    # set directory to save sketches
    save_dir = 'sketch'
    os.makedirs(save_dir, exist_ok=True)
    modes = ['train', 'val', 'test']
    txt_list = ['data_split/training.txt', 'data_split/validation.txt', 'data_split/test.txt']
    for mode, txt_path in zip(modes, txt_list):
        with open(txt_path, 'r') as f:
            img_list = f.read().splitlines()
        res_dir = 'sketch/' + mode
        os.makedirs(res_dir, exist_ok=True)
        for img_path in tqdm(img_list):
            img_n = int(img_path)
            # get the list of masks for a single image
            masks = glob(os.path.join(mask_dir, str(img_n//2000), '%05d_*.png' % img_n))
            mask = cv2.imread(masks[0])
            for m in masks[1:]:
                mask = mask | cv2.imread(m)
            img = cv2.imread('../CelebA-HQ/CelebA-HQ-img/{}.jpg'.format(img_n)) 
            # mask out background
            mask = cv2.resize(mask, (img.shape[0], img.shape[1]))
            img[mask==0] = 0
            # convert an image from one color space to another
            grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            invert = cv2.bitwise_not(grey_img)  # helps in masking of the image
            # sharp edges in images are smoothed while minimizing too much blurring
            # set sigma value to one-eight of the image width(height), sigma must be odd value
            sig = img.shape[0] // 4 - 1
            blur = cv2.GaussianBlur(invert, (sig, sig), 0)
            invertedblur = cv2.bitwise_not(blur)
            sketch = cv2.divide(grey_img, invertedblur, scale=256.0)
            # set background as white to ensure that it looks like a sketch
            sketch[mask[:,:,0]==0] = 255
            # save sketch to save_dir
            cv2.imwrite(
                os.path.join(res_dir, '{}.jpg'.format(img_n)), sketch
            ) 