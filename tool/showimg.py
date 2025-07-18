import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image


def show(path, mosaic_mode=None):
    row_col_size=(2,4)

    multi=row_col_size[0] * row_col_size[1]+1
    for i in range(1,multi):
        plt.subplot(2,4,i)
        flip_img = mosaic_mode(path)
        flip_img=np.array(flip_img)[:]
        plt.imshow(flip_img)
    plt.show()

if __name__ == '__main__':
    Random_flip=T.RandomHorizontalFlip()
    img_path='/home/dell609/dl_pro/VesselSeg/PANet/Datasets/ARIA/Test/Images/567.tif'
    img=Image.open(img_path)
    show(img,Random_flip)

