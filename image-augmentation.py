import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import os
from PIL import Image
import scipy

# パラメータを編集
# ここから
ROTATION_RANGE = 0
WIDTH_SHIFT_RANGE = 0
HEIGHT_SHIFT_RANGE = 0
SHEAR_RANGE = 0
ZOOM_RANGE = [0,1]
HORIZONTAL_FLIP = True
VERTICAL_FLIP = True
INPUT_PATH = ''
OUTPUT_PATH = ''
IMAGE_NUM = 0
IMAGE_WIDTH = 0
IMAGE_HEIGHT = 0
# ここまで

def pca_color_augmentation_modify(image_array_input):
    assert image_array_input.ndim == 3 and image_array_input.shape[2] == 3
    img = image_array_input.reshape(-1, 3).astype(np.float32)
    # 分散を計算
    ch_var = np.var(img, axis=0)
    # 分散の合計が3になるようにスケーリング
    scaling_factor = np.sqrt(3.0 / sum(ch_var))
    # 平均で引いてスケーリング
    img = (img - np.mean(img, axis=0)) * scaling_factor
    cov = np.cov(img, rowvar=False)
    lambd_eigen_value, p_eigen_vector = np.linalg.eig(cov)
    rand = np.random.randn(3) * 0.1
    delta = np.dot(p_eigen_vector, rand*lambd_eigen_value)
    delta = (delta * 255.0).astype(np.int32)[np.newaxis, np.newaxis, :]
    img_out = np.clip(image_array_input + delta, 0, 255).astype(np.uint8)
    return img_out

files = os.listdir(INPUT_PATH)
files.remove('.DS_Store')

index = 1

for file in files:

    img = Image.open(os.path.join(INPUT_PATH, file))
    img_resize = img.resize((IMAGE_WIDTH,IMAGE_HEIGHT))
    x = image.img_to_array(img_resize)
    x = x.reshape((1,) + x.shape)

    datagen = ImageDataGenerator(
            rotation_range = ROTATION_RANGE,
            width_shift_range = WIDTH_SHIFT_RANGE,
            height_shift_range = HEIGHT_SHIFT_RANGE,
            shear_range = SHEAR_RANGE,
            zoom_range = ZOOM_RANGE,
            horizontal_flip=HORIZONTAL_FLIP,
            vertical_flip=VERTICAL_FLIP)
    
    i = 1

    for d in datagen.flow(x, batch_size=1):
        temp = image.img_to_array(d[0])
        temp = pca_color_augmentation_modify(temp)
        cv2.imwrite(OUTPUT_PATH + "Aug_{0}_{1}.jpg".format(index,i), np.asarray(temp)[..., ::-1])
        if (i % IMAGE_NUM) == 0:
            print("image %d done" %(index))
            index += 1
            break
        i += 1
