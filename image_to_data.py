import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import h5py

def norm_image(img):
    """
    Normalize PIL image

    Normalizes luminance to (mean,std)=(0,1), and applies a [1%, 99%] contrast stretch
    """
    img_y, img_b, img_r = img.convert('YCbCr').split()

    img_y_np = np.asarray(img_y).astype(float)

    img_y_np /= 255
    img_y_np -= img_y_np.mean()
    img_y_np /= img_y_np.std()
    scale = np.max([np.abs(np.percentile(img_y_np, 1.0)),
                    np.abs(np.percentile(img_y_np, 99.0))])
    img_y_np = img_y_np / scale
    img_y_np = np.clip(img_y_np, -1.0, 1.0)
    img_y_np = (img_y_np + 1.0) / 2.0

    img_y_np = (img_y_np * 255 + 0.5).astype(np.uint8)

    img_y = Image.fromarray(img_y_np)

    img_ybr = Image.merge('YCbCr', (img_y, img_b, img_r))

    img_nrm = img_ybr.convert('RGB')

    return img_nrm

def resize_image(img, size):
    """
    Resize PIL image

    Resizes image to be square with sidelength size. Pads with black if needed.
    """
    # Resize
    n_x, n_y = img.size
    if n_y > n_x:
        n_y_new = size
        n_x_new = int(size * n_x / n_y + 0.5)
    else:
        n_x_new = size
        n_y_new = int(size * n_y / n_x + 0.5)

    img_res = img.resize((n_x_new, n_y_new), resample=Image.BICUBIC)

    # Pad the borders to create a square image
    img_pad = Image.new('RGB', (size, size), (128, 128, 128))
    ulc = ((size - n_x_new) // 2, (size - n_y_new) // 2)
    img_pad.paste(img_res, ulc)

    return img_pad


def prep_images(path):

        img = Image.open(path)
        img_nrm = norm_image(img)
        img_res = resize_image(img_nrm, 224)
        return img_res


if __name__ == '__main__':
    df = pd.read_csv('categorical_csv_128.csv')
    X = []
    y_sex = []
    y_emotion = []
    y_age = []
    y_fd = []
    y_multi = []
    for i in range(len(df)):
        img = df.iloc[i]['image']
        img_final = prep_images(img)
        X.append(np.array(img_final))
        y_sex.append(df.iloc[i]['sex_cat'])
        y_emotion.append(df.iloc[i]['emotion_cat'])
        y_age.append(df.iloc[i]['age_cat'])
        y_fd.append(df.iloc[i]['face_dir_cat'])
    X = np.array(X)
    y_sex = np.array(y_sex)
    y_emotion = np.array(y_emotion)
    y_age = np.array(y_age)
    y_fd = np.array(y_fd)


    filed = h5py.File('128_data_file.hdf5', 'w')
    filed.create_dataset('X', data=X)
    filed.create_dataset('y_sex', data=y_sex)
    filed.create_dataset('y_emotion', data=y_emotion)
    filed.create_dataset('y_age', data=y_age)
    filed.create_dataset('y_fd', data=y_fd)
    filed.close()

    print(X.shape, y_sex.shape)
