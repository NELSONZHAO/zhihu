# coding: utf-8

import os

from PIL import Image, ImageOps
import numpy as np
import scipy.misc
from six.moves import urllib


def download(download_link, file_name, expected_bytes):
    """
    下载pre-trained VGG-19
    
    :param download_link: 下载链接
    :param file_name: 文件名
    :param expected_bytes: 文件大小
    """
    if os.path.exists(file_name):
        print("VGG-19 pre-trained model is ready")
        return
    print("Downloading the VGG pre-trained model. This might take a while ...")
    file_name, _ = urllib.request.urlretrieve(download_link, file_name)
    file_stat = os.stat(file_name)
    if file_stat.st_size == expected_bytes:
        print('Successfully downloaded VGG-19 pre-trained model', file_name)
    else:
        raise Exception('File ' + file_name +
                        ' might be corrupted. You should try downloading it with a browser.')


def get_resized_image(img_path, width, height, save=True):
    """
    对图片进行像素尺寸的规范化
    
    :param img_path: 图像路径
    :param width: 像素宽度
    :param height: 像素高度
    :param save: 存储路径
    :return: 
    """
    image = Image.open(img_path)
    # PIL is column major so you have to swap the places of width and height
    image = ImageOps.fit(image, (width, height), Image.ANTIALIAS)
    if save:
        image_dirs = img_path.split('/')
        image_dirs[-1] = 'resized_' + image_dirs[-1]
        out_path = '/'.join(image_dirs)
        if not os.path.exists(out_path):
            image.save(out_path)
    image = np.asarray(image, np.float32)
    return np.expand_dims(image, 0)


def generate_noise_image(content_image, width, height, noise_ratio=0.6):
    """
    对原图片增加白噪声
    
    :param content_image: 内容图片
    :param width: 图片width
    :param height: 图片height
    :param noise_ratio: 噪声比例
    :return: 带有噪声的内容图片
    """
    noise_image = np.random.uniform(-20, 20, (1, height, width, 3)).astype(np.float32)
    return noise_image * noise_ratio + content_image * (1 - noise_ratio)


def save_image(path, image):
    image = image[0]
    image = np.clip(image, 0, 255).astype('uint8')
    scipy.misc.imsave(path, image)


def safe_mkdir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass