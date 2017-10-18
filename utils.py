"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import pprint
import scipy.misc
import numpy as np
import copy
import glob
import os.path
import time


pp = pprint.PrettyPrinter()
IMAGE_CACHE = {}


class ImagePool(object):

    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.num_img = 0
        self.images = []

    def __call__(self, image):
        if self.maxsize <= 0:
            return image
        if self.num_img < self.maxsize:
            self.images.append(image)
            self.num_img += 1
            return image
        if np.random.rand() > 0.5:
            idx = int(np.random.rand()*self.maxsize)
            tmp1 = copy.copy(self.images[idx])[0]
            self.images[idx][0] = image[0]
            idx = int(np.random.rand()*self.maxsize)
            tmp2 = copy.copy(self.images[idx])[1]
            self.images[idx][1] = image[1]
            return [tmp1, tmp2]
        else:
            return image


class ImageMemMap(object):

    def __init__(
        self,
        filename,
        filepattern,
        load_size=286,
        fine_size=256,
    ):
        self.filename = filename
        self.filepattern = filepattern
        self.filenames = glob.glob(self.filepattern)
        self.length = len(self.filenames)
        self.load_size = load_size
        self.fine_size = fine_size
        self.shape = (self.length, fine_size, fine_size, 3)

        if os.path.exists(self.filename):
            print("Loading memmap %s" % repr(self.filename))
            self.fp = np.memmap(
                self.filename,
                dtype='float32',
                mode='r',
                shape=self.shape,
            )
        else:
            self.prepare_train_data(self.filepattern)

    def __len__(self):
        return self.length

    def get_image(self, index):
        img_X = self.fp[index]
        if np.random.rand() > 0.5:
            img_X = np.fliplr(img_X)
        return img_X

    def prepare_train_data(
        self,
        images_pattern,
        flush_after=20,
    ):
        if os.path.exists(self.filename):
            print(
                "WARNING: Won't overwrite dataset %s, delete it manually."
                % self.filename
            )
            return
        image_names = glob.glob(images_pattern)
        fp = np.memmap(
            self.filename,
            dtype='float32',
            mode='w+',
            shape=self.shape,
        )

        time_start = time_end = time.time()
        for i, image_name in enumerate(image_names):
            img = self.get_reshaped_image(
                image_name, self.load_size, self.fine_size)
            fp[i] = img
            if i % flush_after == flush_after - 1:
                fp.flush()
            time_end = time.time()
            print(
                "[%d/%d] %2.4fs %s" % (
                    i,
                    len(image_names),
                    time_end - time_start,
                    repr(image_name)
                )
            )
            time_start = time_end
        fp.flush()
        del fp

        self.fp = np.memmap(
            self.filename,
            dtype='float32',
            mode='r',
            shape=self.shape,
        )

        return self.fp

    def imread(self, path, is_grayscale=False):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)

    def reshape_image(
        self,
        img_X,
        load_size,
        fine_size,
    ):
        if load_size != fine_size:
            img_X = scipy.misc.imresize(img_X, [load_size, load_size])
            h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
            w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
            img_X = img_X[h1:h1+fine_size, w1:w1+fine_size]

        return img_X

    def get_reshaped_image(self, image_path, load_size, fine_size):
        img_X = self.imread(image_path)
        img_X = self.reshape_image(img_X, load_size, fine_size)
        img_X = (img_X / 127.5) - 1.0
        return img_X


# def load_train_data(
#     image_path,
#     load_size=286,
#     fine_size=256,
# ):

#     img_X = scipy.misc.imread(image_path[0], mode='RGB').astype(np.float)
#     img_X = scipy.misc.imresize(img_X, [fine_size, fine_size])
#     img_X = (img_X / 127.5) - 1.0

#     return img_X


def get_stddev(x, k_h, k_w):
    return 1/math.sqrt(k_w*k_h*x.get_shape()[-1])


def load_test_data(image_path, fine_size=256):
    img = scipy.misc.imread(image_path, mode='RGB').astype(np.float)
    img = scipy.misc.imresize(img, [fine_size, fine_size])
    img = (img / 127.5) - 1.0
    return img


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def merge_images(images, size):
    return inverse_transform(images)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img


def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))


def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(
        x[j:j+crop_h,
        i:i+crop_w],
        [resize_h, resize_w],
    )


def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.


def inverse_transform(images):
    return (images+1.)/2.
