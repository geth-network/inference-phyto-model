import logging
from typing import Tuple, Optional

import albumentations as albu
import cv2.cv2 as cv2
import numpy as np
from PIL import Image
from albumentations.augmentations.functional import pad_with_params

from model import PhytoModel

logger = logging.getLogger(__name__)
model = None


def get_phyto_model(path: str, use_cuda: bool = False) -> PhytoModel:
    global model
    if model is None:
        logger.info("Start to load new instance of PhytoModel")
        model = PhytoModel.load_from_checkpoint(path, map_location="cpu")
        if use_cuda is True:
            model.to("cuda")
            logger.info("USE CUDA")
        model.eval()
    else:
        logger.debug("Model is already loaded.")
    return model


def padding_img(img: np.array, min_height: int, min_width: int,
                border_mode=cv2.BORDER_REFLECT_101) -> Tuple[np.ndarray,
                                                             tuple]:

    rows, cols = img.shape[:2]

    if rows < min_height:
        h_pad_top = int((min_height - rows) / 2.0)
        h_pad_bottom = min_height - rows - h_pad_top
    else:
        h_pad_top = 0
        h_pad_bottom = 0

    if cols < min_width:
        w_pad_left = int((min_width - cols) / 2.0)
        w_pad_right = min_width - cols - w_pad_left
    else:
        w_pad_left = 0
        w_pad_right = 0

    padded_img = pad_with_params(
        img,
        h_pad_top,
        h_pad_bottom,
        w_pad_left,
        w_pad_right,
        border_mode=border_mode,
    )
    return padded_img, (h_pad_top, h_pad_bottom, w_pad_left, w_pad_right)


def image_preprocessing(img: np.ndarray,
                        target_size: Tuple[int, int]) -> Tuple[np.ndarray,
                                                               np.ndarray,
                                                               Optional[tuple]]:
    image_original = img.copy()
    if image_original.shape[2] == 4:
        image_original = cv2.cvtColor(image_original, cv2.COLOR_BGRA2BGR)
    image = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
    padded_sizes = None
    if image.shape[0] > target_size[0] or image.shape[1] > target_size[1]:
        im_pil = Image.fromarray(image)
        im_pil_resized = im_pil.resize(target_size[::-1],
                                       resample=Image.Resampling.BICUBIC)
        image = np.array(im_pil_resized)
    else:
        image, padded_sizes = padding_img(image, target_size[0],
                                          target_size[1])
    return image, image_original, padded_sizes
