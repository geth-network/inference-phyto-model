import logging
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import segmentation_models_pytorch as smp
from cv2 import cv2
from PIL import Image
from tqdm import tqdm

from tools import check_args
from image_utils import image_preprocessing, get_phyto_model


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
preprocessing_fn = smp.encoders.get_preprocessing_fn("se_resnext50_32x4d",
                                                     "imagenet")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("-m", "--model_path", type=str,
                        help="path to model checkpoint (*.ckpt)")
    parser.add_argument("-f", "--image_folder", type=str, default="examples",
                        help="path to folder contains satellite images (*.png)")
    parser.add_argument("-o", "--output_folder", type=str, default="result",
                        help="path to folder contains result images")

    args = check_args(parser.parse_args())

    target_size = (384, 608)  # width, height
    images = list(Path(args.image_folder).glob("*.png"))
    for image_filepath in tqdm(images, desc="Image processing"):
        image = cv2.imread(str(image_filepath))
        image, image_original, padded_size = image_preprocessing(image,
                                                                 target_size)
        image_trans = preprocessing_fn(x=image).transpose(2, 0, 1).astype('float32')

        device = "cpu" if not args.cuda else "cuda"
        image_tensor = torch.from_numpy(image_trans).unsqueeze(0).to(device)
        model = get_phyto_model(args.model_path, args.cuda)

        pr_mask = model(image_tensor)
        pr_mask = pr_mask.sigmoid()
        pr_mask = (pr_mask.squeeze().cpu().detach().numpy().round())

        if padded_size is None and pr_mask.shape[:2] != image_original.shape[:2]:
            pr_mask_pil = Image.fromarray(pr_mask)
            pr_mask_pil_resized = pr_mask_pil.resize(image_original.shape[:2][::-1],
                                                     resample=Image.Resampling.BICUBIC)
            pr_mask = np.array(pr_mask_pil_resized)
        elif isinstance(padded_size, tuple):
            h_pad_top, h_pad_bottom, w_pad_left, w_pad_right = padded_size
            pr_mask = pr_mask[h_pad_top:pr_mask.shape[0] - h_pad_bottom,
                              w_pad_left:pr_mask.shape[1] - w_pad_right]

        image_segm = image_original.copy()
        image_mask_red = image_segm[:, :, 2]
        image_mask_red[pr_mask == 1] = 200
        image_segm[:, :, 2] = image_mask_red

        result_image_mask_path = Path(args.output_folder) / f"{image_filepath.stem}_segm.png"
        original_image_path = Path(args.output_folder) / image_filepath.name
        cv2.imwrite(str(result_image_mask_path), image_segm)
        cv2.imwrite(str(original_image_path), image_original)
        logger.debug(f"saved result of {image_filepath.name} segmentation to "
                    f"{result_image_mask_path.parent.resolve()}")
    logger.info(f"Result was saved to {args.output_folder}")