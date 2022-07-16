import logging
import time
from pathlib import Path

import torch

from exceptions import CUDANotAvailable, NotFoundFolder


logger = logging.getLogger(__name__)


def check_args(args):
    if args.cuda is True and torch.cuda.is_available() is False:
        raise CUDANotAvailable("CUDA is not available. Is Pytorch installed "
                               "with cuda extensions? See https://pytorch.org/get-started/locally/")

    image_folder = Path(args.image_folder).expanduser()
    if not image_folder.exists():
        raise NotFoundFolder(f"{image_folder.resolve()} is not exists.")

    result_folder = Path(args.output_folder).expanduser()
    img_result_folder = result_folder / f"{int(time.time())}"
    img_result_folder.mkdir(parents=True)
    args.output_folder = str(img_result_folder.resolve())
    return args
