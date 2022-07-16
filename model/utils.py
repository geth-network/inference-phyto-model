import albumentations as albu


def get_training_augmentation():
    train_transform = [
        albu.PadIfNeeded(min_height=384, min_width=608, always_apply=True,
                         border_mode=0),
        albu.RandomCrop(height=384, width=608, always_apply=True),
    ]
    return albu.Compose(train_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)
