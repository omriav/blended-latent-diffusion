import cv2
import numpy as np


def _get_bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def poisson_seamless_clone(
    source_image: np.ndarray, destination_image: np.ndarray, mask: np.ndarray
):
    rmin, rmax, cmin, cmax = _get_bbox(mask)
    source_image = source_image[rmin:rmax, cmin:cmax, :]
    mask = mask[rmin:rmax, cmin:cmax]
    center = ((cmin + cmax) // 2 + 1, (rmin + rmax) // 2 + 1)

    clone = cv2.seamlessClone(
        src=(source_image * 255).astype(np.uint8),
        dst=(destination_image * 255).astype(np.uint8),
        mask=(mask * 255).astype(np.uint8),
        p=center,
        flags=cv2.NORMAL_CLONE,
    )
    clone = (clone / 255).astype(np.float32)

    return clone
