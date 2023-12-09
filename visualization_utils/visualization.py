import einops
import numpy as np
import torch
import cv2
import copy
from skimage.measure import find_contours
from .color import COLOR_PALETTE


def parse_tensor(func):
    def wrapper(tensor, *args):
        try:
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.detach().cpu().numpy()
        except:
            'torch is not use here'
        tensor = copy.deepcopy(tensor)
        tensor = np.array(tensor)
        return func(tensor, *args)
    return wrapper


@parse_tensor
def parse_image(image):
    if image.dtype != np.uint8:
        # image -= np.min(image) # do not scale because we don't know image distribution
        image /= np.max(image)/255
        image = image.astype(np.uint8)

    if len(image.shape) == 2:  # no batch, no channels
        image = einops.repeat(image, 'h w -> h w c', c=3)

    if len(image.shape) == 4:  # batch and channels
        if image.shape[0] == 1:
            image = image[0]
        else:
            raise Exception('multiple images?')

    if image.shape[0] in [3, 4]:  # move channel_first to channel_last
        image = einops.rearrange(image, 'c h w -> h w c')

    if image.shape[-1] == 4:
        image = image[:, :, :3]

    return image.copy()


@parse_tensor
def parse_boxes(boxes, image_shape):
    h, w, c = image_shape

    if len(boxes.shape) == 3:
        if boxes.shape[0] == 1:
            boxes = boxes[0]
        else:
            raise Exception('batch_size is more than 1')
    
    if len(boxes.shape) == 1:
        boxes = boxes[None]

    if np.logical_and(boxes >= 0, boxes <= 1).all():
        boxes[:, 0] *= w
        boxes[:, 1] *= h
        boxes[:, 2] *= w
        boxes[:, 3] *= h

    boxes[:, :4] = np.round(boxes[:, :4])

    return boxes


@parse_tensor
def parse_masks(masks, image_shape, mask_threshold):
    h, w, c = image_shape

    if len(masks.shape) == 3:
        pass
    elif len(masks.shape) == 2:
        masks = masks[None]
    else:
        raise Exception('format')
    if np.unique(masks).shape[0] > 2:
        masks = masks > mask_threshold

    return masks


def vis_box(image, box, color, thickness):
    xmin, ymin, xmax, ymax = box
    image = cv2.rectangle(image, (int(xmin), int(ymin)),
                          (int(xmax), int(ymax)), color, thickness)
    return image


# @parse_image
def _vis_mask(image, mask, is_countur, is_bbox, alpha, color, thickness):
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c],
                                  image[:, :, c])

    if is_countur or is_bbox:
        contours = find_contours(mask.T, 0.5)
        contours = [np.array(i).round().astype(np.int32) for i in contours]

    if is_countur:
        image = cv2.polylines(image, contours, True, color, thickness=1)

    if is_bbox:
        contours = np.concatenate(contours)
        xmin, xmax = np.min(contours[:, 0]), np.max(contours[:, 0])
        ymin, ymax = np.min(contours[:, 1]), np.max(contours[:, 1])
        image = vis_box(image, box=(xmin, ymin, xmax, ymax),
                        color=color, thickness=thickness)

    return image


def draw_on_image(image, masks=None, boxes=None, labels=None,
                  color=None, mask_threshold=0.5,
                  boxes_format='xyxy', thickness=1,
                  is_countur=False, is_mask_box=False,
                  ):
    image = parse_image(image)

    if masks is None and boxes is None:
        raise Exception('no data to display')

    if boxes is not None:
        boxes = parse_boxes(boxes, image.shape)
        n = boxes.shape[0]

    if masks is not None:
        masks = parse_masks(masks, image.shape, mask_threshold)
        n = masks.shape[0]

    for i in range(n):
        if boxes is not None:
            image = vis_box(image, boxes[i],
                            COLOR_PALETTE[i % len(COLOR_PALETTE)],
                            thickness=thickness)

        if masks is not None:
            image = _vis_mask(image, masks[i], is_countur=is_countur, is_bbox=is_mask_box,
                              alpha=0.5, color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
                              thickness=thickness)

    return image
