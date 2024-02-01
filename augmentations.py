import cv2
import torch
import numpy as np
from math import sqrt
import torch.nn.functional as F

# from config import config.cfg, config.MEANS, config.STD
import config


class Resize(object):
    """ If preserve_aspect_ratio is true, this resizes to an approximate area of max_size * max_size """

    @staticmethod
    def calc_size_preserve_ar(img_w, img_h, max_size):
        """ I mathed this one out on the piece of paper. Resulting width*height = approx max_size^2 """
        ratio = sqrt(img_w / img_h)
        w = max_size * ratio
        h = max_size / ratio
        return int(w), int(h)

    def __init__(self, resize_gt=True):
        self.resize_gt = resize_gt
        self.max_size = config.cfg.max_size
        self.preserve_aspect_ratio = config.cfg.preserve_aspect_ratio

    def __call__(self, image, masks, boxes, labels=None):
        img_h, img_w, _ = image.shape
        
        if self.preserve_aspect_ratio:
            width, height = Resize.calc_size_preserve_ar(img_w, img_h, self.max_size)
        else:
            width, height = self.max_size, self.max_size

        image = cv2.resize(image, (width, height))

        if self.resize_gt:
            # Act like each object is a color channel
            masks = masks.transpose((1, 2, 0))
            masks = cv2.resize(masks, (width, height))
            
            # OpenCV resizes a (w,h,1) array to (s,s), so fix that
            if len(masks.shape) == 2:
                masks = np.expand_dims(masks, 0)
            else:
                masks = masks.transpose((2, 0, 1))

            # Scale bounding boxes (which are currently absolute coordinates)
            boxes[:, [0, 2]] *= (width  / img_w)
            boxes[:, [1, 3]] *= (height / img_h)

        # Discard boxes that are smaller than we'd like
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]

        keep = (w > config.cfg.discard_box_width) * (h > config.cfg.discard_box_height)
        masks = masks[keep]
        boxes = boxes[keep]
        labels['labels'] = labels['labels'][keep]
        labels['num_crowds'] = (labels['labels'] < 0).sum()

        return image, masks, boxes, labels



class FastBaseTransform(torch.nn.Module):
    """
    Transform that does all operations on the GPU for super speed.
    This doesn't suppport a lot of config settings and should only be used for production.
    Maintain this as necessary.
    """

    def __init__(self):
        super().__init__()
        
        self.mean = torch.Tensor(config.MEANS).float().cuda()[None, :, None, None]
        self.std  = torch.Tensor( config.STD ).float().cuda()[None, :, None, None]  # BGR
        self.transform = config.cfg.backbone.transform

    def forward(self, img):
        self.mean = self.mean.to(img.device)
        self.std  = self.std.to(img.device)
        
        # img assumed to be a pytorch BGR image with channel order [n, h, w, c]
        if config.cfg.preserve_aspect_ratio:
            _, h, w, _ = img.size()
            img_size = Resize.calc_size_preserve_ar(w, h, config.cfg.max_size)
            img_size = (img_size[1], img_size[0]) # Pytorch needs h, w
        else:
            img_size = (config.cfg.max_size, config.cfg.max_size)

        img = img.permute(0, 3, 1, 2).contiguous()      # [n, c, h, w]
        img = F.interpolate(img, img_size, mode='bilinear', align_corners=False)

        if self.transform.normalize:
            img = (img - self.mean) / self.std
        elif self.transform.subtract_config.MEANS:
            img = (img - self.mean)
        elif self.transform.to_float:
            img = img / 255
        
        if self.transform.channel_order != 'RGB':
            raise NotImplementedError
        
        img = img[:, (2, 1, 0), :, :].contiguous()   # BGR -> RGB

        # Return value is in channel order [n, c, h, w] and RGB
        return img