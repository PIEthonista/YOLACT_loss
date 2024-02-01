import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import math
from collections import deque
from pathlib import Path

# from layers import InterpolateModule, Concat
import layers
# from config import cfg, mask_type, activation_func, MEANS, STD
import config
# from box_utils import sanitize_coordinates, crop
import box_utils
import timer



class ProgressBar():
    """ A simple progress bar that just outputs a string. """

    def __init__(self, length, max_val):
        self.max_val = max_val
        self.length = length
        self.cur_val = 0
        
        self.cur_num_bars = -1
        self._update_str()

    def set_val(self, new_val):
        self.cur_val = new_val

        if self.cur_val > self.max_val:
            self.cur_val = self.max_val
        if self.cur_val < 0:
            self.cur_val = 0

        self._update_str()
    
    def is_finished(self):
        return self.cur_val == self.max_val

    def _update_str(self):
        num_bars = int(self.length * (self.cur_val / self.max_val))

        if num_bars != self.cur_num_bars:
            self.cur_num_bars = num_bars
            self.string = '█' * num_bars + '░' * (self.length - num_bars)
    
    def __repr__(self):
        return self.string
    
    def __str__(self):
        return self.string


def init_console():
    """
    Initialize the console to be able to use ANSI escape characters on Windows.
    """
    if os.name == 'nt':
        from colorama import init
        init()


class SavePath:
    """
    Why is this a class?
    Why do I have a class for creating and parsing save paths?
    What am I doing with my life?
    """

    def __init__(self, model_name:str, epoch:int, iteration:int):
        self.model_name = model_name
        self.epoch = epoch
        self.iteration = iteration

    def get_path(self, root:str=''):
        file_name = self.model_name + '_' + str(self.epoch) + '_' + str(self.iteration) + '.pth'
        return os.path.join(root, file_name)

    @staticmethod
    def from_str(path:str):
        file_name = os.path.basename(path)
        
        if file_name.endswith('.pth'):
            file_name = file_name[:-4]
        
        params = file_name.split('_')

        if file_name.endswith('interrupt'):
            params = params[:-1]
        
        model_name = '_'.join(params[:-2])
        epoch = params[-2]
        iteration = params[-1]
        
        return SavePath(model_name, int(epoch), int(iteration))

    @staticmethod
    def remove_interrupt(save_folder):
        for p in Path(save_folder).glob('*_interrupt.pth'):
            p.unlink()
    
    @staticmethod
    def get_interrupt(save_folder):
        for p in Path(save_folder).glob('*_interrupt.pth'): 
            return str(p)
        return None
    
    @staticmethod
    def get_latest(save_folder, config):
        """ Note: config should be config.name. """
        max_iter = -1
        max_name = None

        for p in Path(save_folder).glob(config + '_*'):
            path_name = str(p)

            try:
                save = SavePath.from_str(path_name)
            except:
                continue 
            
            if save.model_name == config and save.iteration > max_iter:
                max_iter = save.iteration
                max_name = path_name

        return max_name

def make_net(in_channels, conf, include_last_relu=True):
    """
    A helper function to take a config setting and turn it into a network.
    Used by protonet and extrahead. Returns (network, out_channels)
    """
    def make_layer(layer_cfg):
        nonlocal in_channels
        
        # Possible patterns:
        # ( 256, 3, {}) -> conv
        # ( 256,-2, {}) -> deconv
        # (None,-2, {}) -> bilinear interpolate
        # ('cat',[],{}) -> concat the subnetworks in the list
        #
        # You know it would have probably been simpler just to adopt a 'c' 'd' 'u' naming scheme.
        # Whatever, it's too late now.
        if isinstance(layer_cfg[0], str):
            layer_name = layer_cfg[0]

            if layer_name == 'cat':
                nets = [make_net(in_channels, x) for x in layer_cfg[1]]
                layer = layers.Concat([net[0] for net in nets], layer_cfg[2])
                num_channels = sum([net[1] for net in nets])
        else:
            num_channels = layer_cfg[0]
            kernel_size = layer_cfg[1]

            if kernel_size > 0:
                layer = nn.Conv2d(in_channels, num_channels, kernel_size, **layer_cfg[2])
            else:
                if num_channels is None:
                    layer = layers.InterpolateModule(scale_factor=-kernel_size, mode='bilinear', align_corners=False, **layer_cfg[2])
                else:
                    layer = nn.ConvTranspose2d(in_channels, num_channels, -kernel_size, **layer_cfg[2])
        
        in_channels = num_channels if num_channels is not None else in_channels

        # Don't return a ReLU layer if we're doing an upsample. This probably doesn't affect anything
        # output-wise, but there's no need to go through a ReLU here.
        # Commented out for backwards compatibility with previous models
        # if num_channels is None:
        #     return [layer]
        # else:
        return [layer, nn.ReLU(inplace=True)]

    # Use sum to concat together all the component layer lists
    net = sum([make_layer(x) for x in conf], [])
    if not include_last_relu:
        net = net[:-1]

    return nn.Sequential(*(net)), in_channels



class MovingAverage():
    """ Keeps an average window of the specified number of items. """

    def __init__(self, max_window_size=1000):
        self.max_window_size = max_window_size
        self.reset()

    def add(self, elem):
        """ Adds an element to the window, removing the earliest element if necessary. """
        if not math.isfinite(elem):
            print('Warning: Moving average ignored a value of %f' % elem)
            return
        
        self.window.append(elem)
        self.sum += elem

        if len(self.window) > self.max_window_size:
            self.sum -= self.window.popleft()
    
    def append(self, elem):
        """ Same as add just more pythonic. """
        self.add(elem)

    def reset(self):
        """ Resets the MovingAverage to its initial state. """
        self.window = deque()
        self.sum = 0

    def get_avg(self):
        """ Returns the average of the elements in the window. """
        return self.sum / max(len(self.window), 1)

    def __str__(self):
        return str(self.get_avg())
    
    def __repr__(self):
        return repr(self.get_avg())
    
    def __len__(self):
        return len(self.window)





def undo_image_transformation(img, w, h):
    """
    Takes a transformed image tensor and returns a numpy ndarray that is untransformed.
    Arguments w and h are the original height and width of the image.
    """
    img_numpy = img.permute(1, 2, 0).cpu().numpy()
    img_numpy = img_numpy[:, :, (2, 1, 0)] # To BRG

    if config.cfg.backbone.transform.normalize:
        img_numpy = (img_numpy * np.array(config.STD) + np.array(config.MEANS)) / 255.0
    elif config.cfg.backbone.transform.subtract_means:
        img_numpy = (img_numpy / 255.0 + np.array(config.MEANS) / 255.0).astype(np.float32)
        
    img_numpy = img_numpy[:, :, (2, 1, 0)] # To RGB
    img_numpy = np.clip(img_numpy, 0, 1)

    return cv2.resize(img_numpy, (w,h))



def postprocess(det_output, w, h, batch_idx=0, interpolation_mode='bilinear',
                visualize_lincomb=False, crop_masks=True, score_threshold=0):
    """
    Postprocesses the output of Yolact on testing mode into a format that makes sense,
    accounting for all the possible configuration settings.

    Args:
        - det_output: The lost of dicts that Detect outputs.
        - w: The real with of the image.
        - h: The real height of the image.
        - batch_idx: If you have multiple images for this batch, the image's index in the batch.
        - interpolation_mode: Can be 'nearest' | 'area' | 'bilinear' (see torch.nn.functional.interpolate)

    Returns 4 torch Tensors (in the following order):
        - classes [num_det]: The class idx for each detection.
        - scores  [num_det]: The confidence score for each detection.
        - boxes   [num_det, 4]: The bounding box for each detection in absolute point form.
        - masks   [num_det, h, w]: Full image masks for each detection.
    """
    
    # if isinstance(det_output, dict):
    #     dets = det_output
    # else:
    #     dets = det_output[batch_idx]
    dets = det_output[batch_idx]
    net = dets['net']
    dets = dets['detection']

    if dets is None:
        return [torch.Tensor()] * 4 # Warning, this is 4 copies of the same thing

    if score_threshold > 0:
        keep = dets['score'] > score_threshold

        for k in dets:
            if k != 'proto':
                dets[k] = dets[k][keep]
        
        if dets['score'].size(0) == 0:
            return [torch.Tensor()] * 4
    
    # Actually extract everything from dets now
    classes = dets['class']
    boxes   = dets['box']
    scores  = dets['score']
    masks   = dets['mask']

    if config.cfg.mask_type == config.mask_type.lincomb and config.cfg.eval_mask_branch:
        # At this points masks is only the coefficients
        proto_data = dets['proto']
        
        # Test flag, do not upvote
        if config.cfg.mask_proto_debug:
            np.save('scripts/proto.npy', proto_data.cpu().numpy())
        
        if visualize_lincomb:
            display_lincomb(proto_data, masks)

        masks = proto_data @ masks.t()
        masks = config.cfg.mask_proto_mask_activation(masks)

        # Crop masks before upsampling because you know why
        if crop_masks:
            masks = box_utils.crop(masks, boxes)

        # Permute into the correct output shape [num_dets, proto_h, proto_w]
        masks = masks.permute(2, 0, 1).contiguous()

        if config.cfg.use_maskiou:
            with timer.env('maskiou_net'):                
                with torch.no_grad():
                    maskiou_p = net.maskiou_net(masks.unsqueeze(1))
                    maskiou_p = torch.gather(maskiou_p, dim=1, index=classes.unsqueeze(1)).squeeze(1)
                    if config.cfg.rescore_mask:
                        if config.cfg.rescore_bbox:
                            scores = scores * maskiou_p
                        else:
                            scores = [scores, scores * maskiou_p]

        # Scale masks up to the full image
        masks = F.interpolate(masks.unsqueeze(0), (h, w), mode=interpolation_mode, align_corners=False).squeeze(0)

        # Binarize the masks
        masks.gt_(0.5)

    
    boxes[:, 0], boxes[:, 2] = box_utils.sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, cast=False)
    boxes[:, 1], boxes[:, 3] = box_utils.sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, cast=False)
    boxes = boxes.long()

    if config.cfg.mask_type == config.mask_type.direct and config.cfg.eval_mask_branch:
        # Upscale masks
        full_masks = torch.zeros(masks.size(0), h, w)

        for jdx in range(masks.size(0)):
            x1, y1, x2, y2 = boxes[jdx, :]

            mask_w = x2 - x1
            mask_h = y2 - y1

            # Just in case
            if mask_w * mask_h <= 0 or mask_w < 0:
                continue
            
            mask = masks[jdx, :].view(1, 1, config.cfg.mask_size, config.cfg.mask_size)
            mask = F.interpolate(mask, (mask_h, mask_w), mode=interpolation_mode, align_corners=False)
            mask = mask.gt(0.5).float()
            full_masks[jdx, y1:y2, x1:x2] = mask
        
        masks = full_masks

    return classes, scores, boxes, masks




def display_lincomb(proto_data, masks):
    out_masks = torch.matmul(proto_data, masks.t())
    # out_masks = cfg.mask_proto_mask_activation(out_masks)

    for kdx in range(1):
        jdx = kdx + 0
        import matplotlib.pyplot as plt
        coeffs = masks[jdx, :].cpu().numpy()
        idx = np.argsort(-np.abs(coeffs))
        # plt.bar(list(range(idx.shape[0])), coeffs[idx])
        # plt.show()
        
        coeffs_sort = coeffs[idx]
        arr_h, arr_w = (4,8)
        proto_h, proto_w, _ = proto_data.size()
        arr_img = np.zeros([proto_h*arr_h, proto_w*arr_w])
        arr_run = np.zeros([proto_h*arr_h, proto_w*arr_w])
        test = torch.sum(proto_data, -1).cpu().numpy()

        for y in range(arr_h):
            for x in range(arr_w):
                i = arr_w * y + x

                if i == 0:
                    running_total = proto_data[:, :, idx[i]].cpu().numpy() * coeffs_sort[i]
                else:
                    running_total += proto_data[:, :, idx[i]].cpu().numpy() * coeffs_sort[i]

                running_total_nonlin = running_total
                if config.cfg.mask_proto_mask_activation == config.activation_func.sigmoid:
                    running_total_nonlin = (1/(1+np.exp(-running_total_nonlin)))

                arr_img[y*proto_h:(y+1)*proto_h, x*proto_w:(x+1)*proto_w] = (proto_data[:, :, idx[i]] / torch.max(proto_data[:, :, idx[i]])).cpu().numpy() * coeffs_sort[i]
                arr_run[y*proto_h:(y+1)*proto_h, x*proto_w:(x+1)*proto_w] = (running_total_nonlin > 0.5).astype(np.float)
        plt.imshow(arr_img)
        plt.show()
        # plt.imshow(arr_run)
        # plt.show()
        # plt.imshow(test)
        # plt.show()
        plt.imshow(out_masks[:, :, jdx].cpu().numpy())
        plt.show()