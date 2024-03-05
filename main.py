import os
import cv2
import torch
import torch.utils.data as data
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
from PIL import Image
from collections import defaultdict
from skimage import measure

# from config import config.cfg, Config.Args, config.set_config.cfg, COLORS, Config
from . import config
# from yolact import Yolact
from . import yolact
# from multiboxloss.multiboxloss import Multiboxloss.MultiBoxLoss
from . import multiboxloss
# from augmentations import Augmentations.FastBaseTransform
from . import augmentations
# from utils import utils.undo_image_transformation, utils.postprocess
from . import utils
from . import timer



iou_thresholds = [x / 100 for x in range(50, 100, 5)]
coco_cats = {} # Call prep_coco_cats to fill this
coco_cats_inv = {}
color_cache = defaultdict(lambda: {})

def prep_display(dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, fps_str=''):
    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    """
    if undo_transform:
        img_numpy = utils.undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy).cuda()
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape
    
    with timer.env('Utils.postprocess'):
        save = config.cfg.rescore_bbox
        config.cfg.rescore_bbox = True
        t = utils.postprocess(dets_out, w, h, visualize_lincomb = config.args.display_lincomb,
                                        crop_masks        = config.args.crop,
                                        score_threshold   = config.args.score_threshold)
        config.cfg.rescore_bbox = save

    with timer.env('Copy'):
        idx = t[1].argsort(0, descending=True)[:config.args.top_k]
        
        if config.cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            masks = t[3][idx]
        classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]

    num_dets_to_consider = min(config.args.top_k, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < config.args.score_threshold:
            num_dets_to_consider = j
            break

    # Quick and dirty lambda for selecting the color for a particular index
    # Also keeps track of a per-gpu color cache for maximum speed
    def get_color(j, on_gpu=None):
        global color_cache
        color_idx = (classes[j] * 5 if class_color else j * 5) % len(config.COLORS)
        
        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            return color_cache[on_gpu][color_idx]
        else:
            color = config.COLORS[color_idx]
            if not undo_transform:
                # The image might come in as RGB or BRG, depending
                color = (color[2], color[1], color[0])
            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.
                color_cache[on_gpu][color_idx] = color
            return color

    # First, draw the masks on the GPU where we can do it really fast
    # Beware: very fast but possibly unintelligible mask-drawing code ahead
    # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
    if config.args.display_masks and config.cfg.eval_mask_branch and num_dets_to_consider > 0:
        # After this, mask is of size [num_dets, h, w, 1]
        masks = masks[:num_dets_to_consider, :, :, None]
        
        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
        colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
        masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

        # This is 1 everywhere except for 1-mask_alpha where the mask is
        inv_alph_masks = masks * (-mask_alpha) + 1
        
        # I did the math for this on pen and paper. This whole block should be equivalent to:
        #    for j in range(num_dets_to_consider):
        #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
        masks_color_summand = masks_color[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)

        img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand
    
    if config.args.display_fps:
            # Draw the box for the fps on the GPU
        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 1

        text_w, text_h = cv2.getTextSize(fps_str, font_face, font_scale, font_thickness)[0]

        img_gpu[0:text_h+8, 0:text_w+8] *= 0.6 # 1 - Box alpha


    # Then draw the stuff that needs to be done on the cpu
    # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
    img_numpy = (img_gpu * 255).byte().cpu().numpy()

    if config.args.display_fps:
        # Draw the text on the CPU
        text_pt = (4, text_h + 2)
        text_color = [255, 255, 255]

        cv2.putText(img_numpy, fps_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    if num_dets_to_consider == 0:
        return img_numpy

    if config.args.display_text or config.args.display_bboxes:
        for j in reversed(range(num_dets_to_consider)):
            x1, y1, x2, y2 = boxes[j, :]
            color = get_color(j)
            score = scores[j]

            if config.args.display_bboxes:
                cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

            if config.args.display_text:
                _class = config.cfg.dataset.class_names[classes[j]]
                text_str = '%s: %.2f' % (_class, score) if config.args.display_scores else _class

                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.6
                font_thickness = 1

                text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                text_pt = (x1, y1 - 3)
                text_color = [255, 255, 255]

                cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    return img_numpy




# def validation_image_generation_wrapper(net:yolact.Yolact, args:config.Args, cfg:config, input_imgs:dict, output_imgs_folder:str, output_imgs_filenames:dict):
#     if args.config is not None:
#         config.set_cfg(args.config)

#     if args.detect:
#         cfg.eval_mask_branch = False

#     with torch.no_grad():
#         if not os.path.exists('results'):
#             os.makedirs('results')

#         if args.cuda:
#             cudnn.fastest = True
#             torch.set_default_tensor_type('torch.cuda.FloatTensor')
#         else:
#             torch.set_default_tensor_type('torch.FloatTensor')

#         net.detect.use_fast_nms = args.fast_nms
#         net.detect.use_cross_class_nms = args.cross_class_nms
#         cfg.mask_proto_debug = args.mask_proto_debug

#         if not os.path.exists(output_imgs_folder):
#             os.makedirs(output_imgs_folder)
            
#         for idx in list(input_imgs.keys()):
#             current_img = input_imgs[idx]  # RGB [0-255], 1HWC
            
#             # permute channels from RGB to BGR for code matching
#             current_img_bgr = current_img[..., ::-1]
#             # .copy() to prevent numpy tensor memory negative stride problem
#             frame = torch.from_numpy(current_img_bgr.squeeze(0).copy()).cuda().float()     # BGR, [H,W,C], [0-255]
#             batch = augmentations.FastBaseTransform()(frame.unsqueeze(0))      # BGR, [H,W,C] -> # BGR, [0,H,W,C] -> # RGB [B,C,H,W]
#             preds = net(batch, output_image=True)

#             img_numpy = prep_display(preds, frame, None, None, undo_transform=False)   # RGB
            
#             if len(output_imgs_filenames) == 0:
#                 img_numpy = img_numpy[:, :, (2, 1, 0)]  # BGR
                
#             cv2.imwrite(os.path.join(output_imgs_folder, output_imgs_filenames[idx]), img_numpy)
#             print(f"Saved  idx:{idx}  {os.path.join(output_imgs_folder, output_imgs_filenames[idx])}")



def preprocess_img_for_yolact_loss(img, in_img_type="PIL", in_img_val_range="0-255", in_shape_order="HWC", in_channel_order="RGB"):
    # convert img to torch tensor
    if in_img_type == "PIL":
        img = torch.from_numpy(np.array(img))
    elif in_img_type == "numpy":
        img = torch.from_numpy(img)
    else:
        raise NotImplementedError()
    
    # rearrange whatever input shape order to CHW
    out_shape_order = "CHW"
    shape_map = {c: idx for idx, c in enumerate(in_shape_order)}
    shape_permutation = [shape_map[c] for c in out_shape_order]
    img = img.permute(shape_permutation)    # CHW
    
    # rearrange whatever input channel ordering to RGB
    out_channel_order = "RGB"
    channel_map = {c: idx for idx, c in enumerate(in_channel_order)}
    channel_permutation = [channel_map[c] for c in out_channel_order]
    img = img[channel_permutation, :, :]    # RGB, H, W
    
    # MEANS & STD of ImageNet in RGB order
    MEANS = (123.68, 116.78, 103.94)
    STD   = (58.40, 57.12, 57.38)
    mean = torch.Tensor(MEANS).float()[:, None, None]  # RGB,H,W / RGB,W,H
    std  = torch.Tensor( STD ).float()[:, None, None]
    
    if in_img_val_range == "0-1":
        img = img * 255
    elif in_img_val_range == "0-255":
        pass
    elif in_img_val_range == "-1+1":
        img = ((img + 1) / 2) * 255
    else:
        raise NotImplementedError()
    
    img = (img - mean) / std
    
    # output_format: normalised by -MEANS / STD not in [0-255],  RGB,H,W
    return img



def preprocess_segmap_for_yolact_loss(segmap, in_shape_order="HWC", class_label_value=3):
    """
    segmap : [0-1], CHW, numpy.array
    
    bbox/targets,    : [[tlx, tly, brx, bry, lbl-1], [..], [...]] in relative units
    masks,    : [num_obj, h, w], [0-1], masks ordering is according to bbox ordering. Hence in our case: num_obj=1, since just one car
    num_crowds    : int (can set to 0)   -> say batch size 8, [0,0,0,0,0,0,0,0]
    """
    
    segmap = torch.tensor(segmap)
    
    # rearrange whatever input shape order to CHW
    out_shape_order = "CHW"
    shape_map = {c: idx for idx, c in enumerate(in_shape_order)}
    shape_permutation = [shape_map[c] for c in out_shape_order]
    segmap = segmap.permute(shape_permutation)    # CHW
    
    segmap = segmap[0, :, :]   # random pick one layer since all same  H,W

    # bbox
    # =================================================
    labeled_image, num_labels = measure.label(segmap.cpu().numpy(), connectivity=2, return_num=True)
    # find largest connected object: main car
    largest_component_label = np.argmax(np.bincount(labeled_image.flat)[1:]) + 1
    # new binary image with largest object only
    largest_component_image = labeled_image == largest_component_label
    # find coordinates: topleft, bottomright
    coords = np.column_stack(np.where(largest_component_image))
    top_left = np.min(coords, axis=0)
    bottom_right = np.max(coords, axis=0)
    # normalise coordinates according to image h,w
    height, width = segmap.shape   # 600, 800
    normalized_top_left = (top_left[1] / width, top_left[0] / height)    # invert for x,y

    normalized_bottom_right = (bottom_right[1] / width, bottom_right[0] / height)  # invert for x,y
    
    #        top_left_x              top_left_y              bottom_left_x               bottom_left_y               label_idx: car (Yolact's code does label-1)
    bbox = [[normalized_top_left[0], normalized_top_left[1], normalized_bottom_right[0], normalized_bottom_right[1], class_label_value-1]]
    bbox = torch.Tensor(bbox)
    #       expand 2D mask to 3D: [num_obj, H, W], where num_object=1, since we have only 1 object: main car
    masks = np.expand_dims(segmap.cpu().numpy(), axis=0).astype(np.float32)
    #       no crowds in our use case
    num_crowds = 0
    
    return bbox, masks, num_crowds




def yolact_training_wrapper(net, input_imgs, gt_segmaps):
    """
    input_imgs  -  RGB [0-1], 3CHW
    gt_segmaps  -  RGB [0-1], 3CHW
    """
    
    (yolact_net, criterion, args, cfg) = net
    
    yolact_net.eval()
    yolact_net.freeze_bn()
    for params in yolact_net.parameters():
        params.requires_grad = False
        
    input_imgs = Variable(input_imgs).to(yolact_net.parameters().__next__().device)
    input_imgs = input_imgs * 255
    
    # MEANS & STD of ImageNet in RGB order
    MEANS = (123.68, 116.78, 103.94)
    STD   = (58.40, 57.12, 57.38)
    mean = torch.Tensor(MEANS).float()[None, :, None, None].to(yolact_net.parameters().__next__().device)  # RGB,H,W / RGB,W,H
    std  = torch.Tensor( STD ).float()[None, :, None, None].to(yolact_net.parameters().__next__().device)
    
    input_imgs = (input_imgs - mean) / std
    
    bbox_list = []
    masks_list = []
    num_crowds_list = []
    
    for idx in range(gt_segmaps.shape[0]):
        gt_segmap = gt_segmaps[idx]
        assert len(gt_segmap.shape) == 3, "GT Segmap shape not in CHW"
        bbox, masks, num_crowds = preprocess_segmap_for_yolact_loss(segmap=gt_segmap,
                                                                    in_shape_order="CHW",
                                                                    class_label_value=3  # car
                                                                    )
        bbox_list.append(torch.Tensor(bbox))
        masks_list.append(torch.Tensor(masks))
        num_crowds_list.append(num_crowds)

    
    bbox_tensor = Variable(torch.stack(bbox_list)).to(yolact_net.parameters().__next__().device)
    masks_tensor = Variable(torch.stack(masks_list)).to(yolact_net.parameters().__next__().device)
    num_crowds_tensor = Variable(torch.tensor(num_crowds_list)).to(yolact_net.parameters().__next__().device)
    
    # feed forward
    # images, (targets, masks, num_crowds) = batch
    preds = yolact_net(input_imgs)
    
    # compute loss
    losses = criterion(yolact_net, preds, bbox_tensor, masks_tensor, num_crowds_tensor)
    loss_dict = { k: (v).mean() for k,v in losses.items() } # Mean here because Dataparallel
    
    return loss_dict






def yolact_validation_wrapper(net, cam_idxs, input_imgs, gt_imgs, gt_segmaps):
    
    """
    input_img  -  RGB [0-255], BHWC, numpy.array
    gt_imgs    -  RGB [0-255], HWC,  PIL.Image
    gt_segmaps -      [0-1],   BCHW, numpy.array
    """
    
    (yolact_net, criterion, args, cfg) = net
    
    yolact_net.eval()
    yolact_net.freeze_bn()
    for params in yolact_net.parameters():
        params.requires_grad = False
    
    # here, instead of doing inference twice: on input_imgs & gt_imgs, since we have the gt_segmap,
    # we will just inference once on the input_imgs & then later calclate loss between the predicted
    # items and the (bbox, mask, num_crowds) generated from the gt_segmap
    
    lookup_cam_idx_to_det_image = {}
    
    input_imgs_list = []
    bbox_list = []
    masks_list = []
    num_crowds_list = []
    
    for idx in cam_idxs:
        # preprocess input image
        input_img = input_imgs[idx]
        assert len(input_img.shape) == 4, "Input Image shape not in BHWC"
        processed_img = preprocess_img_for_yolact_loss(img=input_img.squeeze(),
                                                       in_img_type="numpy",
                                                       in_img_val_range="0-255",
                                                       in_shape_order="HWC",
                                                       in_channel_order="RGB")
        input_imgs_list.append(processed_img)
        
        # preprocess input segmaps
        gt_segmap = gt_segmaps[idx]
        assert len(gt_segmap.shape) == 4, "GT Segmap shape not in BCHW"
        bbox, masks, num_crowds = preprocess_segmap_for_yolact_loss(segmap=gt_segmap.squeeze(),
                                                                    in_shape_order="CHW",
                                                                    class_label_value=3  # car
                                                                    )
        bbox_list.append(torch.Tensor(bbox))
        masks_list.append(torch.Tensor(masks))
        num_crowds_list.append(num_crowds)
    
    input_imgs_tensor = torch.stack(input_imgs_list).to(yolact_net.parameters().__next__().device)
    bbox_tensor = torch.stack(bbox_list).to(yolact_net.parameters().__next__().device)
    masks_tensor = torch.stack(masks_list).to(yolact_net.parameters().__next__().device)
    num_crowds_tensor = torch.tensor(num_crowds_list).to(yolact_net.parameters().__next__().device)
    
    # feed forward
    # images, (targets, masks, num_crowds) = batch
    preds = yolact_net(input_imgs_tensor)
    
    # compute loss
    losses = criterion(yolact_net, preds, bbox_tensor, masks_tensor, num_crowds_tensor)
    loss_dict = { k: (v).mean() for k,v in losses.items() } # Mean here because Dataparallel
    
    # inference images
    if args.config is not None:
        config.set_cfg(args.config)

    if args.detect:
        cfg.eval_mask_branch = False

    with torch.no_grad():
        if args.cuda:
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        yolact_net.detect.use_fast_nms = args.fast_nms
        yolact_net.detect.use_cross_class_nms = args.cross_class_nms
        cfg.mask_proto_debug = args.mask_proto_debug
            
        for idx in cam_idxs:
            current_img = input_imgs[idx]  # RGB [0-255], 1HWC
            
            # permute channels from RGB to BGR for code matching
            current_img_bgr = current_img[..., ::-1]
            
            # .copy() to prevent numpy tensor memory negative stride problem
            frame = torch.from_numpy(current_img_bgr.squeeze(0).copy()).to(yolact_net.parameters().__next__().device).float()     # BGR, [H,W,C], [0-255]
            batch = augmentations.FastBaseTransform()(frame.unsqueeze(0))      # BGR, [H,W,C] -> # BGR, [0,H,W,C] -> # RGB [B,C,H,W]
            preds = yolact_net(batch, output_image=True)

            img_numpy = prep_display(preds, frame, None, None, undo_transform=False)   # RGB
            
            lookup_cam_idx_to_det_image[idx] = Image.fromarray(img_numpy)     # Image.fromarray expects to be in HWC, which is already in
            
    return loss_dict, lookup_cam_idx_to_det_image



def init_yolact(model_weights_path, model_config="yolact_base_config", device=None):
    from . import config
    
    global args
    args = config.Args()
    
    args.trained_model = model_weights_path  # need to update this as well, for validation's inferencing to work
    args.config = model_config  # this as well
    config.args = args  # this too
    
    if args.config is not None:
        config.set_cfg(args.config)
        
    net = yolact.Yolact()
    net.load_weights(args.trained_model)
    net.eval()
    net.freeze_bn()
    for params in net.parameters():
        params.requires_grad = False
    
    criterion = multiboxloss.MultiBoxLoss(num_classes=config.cfg.num_classes,
                             pos_threshold=config.cfg.positive_iou_threshold,
                             neg_threshold=config.cfg.negative_iou_threshold,
                             negpos_ratio=config.cfg.ohem_negpos_ratio)
    
    if device != None:
        net = net.to(device)
        criterion = criterion.to(device)
    elif args.cuda:
        net = net.cuda()
        criterion = criterion.cuda()
    
    return (net, criterion, args, config.cfg)









def main():
    ## NOTE for Alex (usage):
    """
    1. yolact_model = init_yolact(...)
    2. loss_dict             = yolact_training_wrapper(yolact_model, ...)
    3. loss_dict, PIL_Images = yolact_validation_wrapper(yolact_model, ...)
    """
    
    
    ########## TRAINING LOSS for Diffusion model
    ####################################################################################################
    ####################################################################################################
    
    yolact_model = init_yolact(model_weights_path="/work/u5832291/yixian/YOLACT_edit/weights/yolact_base_54_800000.pth",
                               model_config="yolact_base_config",
                               device=torch.device("cuda:0"))    # To Alex: specify the device here i.e. torch.device(..), everything below will be moved to this device too
    
    input_size = (416, 416)

    img_folder = "/work/u5832291/view_neti_RT/data/DistentangledCarlaScenes/images/blackcar_day_both/"
    segmap_folder = "/work/u5832291/view_neti_RT/data/DistentangledCarlaScenes/masks/blackcar_day_both/"

    file_names = os.listdir(img_folder)
    file_names = sorted(file_names)
    print(file_names)
    
    image0 = Image.open(img_folder + file_names[0]).convert("RGB")    # RGB [0-255], HWC
    image1 = Image.open(img_folder + file_names[1]).convert("RGB")
    image2 = Image.open(img_folder + file_names[2]).convert("RGB")
    segmap0 = Image.open(segmap_folder + file_names[0]).convert("RGB")
    segmap1 = Image.open(segmap_folder + file_names[1]).convert("RGB")
    segmap2 = Image.open(segmap_folder + file_names[2]).convert("RGB")

    image0 = image0.resize((input_size[0], input_size[1]))
    image1 = image1.resize((input_size[0], input_size[1]))
    image2 = image2.resize((input_size[0], input_size[1]))
    segmap0 = segmap0.resize((input_size[0], input_size[1]))
    segmap1 = segmap1.resize((input_size[0], input_size[1]))
    segmap2 = segmap2.resize((input_size[0], input_size[1]))

    im_data0 = torch.from_numpy(np.array(image0)).float() / 255       # RGB [0-1], HWC
    im_data1 = torch.from_numpy(np.array(image1)).float() / 255
    im_data2 = torch.from_numpy(np.array(image2)).float() / 255
    target_pixel = np.array([0, 0, 142])
    segmap0 = torch.from_numpy(np.stack((np.all(np.array(segmap0) == target_pixel, axis=-1),)*3, axis=-1))
    segmap1 = torch.from_numpy(np.stack((np.all(np.array(segmap1) == target_pixel, axis=-1),)*3, axis=-1))
    segmap2 = torch.from_numpy(np.stack((np.all(np.array(segmap2) == target_pixel, axis=-1),)*3, axis=-1))

    #print('---im_data0.shape={}'.format(im_data0.shape))
    im_data0 = im_data0.permute(2, 0, 1).unsqueeze(0)                 # RGB [0-1], 1CHW
    im_data1 = im_data1.permute(2, 0, 1).unsqueeze(0)
    im_data2 = im_data2.permute(2, 0, 1).unsqueeze(0)
    segmap0 = segmap0.permute(2, 0, 1).unsqueeze(0)
    segmap1 = segmap1.permute(2, 0, 1).unsqueeze(0)
    segmap2 = segmap2.permute(2, 0, 1).unsqueeze(0)

    concatenated_im_data = torch.cat([im_data0, im_data1, im_data2], dim=0)      # RGB [0-1], 3CHW
    concatenated_gt_segmap_data = torch.cat([segmap0, segmap1, segmap2], dim=0)  # RGB [0-1], 3CHW
    
    concatenated_im_data = concatenated_im_data.to(yolact_model[0].parameters().__next__().device)
    concatenated_gt_segmap_data = concatenated_gt_segmap_data.to(yolact_model[0].parameters().__next__().device)
    
    # concatenated_gt_segmap_data should be the same as batch["binary_image_mask"] in coach.py
    
    loss_dict = yolact_training_wrapper(net=yolact_model,
                                        input_imgs=concatenated_im_data,
                                        gt_segmaps=concatenated_gt_segmap_data)
    
    print("===============\nTraining Phase\n")
    print(f"Box Localization Loss:      {loss_dict['B']}")    # Box Localization Loss
    print(f"Class Confidence Loss:      {loss_dict['C']}")    # Class Confidence Loss
    print(f"Mask Loss:                  {loss_dict['M']}")    # Mask Loss
    print(f"Semantic Segmentation Loss: {loss_dict['S']}")    # Semantic Segmentation Loss







    ########## validation
    ####################################################################################################
    ####################################################################################################
    
    
    # ~!to_change
    img_folder = "/work/u5832291/view_neti_RT/data/DistentangledCarlaScenes/images/blackcar_day_both/"
    segmap_folder = "/work/u5832291/view_neti_RT/data/DistentangledCarlaScenes/masks/blackcar_day_both/"

    file_names = os.listdir(img_folder)
    file_names = sorted(file_names)
    lookup_camidx_to_gt_image = {}     # GT image
    lookup_camidx_to_gt_segmap = {}    # GT Segmap & bbox
    lookup_camidx_to_img_pred = {}     # ViewNETI predicted images
    cam_idxs=[0, 1, 2]
    

    for cam_idx, file_name in zip(cam_idxs, file_names):
        img_name = os.path.join(img_folder, file_name)
        segmap_name = os.path.join(segmap_folder, file_name)
        image = Image.open(img_name).convert("RGB")                 # RGB [0-255], HWC
        segmap = np.array(Image.open(segmap_name).convert("RGB"))   # RGB [0-255], HWC
        
        # Define the target pixel value: vehicle class is [0, 0, 142]
        target_pixel = np.array([0, 0, 142])
        binary_mask = np.all(segmap == target_pixel, axis=-1)
        
        # make the binary mask to RGB shape
        # latent mask is 4 in depth
        binary_mask = np.stack((binary_mask,)*3, axis=-1)
        binary_mask = binary_mask.transpose(2,0,1)  # RGB, H, W
        
        image_np = np.array(image)
        
        lookup_camidx_to_gt_image[cam_idx] = image                              # RGB [0-255], HWC,  PIL.Image
        lookup_camidx_to_img_pred[cam_idx] = np.expand_dims(image_np, axis=0)   # RGB [0-255], BHWC, numpy.array
        lookup_camidx_to_gt_segmap[cam_idx] = np.expand_dims(binary_mask, axis=0)   # [0-1], BCHW, numpy.array


    # validation function call - loss calculation and PIL.Image dict return
    loss_dict, lookup_cam_idx_to_det_image = yolact_validation_wrapper(net=yolact_model, 
                                                                       cam_idxs=cam_idxs, 
                                                                       input_imgs=lookup_camidx_to_img_pred, 
                                                                       gt_imgs=lookup_camidx_to_gt_image, 
                                                                       gt_segmaps=lookup_camidx_to_gt_segmap)
    print("===============\nValidation Phase\n")
    print(f"Box Localization Loss:      {loss_dict['B']}")    # Box Localization Loss
    print(f"Class Confidence Loss:      {loss_dict['C']}")    # Class Confidence Loss
    print(f"Mask Loss:                  {loss_dict['M']}")    # Mask Loss
    print(f"Semantic Segmentation Loss: {loss_dict['S']}")    # Semantic Segmentation Loss
    
    
    # Test saving the PIL Images
    output_directory = "/work/u5832291/yixian/YOLACT/test_tun_outputs"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for idx in list(lookup_cam_idx_to_det_image.keys()):
        file_path = os.path.join(output_directory, file_names[idx])
        img = lookup_cam_idx_to_det_image[idx]
        img.save(file_path, format="PNG")
        print(f" >>> Image {idx} saved to: {file_path}")
        
    print("Done.")
        

if __name__ == '__main__':
    main()

