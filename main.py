import os
import cv2
import torch
import torch.utils.data as data
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
from PIL import Image
from collections import defaultdict

# from config import config.cfg, Config.Args, config.set_config.cfg, COLORS, Config
import config
# from yolact import Yolact
import yolact
# from multiboxloss.multiboxloss import Multiboxloss.MultiBoxLoss
import multiboxloss
# from augmentations import Augmentations.FastBaseTransform
import augmentations
# from utils import utils.undo_image_transformation, utils.postprocess
import utils
import timer



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




def validation_image_generation_wrapper(net:yolact.Yolact, args:config.Args, cfg:config, input_imgs:dict, output_imgs_folder:str, output_imgs_filenames:dict):
    if args.config is not None:
        config.set_cfg(args.config)

    if args.detect:
        cfg.eval_mask_branch = False

    with torch.no_grad():
        if not os.path.exists('results'):
            os.makedirs('results')

        if args.cuda:
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        net.detect.use_fast_nms = args.fast_nms
        net.detect.use_cross_class_nms = args.cross_class_nms
        cfg.mask_proto_debug = args.mask_proto_debug

        if not os.path.exists(output_imgs_folder):
            os.makedirs(output_imgs_folder)
            
        for idx in list(input_imgs.keys()):
            current_img = input_imgs[idx]  # RGB [0-255], 1HWC
            
            # permute channels from RGB to BGR for code matching
            current_img_bgr = current_img[..., ::-1]
            # .copy() to prevent numpy tensor memory negative stride problem
            frame = torch.from_numpy(current_img_bgr.squeeze(0).copy()).cuda().float()     # BGR, [H,W,C], [0-255]
            batch = augmentations.FastBaseTransform()(frame.unsqueeze(0))      # BGR, [H,W,C] -> # BGR, [0,H,W,C] -> # RGB [B,C,H,W]
            preds = net(batch, output_image=True)

            img_numpy = prep_display(preds, frame, None, None, undo_transform=False)   # RGB
            
            if len(output_imgs_filenames) == 0:
                img_numpy = img_numpy[:, :, (2, 1, 0)]  # BGR
                
            cv2.imwrite(os.path.join(output_imgs_folder, output_imgs_filenames[idx]), img_numpy)
            print(f"Saved  idx:{idx}  {os.path.join(output_imgs_folder, output_imgs_filenames[idx])}")





def main():

    # For Alex's note
    class simulate_YOLACT_dataset(data.Dataset):
        def __init__(self, image_path, seg_map_path):
            # dummy
            self.image_path = image_path
            self.seg_map_path = seg_map_path
        
        def __len__(self):
            # dummy
            return 1
        
        def __getitem__(self, idx):
            """
            YOLACT dataset __getitem__() return is as below:
                image,   : normalised by -MEANS / STD not in [0-255],  RGB,H,W
                (
                    bbox/targets,    : [[tlx, tly, brx, bry, lbl-1], [..], [...]] in relative units
                    masks,    : [num_obj, h, w], [0-1], masks ordering is according to bbox ordering. Hence in our case: num_obj=1, since just one car
                    num_crowds    : int (can set to 0)   -> say batch size 8, [0,0,0,0,0,0,0,0]
                )
                
            bbox output visualisation refer to Figure_1.png
            """

            # image
            # =================================================
            from PIL import Image
            image = Image.open(self.image_path)   # RGB [0-255], HWC
            image = torch.from_numpy(np.array(image))   # don't do torch.transform(PIL.Image) cuz that will reduce image to value range of [0-1]
            
            # normalise image
            # MEANS & STD of ImageNet in RGB order
            MEANS = (123.68, 116.78, 103.94)
            STD   = (58.40, 57.12, 57.38)
            mean = torch.Tensor(MEANS).float()[None, None, :]  # RGB
            std  = torch.Tensor( STD ).float()[None, None, :]
            
            image = (image - mean) / std
            image = image.permute(2,0,1)  # HWC -> CHW
            
            # segmentation map
            # =================================================
            img_mask = Image.open(self.seg_map_path)   # RGB [0-255], HWC
            semantic_map = np.array(img_mask)
            
            # Define the target pixel value: vehicle class is [0, 0, 142]
            target_pixel = np.array([0, 0, 142])
            binary_mask = np.all(semantic_map == target_pixel, axis=-1)
            
            # make the binary mask to RGB shape
            # latent mask is 4 in depth
            binary_mask = np.stack((binary_mask,)*3, axis=-1)
            binary_mask = binary_mask.transpose(2,0,1)  # RGB, H, W
            
            # Above follow's Alex code in ViewNETI's dataset.
            # However, YOLACT only needs one channel for one object instance.
            yolact_binary_mask = binary_mask[0]  # random pick one layer since all same
            
            # bbox
            # =================================================
            from skimage import measure
            labeled_image, num_labels = measure.label(yolact_binary_mask, connectivity=2, return_num=True)
            # find largest connected object: main car
            largest_component_label = np.argmax(np.bincount(labeled_image.flat)[1:]) + 1
            # new binary image with largest object only
            largest_component_image = labeled_image == largest_component_label
            # find coordinates: topleft, bottomright
            coords = np.column_stack(np.where(largest_component_image))
            top_left = np.min(coords, axis=0)
            bottom_right = np.max(coords, axis=0)
            # normalise coordinates according to image h,w
            height, width = yolact_binary_mask.shape   # 600, 800
            normalized_top_left = (top_left[1] / width, top_left[0] / height)    # invert for x,y

            normalized_bottom_right = (bottom_right[1] / width, bottom_right[0] / height)  # invert for x,y
            
            """
            YOLACT dataset __getitem__() return is as below:
                image,   : normalised by -MEANS / STD not in [0-255],  RGB,H,W
                (
                    bbox/targets,    : [[tlx, tly, brx, bry, lbl-1], [..], [...]] in relative units
                    masks,    : [num_obj, h, w], [0-1], masks ordering is according to bbox ordering. Hence in our case: num_obj=1, since just one car
                    num_crowds    : int (can set to 0)   -> say batch size 8, [0,0,0,0,0,0,0,0]
                )
                
            bbox output visualisation refer to Figure_1.png
            """
            
            # COCO class label mappings (vehicle related)
            """
            'person',     1: 1
            'bicycle',    2: 2
            'car',        3: 3
            'motorcycle', 4: 4
            'airplane',   5: 5
            'bus',        6: 6
            'train',      7: 7
            'truck',      8: 8
            'boat'        9: 9
        
            refer Yolact/data/config.py, line 46
            """
            
            #        top_left_x              top_left_y              bottom_left_x               bottom_left_y               label_idx: car (Yolact's code does label-1)
            bbox = [[normalized_top_left[0], normalized_top_left[1], normalized_bottom_right[0], normalized_bottom_right[1], 3-1]]
            bbox = torch.Tensor(bbox)
            #       expand 2D mask to 3D: [num_obj, H, W], where num_object=1, since we have only 1 object: main car
            masks = np.expand_dims(yolact_binary_mask, axis=0).astype(np.float32)
            #       no crowds in our use case
            num_crowds = 0
            
            return image, (bbox, masks, num_crowds)
    
    
    # change all these only ~!to_change
    # other configs are in config.py, you will wanna look at a variable called "yolact_base_config"
    image_path = "/work/u5832291/view_neti_RT/data/DistentangledCarlaScenes/images/blackcar_day_both/rect_001_3_r5000.png"
    seg_map_path = "/work/u5832291/view_neti_RT/data/DistentangledCarlaScenes/masks/blackcar_day_both/rect_001_3_r5000.png"
    model_weights = "/work/u5832291/yixian/YOLACT_edit/weights/yolact_base_54_800000.pth"
    model_config = "yolact_base_config"
    
    
    # no_detection_result_default_value = 10
    # selected_classes = ('bicycle', 'bus', 'car','motorbike', 'person', 'train')    # set in config.py, line 61
    # config.cfg.input_size = (416, 416)
    # config.cfg.test_input_size = (416, 416)
    
    
    ########## TRAINING LOSS for Diffusion model
    yolact_dataset = simulate_YOLACT_dataset(image_path=image_path, seg_map_path=seg_map_path)
    yolact_dataloader = data.DataLoader(yolact_dataset, 
                                        batch_size=1,
                                        num_workers=1,
                                        shuffle=False,
                                        pin_memory=True)
                                        # generator=torch.Generator(device='cuda'))
    
    # from config import config.cfg, Config.Args, config.set_config
    import config
        
    global args
    args = config.Args()
    
    args.trained_model = model_weights  # need to update this as well, for validation's inferencing to work
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
    
    if args.cuda:
        net = net.cuda()
        criterion = criterion.cuda()
            
            
    # feed to model
    for batch in yolact_dataloader:
        
        # optimizer.zero_grad() here
        
        # feed forward
        images, (targets, masks, num_crowds) = batch
        preds = net(images.cuda())
        
        # compute loss
        losses = criterion(net, preds, targets.cuda(), masks.cuda(), num_crowds.cuda())
        
        losses = { k: (v).mean() for k,v in losses.items() } # Mean here because Dataparallel
        loss = sum([losses[k] for k in losses])
        
        # loss.backward() here
        # optimizer.step() here
        
        """
        Loss Key, losses[key]:
         - B: Box Localization Loss
         - C: Class Confidence Loss
         - M: Mask Loss
         - P: Prototype Loss
         - D: Coefficient Diversity Loss
         - E: Class Existence Loss
         - S: Semantic Segmentation Loss
        """
        
        print(f"Box Localization Loss:      {losses['B']}")
        print(f"Class Confidence Loss:      {losses['C']}")
        print(f"Mask Loss:                  {losses['M']}")
        print(f"Semantic Segmentation Loss: {losses['S']}")
        
        # print(f"Prototype Loss:             {losses['P']}")    # not used, no such key available
        # print(f"Coefficient Diversity Loss: {losses['D']}")    # not used, no such key available
        # print(f"Class Existence Loss:       {losses['E']}")    # not used, no such key available
    
    
    
    ########## validation
    # NOTE: loss calculation part is exactly same as during training, just that for validation, there will be another 
    #       function to perform inference on the validation images so that we can save and view them
    
    # loss calculation part: same as above training loss
    
    # inference
    
    # ~!to_change
    img_folder = "/work/u5832291/view_neti_RT/data/DistentangledCarlaScenes/images/blackcar_day_both"

    file_names = os.listdir(img_folder)
    file_names = sorted(file_names)
    lookup_camidx_to_gt_image = {}
    lookup_camidx_to_img_pred = {}
    cam_idxs=[0, 1, 2]
    # cam_idxs=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    
    #### yixian added (for file saving) ~!to_change
    output_imgs_folder = "/work/u5832291/yixian/YOLACT/test_tun_outputs"
    output_imgs_filenames = {index: file_name for index, file_name in zip(cam_idxs, file_names)}


    for cam_idx, file_name in zip(cam_idxs, file_names):
        img_name = os.path.join(img_folder, file_name)
        image = Image.open(img_name).convert("RGB")   # RGB [0-255], HWC
        lookup_camidx_to_gt_image[cam_idx] = image

        image_np = np.array(image)
        lookup_camidx_to_img_pred[cam_idx] = np.expand_dims(image_np, axis=0)   # RGB [0-255], BHWC

    # print(f"--lookup_camidx_to_gt_image={lookup_camidx_to_gt_image}")
    # print(f"--lookup_camidx_to_img_pred={lookup_camidx_to_img_pred}")
    
    validation_image_generation_wrapper(net=net, 
                                        args=args, 
                                        cfg=config.cfg, 
                                        input_imgs=lookup_camidx_to_img_pred, 
                                        output_imgs_folder=output_imgs_folder, 
                                        output_imgs_filenames=output_imgs_filenames)
    print("Done.")
        

if __name__ == '__main__':
    main()

