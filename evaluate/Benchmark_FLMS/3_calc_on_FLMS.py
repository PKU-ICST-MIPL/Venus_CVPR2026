import torch
import json
from PIL import Image
import numpy as np
import os


img_folder = "Venus_CVPR2026/data/Benchmark_FLMS/images"
gt_file = "Venus_CVPR2026/data/Benchmark_FLMS/json/image_crops.json"
pred_file = "Venus_CVPR2026/evaluate/Benchmark_FLMS/2_inference_on_FLMS_turn.json"


def compute_iou_and_disp(gt_crop, pre_crop, im_w, im_h):
    gt_crop = gt_crop[gt_crop[:,0] >= 0] 
    zero_t  = torch.zeros(gt_crop.shape[0])
    over_x1 = torch.maximum(gt_crop[:,0], pre_crop[:,0]) 
    over_y1 = torch.maximum(gt_crop[:,1], pre_crop[:,1])
    over_x2 = torch.minimum(gt_crop[:,2], pre_crop[:,2])
    over_y2 = torch.minimum(gt_crop[:,3], pre_crop[:,3])
    over_w  = torch.maximum(zero_t, over_x2 - over_x1)
    over_h  = torch.maximum(zero_t, over_y2 - over_y1)
    inter   = over_w * over_h
    area1   = (gt_crop[:,2] - gt_crop[:,0]) * (gt_crop[:,3] - gt_crop[:,1])
    area2   = (pre_crop[:,2] - pre_crop[:,0]) * (pre_crop[:,3] - pre_crop[:,1])
    union   = area1 + area2 - inter
    iou     = inter / union 
    disp    = (torch.abs(gt_crop[:, 0] - pre_crop[:, 0]) + torch.abs(gt_crop[:, 2] - pre_crop[:, 2])) / im_w + \
              (torch.abs(gt_crop[:, 1] - pre_crop[:, 1]) + torch.abs(gt_crop[:, 3] - pre_crop[:, 3])) / im_h #[9]
    iou_idx = torch.argmax(iou, dim=-1) 
    dis_idx = torch.argmin(disp, dim=-1)
    index   = dis_idx if (iou[iou_idx] == iou[dis_idx]) else iou_idx
    return iou[index].item(), disp[index].item()


with open(gt_file, 'r') as f:
    data = json.load(f)
with open(pred_file, 'r') as f_pred:
    data_pred = json.load(f_pred)


accum_disp = 0
accum_iou  = 0
alpha = 0.75
alpha_cnt = 0
cnt = 0
valid_crop = 0
for image_name, gt_crop in data.items():
    gt_crop = np.array(gt_crop).reshape(-1,4).astype(np.float32)
    gt_crop= torch.tensor(gt_crop, dtype=torch.float32)
    gt_crop = gt_crop.reshape(-1,4)

    image_file = os.path.join(img_folder, image_name)
    image = Image.open(image_file).convert('RGB')
    width, height = image.size
    crop_values = list(map(float, data_pred[image_name].split()))
    crop = torch.tensor([crop_values], dtype=torch.float32)

    is_valid = torch.all((crop >= 0) & (crop <= 1000))
    if not is_valid:
        iou = 0
        disp = (torch.abs(gt_crop[:, 0] - 0) + torch.abs(gt_crop[:, 2] - width)) / width + \
              (torch.abs(gt_crop[:, 1] - 0) + torch.abs(gt_crop[:, 3] - height)) / height #[9]
        dis_idx = torch.argmin(disp, dim=-1)
        accum_iou += iou
        accum_disp += disp[dis_idx].item()
        cnt += 1
        continue
        
    crop[:, [0, 2]] = crop[:, [0, 2]] * width / 1000  
    crop[:, [1, 3]] = crop[:, [1, 3]] * height / 1000 
    pred_crop = crop.detach()
    pred_crop[:,0::2] = torch.clip(pred_crop[:,0::2], min=0, max=width)
    pred_crop[:,1::2] = torch.clip(pred_crop[:,1::2], min=0, max=height)
    iou, disp = compute_iou_and_disp(gt_crop, pred_crop, width, height)
    
    if iou >= alpha:
        alpha_cnt += 1
    accum_iou += iou
    accum_disp += disp
    cnt += 1
    valid_crop += 1

avg_iou  = accum_iou / cnt
avg_disp = accum_disp / (cnt * 4.0)
avg_recall = float(alpha_cnt) / cnt
print('Test on {} images, IoU={:.4f}, Disp={:.4f}, recall={:.4f}(iou>={:.2f})'.format(
    cnt, avg_iou, avg_disp, avg_recall, alpha))
print("Valid crop coordinates are:", valid_crop)