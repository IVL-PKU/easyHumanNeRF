import matplotlib.pyplot as plt
import torch
import cv2
import yaml
from torchvision import transforms
import numpy as np
from tqdm import tqdm

from utils.datasets import letterbox
from utils.general import non_max_suppression_mask_conf
import argparse

from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import paste_masks_in_image


def get_model(device="cuda:0"):
    weigths = torch.load('yolov7-mask.pt')
    model = weigths['model']
    model = model.half().to(device)
    _ = model.eval()
    return model

def preprocess(image, device="cuda:0"):
    image = letterbox(image, 640, stride=64, auto=True)[0]
    image_ = image.copy()
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))
    image = image.to(device)
    image = image.half()
    return image

def postprocess(image, model, hyp, output):
    inf_out, train_out, attn, mask_iou, bases, sem_output = output['test'], output['bbox_and_cls'], output['attn'], output['mask_iou'], output['bases'], output['sem']
    bases = torch.cat([bases, sem_output], dim=1)
    nb, _, height, width = image.shape
    names = model.names
    pooler_scale = model.pooler_scale
    pooler = ROIPooler(output_size=hyp['mask_resolution'], scales=(pooler_scale,), sampling_ratio=1, pooler_type='ROIAlignV2', canonical_level=2)
    output, output_mask, output_mask_score, output_ac, output_ab = non_max_suppression_mask_conf(inf_out, attn, bases, pooler, hyp, conf_thres=0.25, iou_thres=0.65, merge=False, mask_iou=None)
    pred, pred_masks = output[0], output_mask[0]
    base = bases[0]
    bboxes = Boxes(pred[:, :4])
    original_pred_masks = pred_masks.view(-1, hyp['mask_resolution'], hyp['mask_resolution'])
    pred_masks = retry_if_cuda_oom(paste_masks_in_image)( original_pred_masks, bboxes, (height, width), threshold=0.5)
    pred_masks_np = pred_masks.detach().cpu().numpy()
    pred_cls = pred[:, 5].detach().cpu().numpy()
    pred_conf = pred[:, 4].detach().cpu().numpy()
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    nbboxes = bboxes.tensor.detach().cpu().numpy().astype(np.int)
    pnimg = nimg.copy()

    return pnimg, pred_masks_np, nbboxes, pred_cls, pred_conf



def _main():
    with open('data/hyp.scratch.mask.yaml') as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(device=device)

    image = cv2.imread('001.png')
    original_shape = image.shape
    origin = image.copy()

    image = preprocess(image, device=device)


    output = model(image)
    pnimg, pred_masks_np, nbboxes, pred_cls, pred_conf = postprocess(image, model, hyp, output)

    mask = np.zeros_like(pnimg)

    for one_mask, bbox, cls, conf in zip(pred_masks_np, nbboxes, pred_cls, pred_conf):
        if cls != 0.0: # only handle human
            continue
        if conf < 0.25:
            continue
        white = [255, 255, 255]
        mask[one_mask] = np.array(white, dtype=np.uint8)


        color = [np.random.randint(255), np.random.randint(255), np.random.randint(255)]
        pnimg[one_mask] = pnimg[one_mask] * 0.5 + np.array(color, dtype=np.uint8) * 0.5
        # pnimg = cv2.rectangle(pnimg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

    # print(pnimg.shape)
    # cv2.imwrite('hh.png', pnimg)

    mask = cv2.resize(mask, [original_shape[1], original_shape[0]])
    cv2.imwrite('mask.png', mask)

    # tosave = origin * 0.5 + mask * 0.5
    # cv2.imwrite('res.png', tosave)
    

def _run_folder(folder, save_folder):
    import os
    import os.path as osp

    with open('data/hyp.scratch.mask.yaml') as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(device=device)

    images = os.listdir(folder)
    images = [osp.join(folder, f) for f in images]

    for im in tqdm(images):
        image = cv2.imread(im)
        origin = image.copy()
        image = preprocess(image, device=device)
        output = model(image)
        pnimg, pred_masks_np, nbboxes, pred_cls, pred_conf = postprocess(image, model, hyp, output)
        mask = np.zeros_like(pnimg)

        for one_mask, bbox, cls, conf in zip(pred_masks_np, nbboxes, pred_cls, pred_conf):
            if cls != 0.0 or conf < 0.25:
                continue
            white = [255, 255, 255]
            mask[one_mask] = np.array(white, dtype=np.uint8)

        h, w, c = origin.shape
        # mask = cv2.resize(mask, [w, h])
        cv2.imwrite(osp.join(save_folder, osp.basename(im)), mask)
        # break


if __name__ == '__main__':
    # _main()
    parser = argparse.ArgumentParser('demo', add_help=False)
    parser.add_argument('--input', default="./images/", type=str)
    parser.add_argument('--output', default='./masks/', type=str)
    args = parser.parse_args()

    print(args.output)
    _run_folder(args.input, args.output)
