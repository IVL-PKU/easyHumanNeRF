import os
import os.path as osp
import sys

import cv2
from torchvision import transforms

abs_path = os.path.realpath(__file__)
sys.path.insert(0, osp.dirname(abs_path))

from .models import *
from .models.utils import non_max_suppression_mask_conf, letterbox

from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import paste_masks_in_image


class Masker:
    def __init__(self, ckpt='yolov7-mask.pt',
                 device="cuda:0"):
        self.model = torch.load(ckpt)['model']
        self.model = self.model.half().to(device)
        self.model.eval()

        self.device = device
        self.hyp = {'lr0': 0.01, 'lrf': 0.1, 'momentum': 0.937, 'weight_decay': 0.0005, 'warmup_epochs': 3.0,
                    'warmup_momentum': 0.8, 'warmup_bias_lr': 0.1, 'box': 0.05, 'cls': 0.3, 'cls_pw': 1.0, 'obj': 0.7,
                    'obj_pw': 1.0, 'mask': 0.05, 'mask_pw': 1.0, 'pointrend': 0.05, 'iou_t': 0.2, 'anchor_t': 4.0,
                    'fl_gamma': 0.0, 'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4, 'degrees': 0.0, 'translate': 0.1,
                    'scale': 0.5, 'shear': 0.0, 'perspective': 0.0, 'flipud': 0.0, 'fliplr': 0.5, 'mosaic': 1.0,
                    'mixup': 0.0, 'copy_paste': 0.0, 'paste_in': 0.0, 'attn_resolution': 14, 'num_base': 5,
                    'mask_resolution': 56}
        # with open(hyp_yaml) as f:
        #     self.hyp = yaml.load(f, Loader=yaml.FullLoader)

    def pre_process(self, image):
        image = letterbox(image, 640, stride=64, auto=True)[0]
        image = transforms.ToTensor()(image)
        image = torch.tensor(np.array([image.numpy()]))
        image = image.to(self.device)
        image = image.half()
        return image

    def seg_one(self, image):
        image = self.pre_process(image)
        output = self.model(image)
        pnimg, pred_masks_np, nbboxes, pred_cls, pred_conf = self.post_process(image, output)
        mask = np.zeros_like(pnimg)

        for one_mask, bbox, cls, conf in zip(pred_masks_np, nbboxes, pred_cls, pred_conf):
            if cls != 0.0 or conf < 0.25:
                continue
            white = [255, 255, 255]
            mask[one_mask] = np.array(white, dtype=np.uint8)
        return mask

    def post_process(self, image, output):
        inf_out, train_out, attn, mask_iou, bases, sem_output = output['test'], \
            output['bbox_and_cls'], output['attn'], \
            output['mask_iou'], output['bases'], output['sem']

        bases = torch.cat([bases, sem_output], dim=1)
        nb, _, height, width = image.shape
        pooler_scale = self.model.pooler_scale
        pooler = ROIPooler(output_size=self.hyp['mask_resolution'], scales=[pooler_scale, ], sampling_ratio=1,
                           pooler_type='ROIAlignV2', canonical_level=2)
        output, output_mask, output_mask_score, output_ac, output_ab = non_max_suppression_mask_conf(inf_out, attn,
                                                                                                     bases, pooler,
                                                                                                     self.hyp,
                                                                                                     conf_thres=0.25,
                                                                                                     iou_thres=0.65,
                                                                                                     merge=False,
                                                                                                     mask_iou=None)
        pred, pred_masks = output[0], output_mask[0]
        bboxes = Boxes(pred[:, :4])
        original_pred_masks = pred_masks.view(-1, self.hyp['mask_resolution'], self.hyp['mask_resolution'])
        pred_masks = retry_if_cuda_oom(paste_masks_in_image)(original_pred_masks, bboxes, (height, width),
                                                             threshold=0.5)
        pred_masks_np = pred_masks.detach().cpu().numpy()
        pred_cls = pred[:, 5].detach().cpu().numpy()
        pred_conf = pred[:, 4].detach().cpu().numpy()
        nimg = image[0].permute(1, 2, 0) * 255
        nimg = nimg.cpu().numpy().astype(np.uint8)
        nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
        nbboxes = bboxes.tensor.detach().cpu().numpy().astype(np.int)
        pnimg = nimg.copy()

        return pnimg, pred_masks_np, nbboxes, pred_cls, pred_conf


def build_model():
    if not osp.exists('/tmp/yolo_seg'):
        os.system("mkdir -p /tmp/yolo_seg")
    cmd = "wget -O /tmp/yolo_seg/yolov7-mask.pt " \
          "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-mask.pt"
    if not osp.exists('/tmp/yolo_seg/yolov7-mask.pt'):
        os.system(cmd)
    masker = Masker(ckpt="/tmp/yolo_seg/yolov7-mask.pt")
    return masker


def _main():
    masker = Masker()
    img = cv2.imread("/home/liweiliao/Projects/NeRF/human/liwei_0721/images/000001.png")
    print(img.shape)

    mask = masker.seg_one(img)
    cv2.imwrite('mask.png', mask)


if __name__ == '__main__':
    # _main()
    build_model()
