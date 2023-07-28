import numpy as np
import joblib
import json
from tqdm import tqdm
import argparse


def get_camera_parameters(pred_cam, bbox):
    FOCAL_LENGTH = 5000.
    CROP_SIZE = 224

    bbox_cx, bbox_cy, bbox_w, bbox_h = bbox
    assert bbox_w == bbox_h

    bbox_size = bbox_w
    bbox_x = bbox_cx - bbox_w / 2.
    bbox_y = bbox_cy - bbox_h / 2.

    scale = bbox_size / CROP_SIZE

    cam_intrinsics = np.eye(3)
    cam_intrinsics[0, 0] = FOCAL_LENGTH * scale
    cam_intrinsics[1, 1] = FOCAL_LENGTH * scale
    cam_intrinsics[0, 2] = bbox_size / 2. + bbox_x 
    cam_intrinsics[1, 2] = bbox_size / 2. + bbox_y

    cam_s, cam_tx, cam_ty = pred_cam
    trans = [cam_tx, cam_ty, 2*FOCAL_LENGTH/(CROP_SIZE*cam_s + 1e-9)]

    cam_extrinsics = np.eye(4)
    cam_extrinsics[:3, 3] = trans

    return cam_intrinsics, cam_extrinsics

def _main(input_folder, output):
    with open(input_folder, 'rb') as f:
        ct = joblib.load(f)

    res = ct[1]

    metas = dict()
    for i, (pose, beta, pred_cam, bbox) in tqdm(enumerate(zip(res['pose'], res['betas'], res['pred_cam'], res['bboxes']))):
        meta = dict()
        meta["poses"] = pose.tolist()
        meta["betas"] = beta.tolist()
        intri, extri = get_camera_parameters(pred_cam, bbox)
        meta["cam_intrinsics"] = intri.tolist()
        meta["cam_extrinsics"] = extri.tolist()
        
        item_id = "%06d" % (i + 1)
        metas[item_id] = meta
    # print(metas.keys())
    with open(output, 'w') as f:
        f.write(json.dumps(metas))



if __name__ == '__main__':
    parser = argparse.ArgumentParser('demo', add_help=False)
    parser.add_argument('--input', default="vibe_output.pkl", type=str)
    parser.add_argument('--output', default='metadata.json', type=str)
    args = parser.parse_args() 
    _main(args.input, args.output)
