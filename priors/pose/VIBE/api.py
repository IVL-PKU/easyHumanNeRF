import os
import os.path as osp

os.environ['PYOPENGL_PLATFORM'] = 'egl'

import cv2
import json
import time
import torch
import joblib
import shutil
import colorsys
import argparse
import numpy as np
from tqdm import tqdm
from multi_person_tracker import MPT
from torch.utils.data import DataLoader

from .lib.models.vibe import VIBE_Demo
from .lib.utils.renderer import Renderer
from .lib.dataset.inference import Inference
from .lib.utils.smooth_pose import smooth_pose
from .lib.data_utils.kp_utils import convert_kps
from .lib.utils.pose_tracker import run_posetracker

from .lib.utils.demo_utils import (
    download_youtube_clip,
    smplify_runner,
    convert_crop_coords_to_orig_img,
    convert_crop_cam_to_orig_img,
    prepare_rendering_results,
    video_to_images,
    images_to_video,
    download_ckpt,
)

MIN_NUM_FRAMES = 25


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
    trans = [cam_tx, cam_ty, 2 * FOCAL_LENGTH / (CROP_SIZE * cam_s + 1e-9)]

    cam_extrinsics = np.eye(4)
    cam_extrinsics[:3, 3] = trans

    return cam_intrinsics, cam_extrinsics


def download_vibe_data(data_path):
    if not osp.exists(data_path):
        os.makedirs(data_path)
        os.system("gdown 'https://drive.google.com/uc?id=1untXhYOLQtpNEy4GTY_0fL_H-k6cTf_r'")
        os.system("unzip vibe_data.zip")
        os.remove("vibe_data.zip")
        os.system("mv vibe_data %s" % data_path)
        os.system("mv %s/yolov3.weights $HOME/.torch/models" % data_path)


class PoseEstimator:
    def __init__(self, input_folder, device=torch.device('cuda')):
        data_path = "/tmp/vibe_data"
        # download_vibe_data(data_path)

        self.input_folder = input_folder
        self.model = VIBE_Demo(
            seqlen=16,
            n_layers=2,
            hidden_size=1024,
            add_linear=True,
            use_residual=True,
        ).to(device)
        pretrained_file = download_ckpt(outdir=data_path, use_3dpw=False)
        ckpt = torch.load(pretrained_file)
        ckpt = ckpt['gen_state_dict']
        self.model.load_state_dict(ckpt, strict=False)
        self.model.eval()
        self.device = device

        self.mot = MPT(
            device=device,
            batch_size=12,
            display=False,
            detector_type="yolo",
            output_format='dict',
            yolo_img_size=416)

        _, self.num_frames, self.img_shape = video_to_images(input_folder,
                                                             return_info=True)

    def mot_inference(self, image_folder):
        tracking_results = self.mot(image_folder)
        for person_id in list(tracking_results.keys()):
            if tracking_results[person_id]['frames'].shape[0] < MIN_NUM_FRAMES:
                del tracking_results[person_id]
        return tracking_results

    def inference(self, image_folder, output_path, save_result=True):
        if not osp.exists(output_path):
            os.makedirs(output_path)
        orig_height, orig_width = self.img_shape[:2]
        tracking_results = self.mot_inference(image_folder)
        vibe_results = {}
        for person_id in tqdm(list(tracking_results.keys())):
            joints2d = None
            bboxes = tracking_results[person_id]['bbox']
            frames = tracking_results[person_id]['frames']

            dataset = Inference(
                image_folder=image_folder,
                frames=frames,
                bboxes=bboxes,
                joints2d=joints2d,
                scale=1.1,
            )

            bboxes = dataset.bboxes
            frames = dataset.frames
            has_keypoints = True if joints2d is not None else False

            dataloader = DataLoader(dataset, batch_size=450, num_workers=16)

            with torch.no_grad():
                pred_cam, pred_verts, pred_pose, = [], [], []
                pred_betas, pred_joints3d, smpl_joints2d, norm_joints2d = [], [], [], []

                for batch in dataloader:
                    if has_keypoints:
                        batch, nj2d = batch
                        norm_joints2d.append(nj2d.numpy().reshape(-1, 21, 3))

                    batch = batch.unsqueeze(0)
                    batch = batch.to(self.device)

                    batch_size, seqlen = batch.shape[:2]
                    output = self.model(batch)[-1]

                    pred_cam.append(output['theta'][:, :, :3].reshape(batch_size * seqlen, -1))
                    pred_verts.append(output['verts'].reshape(batch_size * seqlen, -1, 3))
                    pred_pose.append(output['theta'][:, :, 3:75].reshape(batch_size * seqlen, -1))
                    pred_betas.append(output['theta'][:, :, 75:].reshape(batch_size * seqlen, -1))
                    pred_joints3d.append(output['kp_3d'].reshape(batch_size * seqlen, -1, 3))
                    smpl_joints2d.append(output['kp_2d'].reshape(batch_size * seqlen, -1, 2))

                pred_cam = torch.cat(pred_cam, dim=0)
                pred_verts = torch.cat(pred_verts, dim=0)
                pred_pose = torch.cat(pred_pose, dim=0)
                pred_betas = torch.cat(pred_betas, dim=0)
                pred_joints3d = torch.cat(pred_joints3d, dim=0)
                smpl_joints2d = torch.cat(smpl_joints2d, dim=0)
                del batch

            # ========= Save results to a pickle file ========= #
            pred_cam = pred_cam.cpu().numpy()
            pred_verts = pred_verts.cpu().numpy()
            pred_pose = pred_pose.cpu().numpy()
            pred_betas = pred_betas.cpu().numpy()
            pred_joints3d = pred_joints3d.cpu().numpy()
            smpl_joints2d = smpl_joints2d.cpu().numpy()

            # Runs 1 Euro Filter to smooth out the results
            if True:
                min_cutoff = 0.004
                beta = 1.5
                print(f'Running smoothing on person {person_id}, min_cutoff: {min_cutoff}, beta: {beta}')
                pred_verts, pred_pose, pred_joints3d = smooth_pose(pred_pose, pred_betas,
                                                                   min_cutoff=min_cutoff, beta=beta)

            orig_cam = convert_crop_cam_to_orig_img(
                cam=pred_cam,
                bbox=bboxes,
                img_width=orig_width,
                img_height=orig_height
            )

            joints2d_img_coord = convert_crop_coords_to_orig_img(
                bbox=bboxes,
                keypoints=smpl_joints2d,
                crop_size=224,
            )

            output_dict = {
                'pred_cam': pred_cam,
                'orig_cam': orig_cam,
                'verts': pred_verts,
                'pose': pred_pose,
                'betas': pred_betas,
                'joints3d': pred_joints3d,
                'joints2d': joints2d,
                'joints2d_img_coord': joints2d_img_coord,
                'bboxes': bboxes,
                'frame_ids': frames,
            }

            vibe_results[person_id] = output_dict

        if save_result:
            joblib.dump(vibe_results, os.path.join(output_path, "vibe_output.pkl"))

        return vibe_results

    def render(self, vibe_results, output_path, display=False):
        renderer = Renderer(resolution=(self.img_shape[0], self.img_shape[1]),
                            orig_img=True, wireframe=True)

        output_img_folder = f'{self.input_folder}_output'
        os.makedirs(output_img_folder, exist_ok=True)

        print(f'Rendering output video, writing frames to {output_img_folder}')

        # prepare results for rendering
        frame_results = prepare_rendering_results(vibe_results, self.num_frames)
        mesh_color = {k: colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0) for k in vibe_results.keys()}

        image_file_names = sorted([
            os.path.join(self.input_folder, x)
            for x in os.listdir(self.input_folder)
            if x.endswith('.png') or x.endswith('.jpg')
        ])

        for frame_idx in tqdm(range(len(image_file_names))):
            img_fname = image_file_names[frame_idx]
            img = cv2.imread(img_fname)

            side_img = np.zeros_like(img)

            for person_id, person_data in frame_results[frame_idx].items():
                frame_verts = person_data['verts']
                frame_cam = person_data['cam']

                mc = mesh_color[person_id]

                mesh_filename = None

                mesh_folder = os.path.join(output_path, 'meshes', f'{person_id:04d}')
                os.makedirs(mesh_folder, exist_ok=True)
                mesh_filename = os.path.join(mesh_folder, f'{frame_idx:06d}.obj')

                img = renderer.render(
                    img,
                    frame_verts,
                    cam=frame_cam,
                    color=mc,
                    mesh_filename=mesh_filename,
                )

                side_img = renderer.render(
                    side_img,
                    frame_verts,
                    cam=frame_cam,
                    color=mc,
                    angle=270,
                    axis=[0, 1, 0],
                )

            img = np.concatenate([img, side_img], axis=1)

            cv2.imwrite(os.path.join(output_img_folder, f'{frame_idx:06d}.png'), img)

            if display:
                cv2.imshow('Video', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # ========= Save rendered video ========= #
        save_name = f'{self.input_folder.replace(".mp4", "")}_vibe_result.mp4'
        save_name = os.path.join(output_path, save_name)
        print(f'Saving result video to {save_name}')
        images_to_video(img_folder=output_img_folder, output_vid_file=save_name)
        shutil.rmtree(output_img_folder)

    def output_for_humannerf(self, vibe_results, output_path="metadata.json", save_results=True):
        res = vibe_results[list(vibe_results.keys())[0]]
        metas = dict()
        for i, (pose, beta, pred_cam, bbox) in tqdm(
                enumerate(zip(res['pose'], res['betas'], res['pred_cam'], res['bboxes']))):
            meta = dict()
            meta["poses"] = pose.tolist()
            meta["betas"] = beta.tolist()
            intri, extri = get_camera_parameters(pred_cam, bbox)
            meta["cam_intrinsics"] = intri.tolist()
            meta["cam_extrinsics"] = extri.tolist()

            item_id = "%06d" % (i + 1)
            metas[item_id] = meta
        # print(metas.keys())
        if save_results:
            with open(output_path, 'w') as f:
                f.write(json.dumps(metas))
        return metas


def _main():
    folder = "your/path/images"
    poser = PoseEstimator(folder)
    results = poser.inference(folder, "output")
    # poser.render(results, "output")
    poser.output_for_humannerf(results)


if __name__ == "__main__":
    _main()
