import cv2
import os
import os.path as osp

from tqdm import tqdm
from priors.mask import build_model
from priors.pose.VIBE import PoseEstimator

from configs import cfg
import numpy as np
import pickle
import argparse

from core.utils.log_util import Logger
from core.data import create_dataloader
from core.nets import create_network
from core.train import create_trainer, create_optimizer

from third_parties.smpl.smpl_numpy import SMPL

MODEL_DIR = "./third_parties/smpl/models"


def make_args():
    parser = argparse.ArgumentParser('easyHumanNeRF', add_help=False)
    parser.add_argument('--workspace', default="./workspace/demo02/", type=str)
    args = parser.parse_args()
    return args


def prepare_dataset(frame_info, workspace):
    sex = "neutral"
    output_path = workspace

    smpl_model = SMPL(sex=sex, model_dir=MODEL_DIR)

    cameras = {}
    mesh_infos = {}
    all_betas = []
    for frame_base_name in tqdm(frame_info):
        cam_body_info = frame_info[frame_base_name]
        poses = np.array(cam_body_info['poses'], dtype=np.float32)
        betas = np.array(cam_body_info['betas'], dtype=np.float32)
        K = np.array(cam_body_info['cam_intrinsics'], dtype=np.float32)
        E = np.array(cam_body_info['cam_extrinsics'], dtype=np.float32)

        all_betas.append(betas)

        ##############################################
        # Below we tranfer the global body rotation to camera pose

        # Get T-pose joints
        _, tpose_joints = smpl_model(np.zeros_like(poses), betas)

        # get global Rh, Th
        pelvis_pos = tpose_joints[0].copy()
        Th = pelvis_pos
        Rh = poses[:3].copy()

        # get refined T-pose joints
        tpose_joints = tpose_joints - pelvis_pos[None, :]

        # remove global rotation from body pose
        poses[:3] = 0

        # get posed joints using body poses without global rotation
        _, joints = smpl_model(poses, betas)
        joints = joints - pelvis_pos[None, :]

        mesh_infos[frame_base_name] = {
            'Rh': Rh,
            'Th': Th,
            'poses': poses,
            'joints': joints,
            'tpose_joints': tpose_joints
        }

        cameras[frame_base_name] = {
            'intrinsics': K,
            'extrinsics': E
        }

    # write camera infos
    with open(os.path.join(output_path, 'cameras.pkl'), 'wb') as f:
        pickle.dump(cameras, f)

    # write mesh infos
    with open(os.path.join(output_path, 'mesh_infos.pkl'), 'wb') as f:
        pickle.dump(mesh_infos, f)

    # write canonical joints
    avg_betas = np.mean(np.stack(all_betas, axis=0), axis=0)
    smpl_model = SMPL(sex, model_dir=MODEL_DIR)
    _, template_joints = smpl_model(np.zeros(72), avg_betas)
    with open(os.path.join(output_path, 'canonical_joints.pkl'), 'wb') as f:
        pickle.dump({'joints': template_joints, }, f)


def _main():
    args = make_args()
    folder = args.workspace
    image_folder = osp.join(folder, "images")
    mask_folder = osp.join(folder, "masks")
    if not osp.exists(mask_folder):
        os.makedirs(mask_folder)

    # load  segmentation tool
    masker = build_model()
    print("\033[0;31;40mPerson Segmentation...\033[0m")
    for img_name in tqdm(os.listdir(image_folder)):
        ima = osp.join(image_folder, img_name)
        img = cv2.imread(ima)
        mask = masker.seg_one(img)
        save_name = osp.join(mask_folder, img_name)
        cv2.imwrite(save_name, mask)

    # load SMPL pose estimator
    poser = PoseEstimator(image_folder)
    print("\033[0;31;40mPose Estimation...\033[0m")
    results = poser.inference(image_folder, "output", save_result=False)
    # poser.render(results, "output")
    metas = poser.output_for_humannerf(results, output_path=osp.join(folder, "metadata.json"))

    # prepare HumanNeRF dataset
    prepare_dataset(metas, args.workspace)

    # training HumanNeRF
    print("\033[0;31;40mTraining HumanNeRF...\033[0m")
    log = Logger()
    log.print_config()

    model = create_network()
    optimizer = create_optimizer(model)
    trainer = create_trainer(model, optimizer, args.workspace)

    train_loader = create_dataloader('train', workspace=args.workspace)

    # estimate start epoch
    epoch = trainer.iter // len(train_loader) + 1
    print("maxiter:", cfg.train.maxiter)
    print(epoch)
    while True:
        if trainer.iter > cfg.train.maxiter:
            break

        trainer.train(epoch=epoch,
                      train_dataloader=train_loader)
        epoch += 1

    trainer.finalize()


if __name__ == "__main__":
    _main()
