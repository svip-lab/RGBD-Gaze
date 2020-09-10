import os
from tqdm import tqdm
import numpy as np
import pandas as pd


def gen_filelist(data_root, out_root=None, log_root=None):
    rgb_dir = "color"
    gt_dir = "coordinate"
    eye_coord_dir = "eyecoordinate"
    depth_dir = "projected_depth_calibration"
    landmark_dir = "landmark"

    size_train_set = 159

    anno_filedict = dict(
        face_image=[],
        face_depth=[],
        face_landmark=[],
        face_bbox=[],
        left_eye_image=[],
        right_eye_image=[],
        left_eye_depth=[],
        right_eye_depth=[],
        left_eye_bbox=[],
        right_eye_bbox=[],
        left_eye_coord=[],
        right_eye_coord=[],
        gaze_point=[],
        is_train=[],
        has_landmark=[]
    )

    # filter person id
    rgb_ids = set(os.listdir(os.path.join(data_root, rgb_dir)))
    gt_ids = set(os.listdir(os.path.join(data_root, gt_dir)))
    eye_coord_ids = set(os.listdir(os.path.join(data_root, eye_coord_dir)))
    depth_ids = set(os.listdir(os.path.join(data_root, depth_dir)))
    log_root = data_root if log_root is None else log_root

    valid_pids = rgb_ids.intersection(gt_ids).intersection(eye_coord_ids).intersection(depth_ids)
    print(f"pids valid/all: {len(valid_pids)}/{len(rgb_ids)} ")

    # filter samples
    valid_sample_ids = {}
    has_landmark = {}
    for pid in tqdm(valid_pids, desc="filtering samples"):
        valid_sample_ids[pid] = set()
        samples = [sample for sample in os.listdir(os.path.join(data_root, gt_dir, pid)) if sample.endswith(".npy")]
        for sample in samples:
            is_valid = True
            sample_id = f"{int(sample[3:8]) + 1:05d}"
            err_msg = f"[pid={pid}, sid={sample_id}]\n"
            has_landmark[(pid, sample_id)] = True
            # verify face bbox
            if not os.path.isfile(os.path.join(data_root, rgb_dir, pid, "color" + sample_id + "_face.txt")):
                err_msg += f'missing {os.path.join(rgb_dir, pid, "color" + sample_id + "_face.txt")}\n'
                is_valid = False
            # verify face landmark
            if not os.path.isfile(os.path.join(data_root, landmark_dir, pid, sample_id + ".npy")):
                err_msg += f'missing {os.path.join(data_root, landmark_dir, pid, sample_id + ".npy")}\n'
                has_landmark[(pid, sample_id)] = False
            # verify left eye bbox
            if not os.path.isfile(os.path.join(data_root, rgb_dir, pid, "color" + sample_id + "_left_eye.txt")):
                err_msg += f'missing {os.path.join(rgb_dir, pid, "color" + sample_id + "_left_eye.txt")}\n'
                is_valid = False
            # verify left eye image
            if not os.path.isfile(os.path.join(data_root, rgb_dir, pid, "color" + sample_id + "_lefteye.jpg")):
                err_msg += f'missing {os.path.join(rgb_dir, pid, "color" + sample_id + "_lefteye.jpg")}\n'
                is_valid = False
            # verify right eye bbox
            if not os.path.isfile(os.path.join(data_root, rgb_dir, pid, "color" + sample_id + "_right_eye.txt")):
                err_msg += f'missing {os.path.join(rgb_dir, pid, "color" + sample_id + "_right_eye.txt")}\n'
                is_valid = False
            # verify right eye image
            if not os.path.isfile(os.path.join(data_root, rgb_dir, pid, "color" + sample_id + "_righteye.jpg")):
                err_msg += f'missing {os.path.join(rgb_dir, pid, "color" + sample_id + "_righteye.jpg")}\n'
                is_valid = False
            # verify gaze gt
            if not os.path.isfile(os.path.join(data_root, gt_dir, pid, "xy_" + "{:05d}".format(int(sample_id) - 1) + ".npy")):
                err_msg += f'missing {os.path.join(gt_dir, pid, "xy_" + "{:05d}".format(int(sample_id) - 1) + ".npy")}\n'
                is_valid = False
            # verify left eye coordinate
            if not os.path.isfile(os.path.join(data_root, eye_coord_dir, pid, sample_id + "_le.npy")):
                err_msg += f'missing {os.path.join(eye_coord_dir, pid, sample_id + "_le.npy")}\n'
                is_valid = False
            # verify right eye coordinate
            if not os.path.isfile(os.path.join(data_root, eye_coord_dir, pid, sample_id + "_re.npy")):
                err_msg += f'missing {os.path.join(eye_coord_dir, pid, sample_id + "_re.npy")}\n'
                is_valid = False
            # verify face depth
            if not os.path.isfile(os.path.join(data_root, depth_dir, pid, "projected_depth" + sample_id + "_face.png")):
                err_msg += f'missing {os.path.join(depth_dir, pid, "projected_depth" + sample_id + "_face.png")}\n'
                is_valid = False
            # verify left eye depth
            if not os.path.isfile(
                    os.path.join(data_root, depth_dir, pid, "projected_depth" + sample_id + "_lefteye.png")):
                err_msg += f'missing {os.path.join(depth_dir, pid, "projected_depth" + sample_id + "_lefteye.png")}\n'
                is_valid = False
            # verify right eye depth
            if not os.path.isfile(
                    os.path.join(data_root, depth_dir, pid, "projected_depth" + sample_id + "_righteye.png")):
                err_msg += f'missing {os.path.join(depth_dir, pid, "projected_depth" + sample_id + "_righteye.png")}\n'
                is_valid = False
            if is_valid:
                valid_sample_ids[pid].add(sample_id)
            else:
                with open(os.path.join(log_root, "error.log"), "a+") as fp:
                    print(err_msg, file=fp)
        if len(valid_sample_ids[pid]) == 0:
            valid_sample_ids.pop(pid)
            print(f"warn: pid={pid} has no valid samples, ignored.")

    id_list = []
    pids = sorted(valid_sample_ids.keys())
    for i, pid in enumerate(tqdm(pids, desc="preparing filelist")):
        sample_ids = sorted(valid_sample_ids[pid])
        for sample_id in sample_ids:
            # sample_id = f"{int(sample[3:8]) + 1:05d}"
            id_list.append(pid + sample_id)
            anno_filedict["face_image"].append(os.path.join(rgb_dir, pid, "color" + sample_id + "_face.jpg"))
            anno_filedict["face_depth"].append(
                os.path.join(depth_dir, pid, "projected_depth" + sample_id + "_face.png"))
            anno_filedict["has_landmark"].append(has_landmark[(pid, sample_id)])
            anno_filedict["face_landmark"].append(os.path.join(landmark_dir, pid, sample_id + ".npy"))
            anno_filedict["face_bbox"].append(os.path.join(rgb_dir, pid, "color" + sample_id + "_face.txt"))
            anno_filedict["left_eye_image"].append(os.path.join(rgb_dir, pid, "color" + sample_id + "_lefteye.jpg"))
            anno_filedict["right_eye_image"].append(os.path.join(rgb_dir, pid, "color" + sample_id + "_righteye.jpg"))
            anno_filedict["left_eye_depth"].append(
                os.path.join(depth_dir, pid, "projected_depth" + sample_id + "_lefteye.png"))
            anno_filedict["right_eye_depth"].append(
                os.path.join(depth_dir, pid, "projected_depth" + sample_id + "_righteye.png"))
            anno_filedict["left_eye_bbox"].append(os.path.join(rgb_dir, pid, "color" + sample_id + "_left_eye.txt"))
            anno_filedict["right_eye_bbox"].append(os.path.join(rgb_dir, pid, "color" + sample_id + "_right_eye.txt"))
            anno_filedict["left_eye_coord"].append(os.path.join(eye_coord_dir, pid, sample_id + "_le.npy"))
            anno_filedict["right_eye_coord"].append(os.path.join(eye_coord_dir, pid, sample_id + "_re.npy"))
            anno_filedict["gaze_point"].append(os.path.join(gt_dir, pid, "xy_" + "{:05d}".format(int(sample_id) - 1) + ".npy"))
            if i <= size_train_set:
                anno_filedict["is_train"].append(1)
            else:
                anno_filedict["is_train"].append(0)

    df = pd.DataFrame(data=anno_filedict, index=id_list)
    if out_root is None:
        out_root = data_root
    os.makedirs(out_root, exist_ok=True)
    # write training file list
    df[df["is_train"] == 1].to_csv(os.path.join(out_root, "train_filelist.csv"),)
    df[df["is_train"] == 0].to_csv(os.path.join(out_root, "val_filelist.csv"))


if __name__ == '__main__':
    # gen_filelist(r"/home/ziheng/datasets/gaze")
    gen_filelist(r'D:\data\gaze')
