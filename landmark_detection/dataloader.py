import json
import os
from joblib import Parallel, delayed
import SimpleITK as sitk
import numpy as np
import skimage.transform as skTrans
from torch.utils.data import Dataset
from tqdm import tqdm

class DGFetalMRI(Dataset):
    def __init__(self, dataset_type, mode='', root_path='./dataset/', img_size=128, aug=None, cere_only=True, patch_size=[128, 128, 128], zoom=True):
        # Dataset specific configurations
        if dataset_type == 'lfc':
            self.voxel_path = f"{root_path}{mode}/"
            self.anno_path = f"{root_path}{mode}_label/"
        elif dataset_type == 'atlas':
            self.voxel_path = './dataset/atlas/atlas-reconstruct/'
            self.anno_path = './dataset/atlas/atlas_label/'
        elif dataset_type == 'feta21':
            self.voxel_path = './dataset/feta21/feta21/'
            self.anno_path = './dataset/feta21/feta21_label/'
        else:
            raise ValueError("Invalid dataset type specified")

        self.data_list = []
        self.img_size = img_size
        self.zoom = zoom
        self.aug = aug
        self.patch_size = patch_size
        self.cere_only = cere_only
        case_folders = sorted(os.listdir(self.anno_path), key=int)
        results = Parallel(n_jobs=-1)(delayed(self.process_case_folder)(case_folder) for case_folder in tqdm(case_folders, total=len(case_folders)))
        self.data_list = [result for result in results if result is not None]

    def process_case_folder(self, case_folder):
        json_paths_ori = os.listdir(f"{self.anno_path}/{case_folder}")
        json_paths = sorted(json_paths_ori, key=lambda x: int(x.split('.')[0].split('_')[-1]))
        if len(json_paths) != 6:
            print(case_folder)
            return None

        fetal_mr, origin, direction, factor, spacing = self.process_image(self.voxel_path, case_folder)
        norm_landmarks, case, json_path = self.get_anno(json_paths, self.anno_path, case_folder, origin, factor, spacing)
        case_info = {'name': case_folder, 'origin': origin, 'factor': factor, 'json_path':json_path, 'spacing':spacing}
        return (fetal_mr, norm_landmarks, case_info)

    def window_image(self, image):
        min_intensity = np.percentile(image, 5)
        max_intensity = np.percentile(image, 95)
        windowed_array = np.clip(image, min_intensity, max_intensity)
        normalized = (windowed_array - min_intensity) / (max_intensity - min_intensity)
        return normalized.astype(np.float32)

    def process_image(self, voxel_path, case_folder):
        fetal_mr = sitk.ReadImage(f"{voxel_path}{case_folder}.nii.gz")
        origin = fetal_mr.GetOrigin()
        direction = fetal_mr.GetDirection()
        spacing = fetal_mr.GetSpacing()
        size = fetal_mr.GetSize()
        fetal_mr = sitk.GetArrayFromImage(fetal_mr)
        fetal_mr = self.window_image(fetal_mr)
        factor = np.divide([self.img_size, self.img_size, self.img_size], size)
        factor[0], factor[2] = factor[2], factor[0]
        fetal_mr = skTrans.resize(fetal_mr, (self.img_size, self.img_size, self.img_size), order=3, preserve_range=True)
        return fetal_mr, np.asarray(origin), direction, factor, np.asarray(spacing)

    def get_anno(self, json_paths, anno_path, case_folder, origin, factor, spacing):
        norm_landmarks = []
        annotation_list = []
        metric_list = ['CBD', 'BBD', 'TCD', 'FOD', 'HDV', 'ADV']
        cere_only_idxs = [0, 1, 3] if self.cere_only else []

        metric_idx = 0
        for json_path in json_paths:
            if metric_idx in cere_only_idxs:
                metric_idx += 1
                continue
            path = f"{anno_path}/{case_folder}/{json_path}"
            json_file = json.load(open(path, 'r'))
            length_value = json_file['markups'][0]['measurements'][0]['value']
            start_point, end_point = (np.array(p['position']) for p in json_file['markups'][0]['controlPoints'])
            case = {
                'case_id': case_folder,
                'metric_id': metric_list[metric_idx],
                'length_value': length_value,
                'start_point': start_point,
                'end_point': end_point
            }
            annotation_list.append(case)
            normalized_start_point = (start_point - origin) / (spacing / factor)
            normalized_end_point = (end_point - origin) / (spacing / factor)
            norm_landmarks.append(normalized_start_point)
            norm_landmarks.append(normalized_end_point)
            metric_idx += 1
        return np.asarray(norm_landmarks), annotation_list, path

    def __getitem__(self, idx):
        voxel, gt, case_info = self.data_list[idx]
        if self.aug:
            voxel, gt = self.aug(voxel, gt)
        return voxel, gt, case_info

    def __len__(self):
        return len(self.data_list)
