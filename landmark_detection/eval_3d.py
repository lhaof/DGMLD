import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import DGFetalMRI
from network import ResUnet3D, ViTPose3D
from visualization import plot_3D_points

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt

import numpy as np
import scipy.ndimage

def generate_gaussian_heatmap(shape, points, sigma=5):
    """
    Generates a heatmap with Gaussian peaks at the specified points.
    
    Args:
    - shape (tuple): Shape of the heatmap (D, H, W).
    - points (numpy.ndarray): Array of points (num_points, 3) where each row is (z, y, x).
    - sigma (int or float): Standard deviation of the Gaussian kernel.
    
    Returns:
    - numpy.ndarray: A heatmap with Gaussian peaks.
    """
    heatmap = np.zeros(shape, dtype=np.float32)
    for point in points:
        z, y, x = point.astype(int)
        heatmap[z, y, x] = 1  # Place a peak

    # Apply Gaussian filter
    heatmap = scipy.ndimage.gaussian_filter(heatmap, sigma=sigma, mode='constant')
    return heatmap

def plot_heatmap_2D(heatmap, gt, save_name):
    """
    Plots and saves 2D heatmaps from a 3D or 4D tensor around the GT coordinates using both the model output and GT-based Gaussian heatmap.
    
    Args:
    - heatmap (torch.Tensor): Model's heatmap tensor, shape (batch_size, num_channels, D, H, W).
    - gt (numpy.ndarray): Ground truth coordinates, shape (num_keypoints, 3) for z, y, x coordinates.
    - save_name (str): Base path to save the heatmap images.
    """
    if heatmap.ndim == 5:
        heatmap = heatmap.squeeze().detach().cpu().numpy()[0]  # Simplify to first sample, first channel
    elif heatmap.ndim == 4:
        heatmap = heatmap.squeeze().detach().cpu().numpy()  # Simplify to first sample
    print(heatmap.shape)
    # Generate GT-based heatmap
    gt_heatmap = generate_gaussian_heatmap(heatmap.shape, gt, sigma=5)
    
    # print(gt)
    # Compute an average index along the z-axis from the GT coordinates
    z_indices = gt[0, 1].astype(int)
    
    # Select the slice at the computed z index
    model_slice = heatmap[:,z_indices,:]
    gt_slice = gt_heatmap[:,z_indices,:]

    # Create plots
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(model_slice, cmap='jet', interpolation='nearest')
    plt.colorbar()
    plt.title('Model Heatmap')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(gt_slice, cmap='jet', interpolation='nearest')
    plt.colorbar()
    plt.title('Ground Truth Heatmap')
    plt.axis('off')

    # Save the plot
    plt.savefig(f'{save_name}', bbox_inches='tight', pad_inches=0)
    plt.close()

def get_points_from_heatmap_torch_old(heatmap):
    max_indices = torch.argmax(heatmap.view(heatmap.size(0), heatmap.size(1), -1), dim=2)
    z_indices = max_indices % heatmap.size(4)
    y_indices = (max_indices // heatmap.size(4)) % heatmap.size(3)
    x_indices = max_indices // (heatmap.size(3) * heatmap.size(4))
    final_points = torch.stack([x_indices, y_indices, z_indices], dim=2)
    return final_points

def distance_points(p1, p2):
    return np.sqrt(((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2))

def eval_model(model, test_loader, dataset_name, current_root):
    metric_list = ['CBD', 'CBD_2', 'BBD', 'BBD_2', 'TCD', 'TCD_2', 'FOD', 'FOD_2', 'HDV', 'HDV_2', 'ADV', 'ADV_2']
    distance_losses = [[] for _ in range(6)]
    data_idx = 0
    vis_path = os.path.join(current_root, 'image')
    heatmap_path = os.path.join(current_root, 'heatmap')
    os.makedirs(vis_path, exist_ok=True)
    os.makedirs(heatmap_path, exist_ok=True)
    for data in tqdm(test_loader):
        data_idx += 1
        voxel, gt, case_info = data
        factor = case_info['factor']
        spacing = case_info['spacing']
        voxel = voxel.cuda().unsqueeze(1).float()
        heatmap = model(voxel)
        heatmap = heatmap.detach().cpu()
        pred_3d_coor = get_points_from_heatmap_torch_old(heatmap)
        gt = gt.float().numpy()
        pred_3d_coor = pred_3d_coor.float().numpy()
        # if '1' in current_root:
        if False:
            plot_3D_points(gt.reshape((6, 3)), pred_3d_coor.reshape((6, 3)), save_name=f'{vis_path}/{dataset_name}_{data_idx}.jpg')
            plot_heatmap_2D(heatmap, gt.reshape((6, 3)), save_name=f'{heatmap_path}/{dataset_name}_{data_idx}.jpg')
        for b in range(voxel.shape[0]):
            pred_3d_coor[b] /= (factor[b].float() / spacing[b][:3])
            gt[b] /= (factor[b].float() / spacing[b][:3])
            for idx in range(pred_3d_coor.shape[1]):
                distance_losses[idx].append(distance_points(pred_3d_coor[b][idx], gt[b][idx]))
    distance_array = np.round(np.array(distance_losses), decimals=3)
    np.savetxt(f'{current_root}/{dataset_name}_test_each.csv', distance_array, delimiter=',', fmt='%.3f')

    distance_category_mean = np.round(np.mean(distance_array, axis=1), decimals=2)
    distance_all = np.round(np.mean(distance_category_mean), decimals=2).item()

    with open(f'{current_root}/{dataset_name}_test_avg.csv', 'w') as f:
        line = ','.join(map(str, distance_category_mean)) + ',' + str(distance_all)
        f.write(line)

    distance_array = distance_array.flatten()
    error_categories = np.arange(0, 6.1, 0.1)  # From 0 to 6 with 0.1 intervals
    category_rates = [round(np.sum(distance_array <= category) * 100 / len(distance_array), 4) for category in error_categories]
    with open(f'{current_root}/{dataset_name}_error_rate.csv', 'w') as json_file:
        json_file.write(','.join(map(str, category_rates)))
        print(category_rates)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='resunet', help='Model name to use (resunet, vitpose)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')

    args = parser.parse_args()
    name = args.checkpoint.split('/')[-2]
    current_root = './results/' + name + '/' + args.checkpoint[-5] + '/'
    os.makedirs(current_root, exist_ok=True)

    if 'resunet' in args.model_name:
        model = ResUnet3D(number_classes=6).cuda()
    elif 'vitpose' in args.model_name:
        model = ViTPose3D(num_classes=6).cuda()
    else:
        raise NotImplementedError

    state_dict = torch.load(args.checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    # FetalDataset
    # test_set = DGFetalMRI(dataset_type='atlas', mode='train')
    # test_loader = DataLoader(test_set, batch_size=1, num_workers=12, shuffle=False)
    # eval_model(model, test_loader, 'atlas', current_root)

    # FetalDataset
    test_set = DGFetalMRI(dataset_type='lfc', mode='train')
    test_loader = DataLoader(test_set, batch_size=1, num_workers=12, shuffle=False)
    eval_model(model, test_loader, 'FetalDataset', current_root)

    # Feta21Dataset
    test_set2 = DGFetalMRI(dataset_type='feta21')
    test_loader2 = DataLoader(test_set2, batch_size=1, num_workers=12, shuffle=False)
    eval_model(model, test_loader2, 'Feta21Dataset', current_root)