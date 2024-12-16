import argparse
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataloader_seg import BaseDataSets
from network_pretrain import ResUnet3D, ViTPose3D
from scipy.ndimage import label as nd_label

def extract_boundary_3d(seg_mask, kernel_size=3, erosion_kernel_size=2):
    if not isinstance(seg_mask, torch.Tensor):
        seg_mask = torch.from_numpy(seg_mask)

    seg_mask = (seg_mask > 0).float()
    kernel = torch.ones((1, 1, kernel_size, kernel_size, kernel_size))
    if seg_mask.is_cuda:
        kernel = kernel.cuda()

    dilated = F.conv3d(seg_mask, kernel, padding=kernel_size // 2) > 0
    eroded = F.conv3d(1.0 - seg_mask, kernel, padding=kernel_size // 2) > 0
    boundary = dilated.int() - (1 - eroded.int())

    erosion_kernel = torch.ones((1, 1, erosion_kernel_size, erosion_kernel_size, erosion_kernel_size))
    if seg_mask.is_cuda:
        erosion_kernel = erosion_kernel.cuda()

    eroded_boundary = F.conv3d(1.0 - boundary.float(), erosion_kernel, padding=erosion_kernel_size // 2) > 0
    thin_boundary = 1 - eroded_boundary.int()
    thin_boundary = thin_boundary[..., :seg_mask.shape[2], :seg_mask.shape[3], :seg_mask.shape[4]]
    return thin_boundary

def compute_distance_map(boundary, max_dist):
    dist_map = torch.full_like(boundary, max_dist)
    dist_map[boundary > 0] = 0
    kernel = torch.ones((1, 1, 3, 3, 3), device=boundary.device)

    for distance in range(1, max_dist + 1):
        dilated_boundary = F.conv3d(boundary.float(), kernel, padding=1) > 0
        update_mask = (dilated_boundary > 0) & (dist_map > distance)
        dist_map[update_mask] = distance
        boundary = dilated_boundary.float()
    dist_map = (max_dist - dist_map) / max_dist
    return dist_map

import nibabel as nib
def calculate_center(mask):
    indices = torch.nonzero(mask, as_tuple=True)
    center = torch.stack([x.float().mean() for x in indices]).round().int()
    return tuple(center.tolist())

def create_distance_map(shape, center, max_dist):
    # Generate coordinate grids for each dimension
    z, y, x = torch.meshgrid(torch.arange(shape[0], device='cuda'),
                             torch.arange(shape[1], device='cuda'),
                             torch.arange(shape[2], device='cuda'),
                             indexing='ij')
    
    # Calculate distances using broadcasting
    distances = torch.sqrt((x - center[2])**2 + (y - center[1])**2 + (z - center[0])**2)
    
    # Compute the distance map with linear decrease from center to max_dist
    dist_map = torch.clamp(max_dist - distances, min=0)
    return dist_map

def process_segmentation(gt_seg, value):
    gt_seg = gt_seg.to('cuda')
    bs, _, dz, dy, dx = gt_seg.shape
    max_distance = 5
    processed = torch.zeros_like(gt_seg, dtype=torch.float32)

    for b in range(bs):
        mask = (gt_seg[b, 0] == 8)
        if mask.any():
            center = calculate_center(mask)
            dist_map = create_distance_map((dz, dy, dx), center, max_distance)
            normalized_dist_map = (dist_map / dist_map.max()) * value
            processed[b, 0].masked_scatter_(mask, normalized_dist_map[mask])
    return processed

def save_to_nifti(tensor, filename):
    tensor = tensor.cpu().numpy()
    nifti_image = nib.Nifti1Image(tensor, affine=np.eye(4))
    nib.save(nifti_image, filename)

def sobel_edges_3d(image):
    """
    Apply a 3D Sobel filter to the input image to extract edges.

    Parameters:
    image (torch.Tensor): A 5D tensor of shape (N, C, D, H, W) where N is the batch size,
                          C is the number of channels, and D, H, W are the depth, height,
                          and width of the 3D image respectively.

    Returns:
    torch.Tensor: A 5D tensor of the same shape as the input, containing the gradient magnitude.
    """
    sobel_kernel_x = torch.tensor([[[[-1, 0, 1], [-3, 0, 3], [-1, 0, 1]], 
                                    [[-3, 0, 3], [-6, 0, 6], [-3, 0, 3]], 
                                    [[-1, 0, 1], [-3, 0, 3], [-1, 0, 1]]]], dtype=torch.float32).unsqueeze(1)

    sobel_kernel_y = torch.tensor([[[[-1, -3, -1], [0, 0, 0], [1, 3, 1]], 
                                    [[-3, -6, -3], [0, 0, 0], [3, 6, 3]], 
                                    [[-1, -3, -1], [0, 0, 0], [1, 3, 1]]]], dtype=torch.float32).unsqueeze(1)

    sobel_kernel_z = torch.tensor([[[[-1, -3, -1], [-3, -6, -3], [-1, -3, -1]], 
                                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]], 
                                    [[1, 3, 1], [3, 6, 3], [1, 3, 1]]]], dtype=torch.float32).unsqueeze(1)

    # Move the kernels to the same device as the input image
    device = image.device
    sobel_kernel_x = sobel_kernel_x.to(device)
    sobel_kernel_y = sobel_kernel_y.to(device)
    sobel_kernel_z = sobel_kernel_z.to(device)

    # Apply convolution with the Sobel kernels
    grad_x = F.conv3d(image, sobel_kernel_x, padding=1)
    grad_y = F.conv3d(image, sobel_kernel_y, padding=1)
    grad_z = F.conv3d(image, sobel_kernel_z, padding=1)

    # Compute the gradient magnitude
    grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2)

    return grad_magnitude

def random_rotate_3d(image, heatmap, degree=5):
    B, C, D, H, W = image.shape
    angles = np.radians(np.random.uniform(-degree, degree, size=3))
    cos_vals, sin_vals = np.cos(angles), np.sin(angles)
    Rx = torch.tensor([[1, 0, 0, 0], [0, cos_vals[0], sin_vals[0], 0], [0, -sin_vals[0], cos_vals[0], 0], [0, 0, 0, 1]], dtype=torch.float32, device=image.device)
    Ry = torch.tensor([[cos_vals[1], 0, -sin_vals[1], 0], [0, 1, 0, 0], [sin_vals[1], 0, cos_vals[1], 0], [0, 0, 0, 1]], dtype=torch.float32, device=image.device)
    Rz = torch.tensor([[cos_vals[2], sin_vals[2], 0, 0], [-sin_vals[2], cos_vals[2], 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32, device=image.device)
    R = Rx @ Ry @ Rz
    R = R[:3, :4].unsqueeze(0).repeat(B, 1, 1)
    grid = F.affine_grid(R, size=image.size(), align_corners=False)
    rotated_image = F.grid_sample(image, grid, align_corners=False)
    rotated_heatmap = F.grid_sample(heatmap, grid, align_corners=False)
    return rotated_image, rotated_heatmap

def log_cosh_loss(y_pred, y_true):
    diff = y_pred - y_true
    abs_diff = torch.abs(diff)
    small_diff = abs_diff < 10
    big_diff = abs_diff >= 10

    loss_small = torch.log(torch.cosh(diff[small_diff]))
    loss_big = abs_diff[big_diff] - torch.log(torch.tensor(2.0))
    loss = torch.cat([loss_small, loss_big])
    return torch.mean(loss)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def train(args):
    setup_seed(1234)

    train_seg = BaseDataSets(data_dir=args.root_data, mode="train", list_name=args.train_data_s, patch_size=args.patch_size, transform=None)
    train_loader_seg = DataLoader(train_seg, batch_size=args.batch_size, shuffle=True, num_workers=6, pin_memory=True)

    print('Data have been loaded.')

    if args.model_name == 'resunet':
        model = ResUnet3D(number_classes=args.num_classes).cuda()
    elif args.model_name == 'vitpose':
        model = ViTPose3D(num_classes=args.num_classes).cuda()
    else:
        raise NotImplementedError

    optimizer = torch.optim.Adam(model.parameters())
    local_best = 10000

    if args.max_dist > 0:
        dist_tag = f'dist{args.max_dist}'
    elif args.max_dist == 0:
        dist_tag = 'segdg'
    elif args.max_dist == -1:
        dist_tag = 'edgedg'
    else:
        raise ValueError("Invalid max_dist value")
    root_path = f'runs/{args.model_name}_{dist_tag}{args.task}_{args.max_heatmap}'
    os.makedirs(root_path, exist_ok=True)
    log_file = open(root_path+'/log.txt', 'w')

    for epoch in tqdm(range(1, args.max_epoch+1)):
        print('epoch:', epoch)
        total_train_loss = 0
        model.train()
        for i_batch, sampled_batch in enumerate(train_loader_seg):
            data_seg = sampled_batch
            if args.task == 'decay':
                voxel, gt_seg = data_seg['image'].cuda(), data_seg['mask'].cuda()
                landmark_heatmap = process_segmentation(gt_seg, args.max_heatmap)
                save_to_nifti(landmark_heatmap[0, 0], 'processed_segmentation.nii.gz')
            else:
                voxel, gt_seg = data_seg['image'].cuda(), data_seg['mask'].cuda()
            gt_seg_v2 = gt_seg.clone()

            gt_seg[gt_seg == 8] = 5
            gt_seg[gt_seg != 5] = 0

            if args.max_dist > 0:
                gt_seg_v2[gt_seg != 8] == 0
                boundary_v2 = extract_boundary_3d(gt_seg_v2)
                dist_map_nodiff_v2 = compute_distance_map(boundary_v2, args.max_dist).cuda().squeeze(0).squeeze(0) * args.max_heatmap
                boundary = extract_boundary_3d(gt_seg)
                dist_map_nodiff = compute_distance_map(boundary, args.max_dist).cuda().squeeze(0).squeeze(0) * args.max_heatmap
                heat_map = dist_map_nodiff
            elif args.max_dist == 0:
                heat_map = gt_seg.float()
            elif args.max_dist == -1:
                voxel = sobel_edges_3d(voxel)
                heat_map = gt_seg.float()

            if args.aug:
                voxel, heat_map = random_rotate_3d(voxel, heat_map, degree=15)
            heat_map = heat_map.float()
            out_seg = model(voxel)
            if args.task == 'robust':
                factor = (epoch+1)/(args.max_epoch+1)
                loss = factor * F.mse_loss(out_seg, heat_map.float()) + (1 - factor) * log_cosh_loss(out_seg, heat_map.float())
            elif args.task == 'mt':
                loss = F.mse_loss(out_seg, gt_seg.float()) # F.mse_loss(out_seg, heat_map.float())+
            elif args.task == 'logcosh':
                loss_cosh = torch.mean(torch.log(torch.cosh((out_seg.flatten()-heat_map.repeat(1,6,1,1,1).flatten()))))
                loss_mse = F.mse_loss(out_seg, heat_map.float())
                loss = loss_mse + loss_cosh
                print('logcosh', loss_cosh)
                print('mse', loss_mse)
            elif args.task == 'plogcosh':
                loss =  F.mse_loss(out_seg, heat_map.float()) # log_cosh_loss(out_seg.flatten(), heat_map.repeat(1,6,1,1,1).flatten()) +
            elif args.task == 'l1':
                loss = F.l1_loss(out_seg, heat_map.float())
            elif args.task == 'huber':
                loss = F.smooth_l1_loss(out_seg, heat_map.float())
            else:
                loss = F.mse_loss(out_seg, heat_map.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_train_loss += loss.item()

        feature_norm = torch.norm(next(model.parameters())).item()
        valid_loss = "{:.3f}".format(loss.item())
        feature_norm = "{:.3f}".format(feature_norm)
        log_file.write(valid_loss + ',' + feature_norm + '\n')

        if loss.item() < local_best:
            weights_path = root_path + f'/{epoch}-loss{valid_loss}-fn{feature_norm}.pth'
            torch.save(model.state_dict(), weights_path)
            torch.save(model.state_dict(), root_path+'/best_pretrain.pth')
            local_best = loss.item()

    return 'finish'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_data', type=str, default='./dataset', help='Name of Experiment')
    parser.add_argument('--train_data_s', type=str, default='train.list', help='Name of Dataset')
    parser.add_argument('--test_data', type=str, default='test.list', help='Name of Dataset')
    parser.add_argument('--num_classes', type=int, default=6, help='output channel of network')
    parser.add_argument('--max_epoch', type=int, default=50, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
    parser.add_argument('--patch_size', type=list, default=[128, 128, 128], help='patch size of network input')
    parser.add_argument('--base_lr', type=float, default=0.0001, help='segmentation network learning rate')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--model_name', type=str, default='resunet', help='Model name to use (resunet, vitpose)')
    parser.add_argument('--max_dist', type=int, default=5, help='Maximum distance for distance map computation')
    parser.add_argument('--max_heatmap', type=int, default=255, help='Maximum value for Gaussian heatmap computation')
    parser.add_argument('--task', type=str, default='', help='Task learning design, decay')
    parser.add_argument('--aug', type=bool, default=True, help='Task learning design, decay')

    args = parser.parse_args()
    train(args)
