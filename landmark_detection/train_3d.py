import os.path
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader import DGFetalMRI
from network import ResUnet3D, ViTPose3D
from tqdm import tqdm

def log_cosh_loss(y_pred, y_true):
    diff = y_pred - y_true
    abs_diff = torch.abs(diff)
    small_diff = abs_diff < 10
    big_diff = abs_diff >= 10

    loss_small = torch.log(torch.cosh(diff[small_diff]))
    loss_big = abs_diff[big_diff] - torch.log(torch.tensor(2.0))
    loss = torch.cat([loss_small, loss_big])
    return torch.mean(loss)

def distance_points(p1, p2):
    return np.sqrt(((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2))

def generate_gaussian_heatmap(a_shape, points, sigma=5.0, factor=255):
    batch_size, numbers, x, y, z = a_shape
    grid_x, grid_y, grid_z = torch.meshgrid(torch.arange(x), torch.arange(y), torch.arange(z), indexing='ij')
    heatmap = torch.zeros((batch_size, numbers, x, y, z))
    for b in range(batch_size):
        for n in range(numbers):
            point = points[b, n]
            distance = (grid_x - point[0]) ** 2 + (grid_y - point[1]) ** 2 + (grid_z - point[2]) ** 2
            heatmap[b, n] = torch.exp(-distance / (2 * sigma ** 2))
    heatmap = factor * (heatmap - torch.min(heatmap)) / (torch.max(heatmap) - torch.min(heatmap))
    return heatmap

def get_points_from_heatmap_torch(heatmap):
    max_indices = torch.argmax(heatmap.view(heatmap.size(0), heatmap.size(1), -1), dim=2)
    z_indices = max_indices % heatmap.size(4)
    y_indices = (max_indices // heatmap.size(4)) % heatmap.size(3)
    x_indices = max_indices // (heatmap.size(3) * heatmap.size(4))
    final_points = torch.stack([x_indices, y_indices, z_indices], dim=2)
    return final_points

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

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

def evaluate_model(model, valid_loader):
    distance_losses = []
    model.eval()
    for data in valid_loader:
        voxel, gt, case_info = data
        factor = case_info['factor']
        spacing = case_info['spacing']
        voxel = voxel.cuda().unsqueeze(1).float()
        heatmap = model(voxel)
        pred_3d_coor = get_points_from_heatmap_torch(heatmap).float().detach().cpu().numpy()
        gt = gt.float()
        for b in range(voxel.shape[0]):
            pred_3d_coor[b] /= (factor[b].float() / spacing[b][:3])
            gt[b] /= (factor[b].float() / spacing[b][:3])
            for idx in range(pred_3d_coor.shape[1]):
                distance_losses.append(distance_points(pred_3d_coor[b][idx], gt[b][idx]))
    distance_losses = np.array(distance_losses)
    mean_distance = np.mean(distance_losses)
    return "{:.2f}".format(mean_distance)

def main(args):
    setup_seed(args.seed)

    if args.model_name == 'resunet':
        model = ResUnet3D(number_classes=args.num_classes).cuda()
    elif args.model_name == 'vitpose':
        model = ViTPose3D(num_classes=args.num_classes).cuda()
    else:
        raise NotImplementedError
    
    if args.dg_method not in ['baseline', 'bio']:
        if args.model_weights:
            checkpoint = torch.load(args.model_weights)
            model.load_state_dict(checkpoint)

    optimizer = torch.optim.Adam(model.parameters())
    
    train_set = DGFetalMRI(dataset_type='atlas', img_size=128)
    valid_set_lfc = DGFetalMRI(dataset_type='lfc', mode='valid')
    # test_set_lfc = DGFetalMRI(dataset_type='lfc', mode='train')
    test_set_feta = DGFetalMRI(dataset_type='feta21')

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=12, pin_memory=True)
    valid_loader = DataLoader(valid_set_lfc, batch_size=4, num_workers=12, shuffle=False, pin_memory=True)
    # test_loader_lfc = DataLoader(test_set_lfc, batch_size=1, num_workers=6, shuffle=False)
    test_loader_feta = DataLoader(test_set_feta, batch_size=4, num_workers=12, shuffle=False, pin_memory=True)

    print('Data have been loaded.')
    if args.dg_method == 'logcosh':
        test_save_path = f'runs/{args.model_name}{args.dg_method}'
    elif args.model_weights:
        name = args.model_weights.split('/')[-2]
        test_save_path = f'runs/{name}'
    else:
        test_save_path = f'runs/{args.model_name}{args.dg_method}'
    os.makedirs(test_save_path, exist_ok=True)
    f = open(f'{test_save_path}/log.txt', 'w')
    local_best = 100
    for epoch in tqdm(range(1, 51)):
        total_train_loss = 0
        model.train()
        for data in train_loader:
            voxel, gt, case_info = data
            voxel = voxel.cuda().unsqueeze(1) 
            
            gt_heatmap = generate_gaussian_heatmap((voxel.shape[0], 6, 128, 128, 128), gt, factor=255).cuda()
            if args.dg_method == 'bio':
                voxel, gt_heatmap = random_rotate_3d(voxel, gt_heatmap, degree=15)
            else:
                voxel, gt_heatmap = random_rotate_3d(voxel, gt_heatmap, degree=5)
            out = model(voxel)
            loss = F.mse_loss(out, gt_heatmap)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_train_loss += loss.item()
        valid_mre = evaluate_model(model, valid_loader)
        f.write(f'epoch: {epoch}, {round(total_train_loss / (len(train_loader) * 4), 3)}, {valid_mre} \n')
        print(f'epoch: {epoch}, {round(total_train_loss / (len(train_loader) * 4), 3)}, {valid_mre} \n')
        if local_best > float(valid_mre):
            local_best = float(valid_mre)
            torch.save(model.state_dict(), test_save_path+'/best_model'+str(args.seed)+'.pth')
            if args.test:
                # test_mre_lfc = evaluate_model(model, test_loader_lfc)
                test_mre_feta = evaluate_model(model, test_loader_feta)
                weights_path = f'{test_save_path}/{epoch}-valid-{valid_mre}-feta21-{test_mre_feta}.pth'
                torch.save(model.state_dict(), weights_path)
            else:
                if epoch > 35:
                    weights_path = f'{test_save_path}/{epoch}-valid-{valid_mre}.pth'
                    torch.save(model.state_dict(), weights_path)

    print(f'Best validation loss: {local_best}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='Model name to use (resunet, vitpose)')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--test', type=bool, default=True, help='Random seed')
    parser.add_argument('--model_weights', type=str, default=None, help='Path to model weights')
    parser.add_argument('--num_classes', type=int, default=6, help='Output channel of network')
    parser.add_argument('--dg_method', type=str, required=False, choices=['segdg', 'edgedg', 'dist', 'bio', 'baseline', 'logcosh'], help='Domain generalization method to use')
    args = parser.parse_args()
    print(args)
    main(args)
