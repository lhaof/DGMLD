import json
import os.path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def crop_image(file_path, left, top, right, bottom):
    with Image.open(file_path) as img:
        width, height = img.size
        # Define the box to crop: (left, upper, right, lower)
        crop_box = (left, top, width - right, height - bottom)
        cropped_img = img.crop(crop_box)
        cropped_img.save(file_path)  # Save the cropped image

def plot_3D_points(gt_points, pred_points, labels=["TCD-1", "TCD-2", "HDV-1", "HDV-2", "ADV-1", "ADV-2"],
                   save_name="3D_plot.png", size=300, font_size=18):
    """
    Function to plot two sets of points (gt_points and pred_points)
    in a 3D space. The ground truth points (gt_points)
    are plotted in green and the predicted points (pred_points) are plotted in red.
    """

    # convert points to integer indices
    gt_points = np.rint(gt_points).astype(int)
    pred_points = np.rint(pred_points).astype(int)

    fig = plt.figure(figsize=(9, 8))  # Increase figure width to give more space for z-axis label
    ax = fig.add_subplot(111, projection='3d')

    # Define different markers for different labels
    marker_dict = {
        'TCD': 'o',
        'HDV': '^',
        'ADV': 's'
    }

    for i, label in enumerate(labels):
        marker = marker_dict.get(label.strip()[0:3], 'o')  # default to 'o' if label not found

        xs, ys, zs = pred_points[i]
        ax.scatter(xs, ys, zs, c='r', marker=marker, s=size, alpha=0.5, label='Prediction' if i == 0 else "")

        xs, ys, zs = gt_points[i]
        ax.scatter(xs, ys, zs, c='g', marker=marker, s=size, alpha=0.5, label='Ground Truth' if i == 0 else "")

        ax.text(xs, ys, zs, label, color='black', fontsize=font_size)

    ax.set_xlabel('X Axis', fontsize=font_size, labelpad=10)
    ax.set_ylabel('Y Axis', fontsize=font_size, labelpad=10)
    ax.set_zlabel('Z Axis', fontsize=font_size, labelpad=10)

    ax.dist = 11  # Optionally adjust the distance

    from matplotlib.ticker import MultipleLocator

    # Set the limits of the axes
    ax.set_xlim(35, 80)
    ax.set_ylim(75, 100)
    ax.set_zlim(15, 45)

    # Set the tick label font size for the axes
    tick_label_font_size = 14  # for example, set to 10

    # Adjust the tick parameters for all three axes to change labels size
    ax.tick_params(axis='x', labelsize=tick_label_font_size)
    ax.tick_params(axis='y', labelsize=tick_label_font_size)
    ax.tick_params(axis='z', labelsize=tick_label_font_size)

    # Set the step (interval) between ticks on each axis using MultipleLocator
    x_tick_step = 5  # for example, set to 5
    y_tick_step = 5  # for example, set to 5
    z_tick_step = 5  # for example, set to 5

    # Apply the tick intervals for each axis
    ax.xaxis.set_major_locator(MultipleLocator(x_tick_step))
    ax.yaxis.set_major_locator(MultipleLocator(y_tick_step))
    ax.zaxis.set_major_locator(MultipleLocator(z_tick_step))

    # ... [rest of your plotting code, e.g., plt.show() or plt.savefig()]
    ax.view_init(elev=20, azim=-45)

    plt.legend(prop={'size': font_size})

    # Adjust the padding and layout. Increasing pad value to avoid cropping of the z-axis label
    plt.tight_layout(w_pad=5.0)

    # Save and show the plot with adjusted padding
    plt.savefig(save_name, dpi=200, bbox_extra_artists=(ax,))
    crop_image(save_name, left=170, top=50, right=0, bottom=120)
    # plt.show()


def plot_3D_points_json(origin, pred_points, json_path, current_root):
    """
    Function to plot two sets of points (gt_points and pred_points)
    in independent 2D slices. The ground truth points (gt_points)
    are plotted in green and the predicted points (pred_points) are plotted in red.
    """
    json_path = json_path[0]
    json_file = json.load(open(json_path, 'r'))

    for i in range(int(len(pred_points)/2)):
        point_1 = pred_points[i*2]
        point_2 = pred_points[i*2+1]
        json_file['markups'][0]['controlPoints'][0]['position'] = (point_1+origin).tolist()
        json_file['markups'][0]['controlPoints'][1]['position'] = (point_2+origin).tolist()
        save_path = './'+current_root+'./results_json/'+json_path.split('/')[-2] + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(save_path + str(i+1) + '.mrk.json', 'w') as f:
            f.write(json.dumps(json_file))


if __name__ == "__main__":
    # test the function
    gt_points = np.array([
        [88.1973, 82.2857, 31.4407],
        [47.4515, 82.2857, 33.6446],
        [66.0645, 84.3239, 27.5838],
        [66.0645, 80.1057, 44.3887],
        [66.0645, 89.7139, 34.1956],
        [66.0645, 78.6997, 34.7466]
    ])

    pred_points = np.array([
        [85.0645, 82.2857, 29.5838],
        [45.4515, 82.2857, 33.6446],
        [64.0645, 84.3239, 28.5838],
        [64.0645, 80.1057, 42.3887],
        [64.0645, 89.7139, 32.1956],
        [64.0645, 78.6997, 34.7466]
    ])

    plot_3D_points(gt_points, pred_points)
