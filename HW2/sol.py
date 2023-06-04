import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import timeit as time
from compute_disp import computeDisp
import cv2.ximgproc as xip

WINDOW_SIZE = 9
WINDOW_SIZE_X = 9
WINDOW_SIZE_Y = 7
BASELINE = 0.1

MAX_DISP = None
K = None
FOCAL_LENGTH = None

R = np.array([[1.0, 0.0, 0.0], 
              [0.0, 1.0, 0.0], 
              [0.0, 0.0, 1.0]])
t = np.array([[0.0], 
              [0.0], 
              [0.0]])

def read_additional_data(dir):
    global MAX_DISP, K, FOCAL_LENGTH
    with open(dir / 'max_disp.txt', 'r') as f:
        MAX_DISP = int(f.read())

    with open(dir / 'K.txt', 'r') as f:
        K = np.array([[float(num) for num in line.split()] for line in f])

    FOCAL_LENGTH = K[0, 0]

def load_images(dir):
    left = cv.imread(str(dir / 'im_left.jpg'), cv.IMREAD_GRAYSCALE)
    right = cv.imread(str(dir / 'im_right.jpg'), cv.IMREAD_GRAYSCALE)
    return left, right

def save_disparity_results(dir, left_disparity, right_disparity):
    np.savetxt(str(dir / 'disp_left.txt'), left_disparity, delimiter=',', fmt='%d', newline='\n')
    np.savetxt(str(dir / 'disp_right.txt'), right_disparity, delimiter=',', fmt='%d', newline='\n')
    # normalize disparity map for visualization
    normalized_left_disparity = cv.normalize(left_disparity, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    normalized_right_disparity = cv.normalize(right_disparity, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    cv.imwrite(str(dir / 'disp_left.jpg'), normalized_left_disparity)
    cv.imwrite(str(dir / 'disp_right.jpg'), normalized_right_disparity)

def save_depth_results(dir, depth_left, depth_right):
    with open(dir / 'depth_left.txt', 'w') as f:
        for i in range(depth_left.shape[0]):
            for j in range(depth_left.shape[1]):
                if j == depth_left.shape[1] - 1:
                    f.write(str(depth_left[i, j]))
                else:
                    f.write(str(depth_left[i, j]) + ',')
            f.write('\n')
    with open(dir / 'depth_right.txt', 'w') as f:
        for i in range(depth_right.shape[0]):
            for j in range(depth_right.shape[1]):
                if j == depth_right.shape[1] - 1:
                    f.write(str(depth_right[i, j]))
                else:
                    f.write(str(depth_right[i, j]) + ',')
            f.write('\n')

    # # normalize depth map for visualization
    # depth_left = cv.normalize(depth_left, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    # depth_right = cv.normalize(depth_right, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    # cv.imwrite(str(dir / 'depth_left.jpg'), depth_left)
    # cv.imwrite(str(dir / 'depth_right.jpg'), depth_right)

    cv.imwrite(str(dir / 'depth_left.jpg'), (depth_left / np.max(depth_left) * 255).astype(np.uint8))
    cv.imwrite(str(dir / 'depth_right.jpg'), (depth_right / np.max(depth_right) * 255).astype(np.uint8))

def census_transform2(img):
    h, w = img.shape
    pad = WINDOW_SIZE // 2
    img_padded = np.pad(img, pad, 'constant', constant_values=0)
    
    shifts = [(i, j) for i in range(-pad, pad + 1) for j in range(-pad, pad + 1) if (i, j) != (0, 0)]
    census = np.zeros((h, w), dtype=np.uint64)

    for i, j in shifts:
        binary = (img_padded[pad + i:h + pad + i, pad + j:w + pad + j] > img).astype(np.uint64)
        census = (census << 1) | binary

    return census

def census_transform(img):
    h, w = img.shape
    img = np.pad(img, ((WINDOW_SIZE_Y//2, WINDOW_SIZE_Y//2), (WINDOW_SIZE_X//2, WINDOW_SIZE_X//2)), 'constant', constant_values=0)
    census = np.zeros((h, w), dtype=np.uint64)

    start_x = WINDOW_SIZE_X // 2
    start_y = WINDOW_SIZE_Y // 2

    for i in range(start_x, h - start_x):
        for j in range(start_y, w - start_y):
            binary = ''
            for k in range(i - start_x, i + start_x + 1):
                for l in range(j - start_y, j + start_y + 1):
                    if img[k, l] > img[i, j]:
                        binary += '1'
                    else:
                        binary += '0'
            census[i, j] = int(binary, 2)

    return census

def unpackbits(x, num_bits):
    if np.issubdtype(x.dtype, np.floating):
        raise ValueError("numpy data type needs to be int-like")
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    mask = 2**np.arange(num_bits, dtype=x.dtype).reshape([1, num_bits])
    return (x & mask).astype(bool).astype(int).reshape(xshape + [num_bits])

def hamming_distance2(census, shifted_census):
    distance = np.bitwise_xor(census, shifted_census).astype(np.uint64)
    distance = np.sum(unpackbits(distance, num_bits=64), axis=-1)
    return distance

def hamming_distance(left, right):
    h, w = right.shape
    hamming = np.zeros((h, w), dtype=np.uint16)
    for i in range(h):
        for j in range(w):
            hamming[i, j] = bin(left[i, j] ^ right[i, j]).count('1')
    return hamming

def cost_volume(left_census, right_census):
    h, w = left_census.shape
    rtl_cost_volume = np.zeros((h, w, MAX_DISP), dtype=np.float32)
    ltr_cost_volume = np.zeros((h, w, MAX_DISP), dtype=np.float32)

    for d in range(MAX_DISP):
        right_shifted = np.zeros((h, w), dtype=np.uint64)
        left_shifted = np.zeros((h, w), dtype=np.uint64)
        if d == 0:
            right_shifted = right_census
            left_shifted = left_census
        else:
            right_shifted[:, d:] = right_census[:, :-d]
            left_shifted[:, :-d] = left_census[:, d:]

        ltr_cost_volume[:, :, d] = hamming_distance2(left_census, right_shifted)
        rtl_cost_volume[:, :, d] = hamming_distance2(right_census, left_shifted)

    return ltr_cost_volume, rtl_cost_volume

def aggregate_cost_volume(cost_volume1, cost_volume2, mask_size = (9, 9), sigma = 1):
    for i in range(MAX_DISP):
        cost_volume1[:, :, i] = cv.GaussianBlur(cost_volume1[:, :, i], mask_size, sigma)
        cost_volume2[:, :, i] = cv.GaussianBlur(cost_volume2[:, :, i], mask_size, sigma)
    
    return cost_volume1, cost_volume2

def aggregate_cost_volume2(left_image, right_image, left_cost, right_cost):
    sigma_r, sigma_s = 4, 11
    wndw_size = -1 # calculate window size from spatial kernel
    left_image = np.float32(left_image)
    right_image = np.float32(right_image)
    for d in range(MAX_DISP):
        # left-to-right check
        # fill left border with border_replicate
        #l_cost = cv.copyMakeBorder(cost_volume1, 0, 0, d, 0, cv.BORDER_REPLICATE)
        left_cost[:, :, d] = xip.jointBilateralFilter(left_image, left_cost[:, :, d], wndw_size, sigma_r, sigma_s)
        # right-to-left check
        # fill right border with border_replicate
        #r_cost = cv2.copyMakeBorder(cost, 0, 0, 0, d, cv2.BORDER_REPLICATE)
        right_cost[:, :, d] = xip.jointBilateralFilter(right_image, right_cost[:, :, d], wndw_size, sigma_r, sigma_s)
    return left_cost, right_cost

def consistency_test(left_disparity, right_disparity, direction):
    h, w = left_disparity.shape
    lr_check = np.zeros((h, w), dtype=np.float32)
    x, y = np.meshgrid(range(w),range(h))
    if direction == 'left':
        r_x = (x - left_disparity) # x-DL(x,y)
        mask1 = (r_x >= 0) # coordinate should be non-negative integer
    elif direction == 'right':
        r_x = (x + left_disparity)
        mask1 = (r_x < w) # coordinate should be less than image width
    l_disp = left_disparity[mask1]
    r_disp = right_disparity[y[mask1], r_x[mask1]]
    mask2 = (l_disp == r_disp) # check if DL(x,y) = DR(x-DL(x,y)) or DR(x,y) = DL(x+DR(x,y),y)
    lr_check[y[mask1][mask2], x[mask1][mask2]] = left_disparity[mask1][mask2]
    # create mask for pixels that passed the consistency check
    consistent_mask = np.zeros((h, w), dtype=np.uint8)
    consistent_mask[y[mask1][mask2], x[mask1][mask2]] = 1
    inconsistent_mask = np.logical_not(consistent_mask)
    return lr_check, inconsistent_mask

def hole_filling(disparity, hole_mask):
    h, w = disparity.shape
    x, y = np.meshgrid(range(w),range(h))

    # Get the coordinates of inconsistent pixels
    inconsistent_y = y[hole_mask]
    inconsistent_x = x[hole_mask]

    window_size = 17

    for i in range(len(inconsistent_y)):
        y_coord = inconsistent_y[i]
        x_coord = inconsistent_x[i]

        # Calculate the window boundaries based on the window_size
        ymin = max(0, y_coord - window_size // 2)
        ymax = min(h, y_coord + window_size // 2 + 1)
        xmin = max(0, x_coord - window_size // 2)
        xmax = min(w, x_coord + window_size // 2 + 1)

        # Get the neighbors of the inconsistent pixel within the window
        neighbors = disparity[ymin:ymax, xmin:xmax]

        # Find the most common value among the neighbors
        unique, counts = np.unique(neighbors, return_counts=True)
        most_common_value = unique[np.argmax(counts)]

        # Set the inconsistent pixel to the most common value
        disparity[y_coord, x_coord] = most_common_value

    return disparity

def create_depth_map(disparity):
    h, w = disparity.shape
    depth_map = np.zeros((h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            if disparity[i, j] != 0:
                depth_map[i, j] = (BASELINE * FOCAL_LENGTH) / disparity[i, j]
            else:
                depth_map[i, j] = 0.0
    
    return depth_map

def create_depth_map2(disparity):
    non_zero_disparity = np.where(disparity != 0, disparity, 1)
    depth_map = (BASELINE * FOCAL_LENGTH) / non_zero_disparity
    #depth_map[disparity == 0] = 0.0
    return depth_map

def create_3d_points(depth_map):
    h, w = depth_map.shape
    points_3d = np.zeros((h, w, 3), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            depth_value = depth_map[y, x]

            # Convert pixel coordinates to homogeneous coordinates
            homogeneous_point = np.array([x, y, 1])

            # calculate the 3d point in camera coordinates
            point = depth_value * np.matmul(np.linalg.inv(K), homogeneous_point)

            # append the 3d point to the list of points
            points_3d[y, x, :] = point

    return points_3d

def synthesize_image(i, left, pixel_3d_points, set):
    h, w, _ = left.shape
    t[0, 0] = -i * 0.01
    extrinsic_matrix_shifted = np.hstack((R, t))
    P = np.dot(K, extrinsic_matrix_shifted)
    # print(extrinsic_matrix_shifted)
    
    synth_img = np.zeros_like(left)

    for v in range(h):
        for u in range(w):
            pixel_3d_cam_shifted = pixel_3d_points[v, u, :]
            # Project 3D point onto the camera
            pixel_3d_cam_shifted_homogeneous = np.hstack((pixel_3d_cam_shifted, 1))
            pixel_2d_cam_shifted = np.dot(P, pixel_3d_cam_shifted_homogeneous.T)
            
            if pixel_2d_cam_shifted[2] != 0:
                pixel_2d_cam_shifted = pixel_2d_cam_shifted / pixel_2d_cam_shifted[2]
                point_2d = np.round(pixel_2d_cam_shifted[:2]).astype(int)

                if 0 <= point_2d[1] < h and 0 <= point_2d[0] < w:
                    # synth_img[v, u, :] = left[point_2d[1], point_2d[0], :]
                    synth_img[point_2d[1], point_2d[0], :] = left[v, u, :]
                
    return synth_img
    
def arg_parser():
    parser = argparse.ArgumentParser(description="HW2 of Computer Vision 2023")
    parser.add_argument("-p", "--path", type=str, default=None, help="Path to the direcroty of a specific set. Default is all sets.")
    parser.add_argument("-d", "--debug", action="store_true", default=False, help="Plot extra useful images.")
    return parser.parse_args()

def solve_set(set, dir, args):
    print(f"Solving for set: {str(set)}")

    left, right = load_images(dir)
    read_additional_data(dir)
    # calculate census transform
    left_census = census_transform(left)
    right_census = census_transform(right)

    if args.debug:
        print("finished census transform")
        os.makedirs(f'debug/{set}', exist_ok=True)
        cv.imwrite(f'debug/{set}/left_census.jpg', left_census)
        cv.imwrite(f'debug/{set}/right_census.jpg', right_census)
    
    # calculate cost volumes
    leftToRight_cost_vol, rightToLeft_cost_vol = cost_volume(left_census, right_census)

    if args.debug:
        print("finished cost volume")
        # normalize cost volume for visualization
        normalized_ltr_cost_vol = cv.normalize(leftToRight_cost_vol[:, :, MAX_DISP-1], None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        normalized_rtl_cost_vol = cv.normalize(rightToLeft_cost_vol[:, :, MAX_DISP-1], None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        # save cost volume
        cv.imwrite(f'debug/{set}/leftToRight_cost_vol_{MAX_DISP-1}.jpg', normalized_ltr_cost_vol)
        cv.imwrite(f'debug/{set}/rightToLeft_cost_vol_{MAX_DISP-1}.jpg', normalized_rtl_cost_vol)

    # aggregate cost volume
    #leftToRight_aggregated_cost_vol, rightToLeft_aggregated_cost_vol = aggregate_cost_volume(leftToRight_cost_vol, rightToLeft_cost_vol)
    leftToRight_aggregated_cost_vol, rightToLeft_aggregated_cost_vol = aggregate_cost_volume2(left, right, leftToRight_cost_vol, rightToLeft_cost_vol)
    # find minimum cost
    leftToRight_disparity = np.argmin(leftToRight_aggregated_cost_vol, axis=2)
    rightToLeft_disparity = np.argmin(rightToLeft_aggregated_cost_vol, axis=2)
    # leftToRight_disparity = find_minimum_cost(leftToRight_aggregated_cost_vol)
    # rightToLeft_disparity = find_minimum_cost(rightToLeft_aggregated_cost_vol)

    print("finished calculating disparity map")
    if args.debug:
        # normalize disparity map for visualization
        normalized_ltr_disparity = cv.normalize(leftToRight_disparity, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        normalized_rtl_disparity = cv.normalize(rightToLeft_disparity, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        # save disparity map
        cv.imwrite(f'debug/{set}/left_disp_not_consistent.jpg', normalized_ltr_disparity)
        cv.imwrite(f'debug/{set}/right_disp_not_consistent.jpg', normalized_rtl_disparity)

    # filter with consistency test
    #leftToRight_disparity, rightToLeft_disparity = consistency_test(leftToRight_disparity, rightToLeft_disparity)
    leftToRight_disparity, left_inconsistent_mask = consistency_test(leftToRight_disparity, rightToLeft_disparity, direction='left')
    rightToLeft_disparity, right_inconsistent_mask = consistency_test(rightToLeft_disparity, leftToRight_disparity, direction='right')
    print("finished consistency test")

    # create depth map
    depth_left = create_depth_map(leftToRight_disparity)
    depth_right = create_depth_map(rightToLeft_disparity)

    print("finished calculating depth map")

    # save results
    results_dir = Path("results", set)
    os.makedirs(results_dir, exist_ok=True)
    save_disparity_results(results_dir, leftToRight_disparity, rightToLeft_disparity)
    save_depth_results(results_dir, depth_left, depth_right)

    # calculate reprojection using depth map D and intrinsics matrix K
    left_colored = cv.imread(f'{dir}/im_left.jpg', cv.IMREAD_COLOR)

    pixel_3d_points = create_3d_points(depth_left)

    for i in range(0,12):
        synth_img = synthesize_image(i, left_colored, pixel_3d_points, set)
        # Save the synthesized image
        cv.imwrite(str(results_dir / f"synth_{str(i).zfill(2)}.jpg"), synth_img)
        print("Finished synthesizing synth_{}.jpg".format(i))

    # bonus
    print("Calculating bonus")
    # hole filling
    bonus_ltr_disparity = hole_filling(leftToRight_disparity, left_inconsistent_mask)
    bonus_rtl_disparity = hole_filling(rightToLeft_disparity, right_inconsistent_mask)

    # create depth map
    bonus_depth_left = create_depth_map(bonus_ltr_disparity)
    bonus_depth_right = create_depth_map(bonus_rtl_disparity)

    # save results
    bonus_results_dir = results_dir / "bonus"
    os.makedirs(bonus_results_dir, exist_ok=True)
    save_disparity_results(bonus_results_dir, bonus_ltr_disparity, bonus_rtl_disparity)
    save_depth_results(bonus_results_dir, bonus_depth_left, bonus_depth_right)

    # calculate reprojections
    bonus_pixel_3d_points = create_3d_points(bonus_depth_left)

    for i in range(0,12):
        synth_img = synthesize_image(i, left_colored, bonus_pixel_3d_points, set)
        # Save the synthesized image
        cv.imwrite(str(bonus_results_dir / f"synth_{str(i).zfill(2)}.jpg"), synth_img)
        print("Finished synthesizing synth_{}.jpg".format(i))


def main():
    args = arg_parser()
    sets = []
    if args.path is None:
        # solve all sets
        for path in os.listdir("data"):
            path = os.path.join("data", path)
            if os.path.isdir(path):
                sets.append(path)
    else:
        # solve only the given set
        if os.path.isdir(args.path):
            sets.append(args.path)
    for path in sets:
        set = os.path.basename(path)
        dir = Path(path)
        # time the execution
        start = time.default_timer()
        solve_set(set, dir, args)
        stop = time.default_timer()
        print('Time: ', stop - start)

    
"""
# _________________________________________________________________

    # h, w = depth_left.shape
    # reprojection_left = np.zeros((h, w, 3), dtype=np.float32)

    # left = cv.imread(str(dir / 'im_left.jpg'), cv.IMREAD_COLOR)

    # reprojection_points_left = np.zeros((h, w, 3), dtype=np.float32)
    # for y in range(h):
    #     for x in range(w):
    #         depth_value_left = depth_left[y, x]

    #         # Convert pixel coordinates to homogeneous coordinates
    #         homogeneous_point = np.array([x, y, 1])

    #         # calculate the 3d point in camera coordinates
    #         point_left = depth_value_left * np.matmul(np.linalg.inv(K), homogeneous_point)

    #         # append the 3d point to the list of points
    #         reprojection_points_left[y, x, :] = point_left

    # # synthesize new images by projecting the 3d points back to the image plane 
    # # using the camera matrix K and the rotation matrix R
    # # translate each time 1 cm along the x-axis
    # # also get the RGB values from the original image and assign them to the new image

    # for i in range(12):
    #     t[0, 0] = i * 0.01
    #     P = np.matmul(K, np.hstack((R, t)))
    #     for y in range(h):
    #         for x in range(w):
    #             point_left = reprojection_points_left[y, x, :]
    #             # point_left = np.hstack((point_left, 1))
    #             # point_left = point_left.reshape((4, 1))
    #             point_left = point_left.reshape((3, 1))
    #             point_left = np.matmul(R, point_left) + t
    #             point_left = np.matmul(K, point_left)
    #             # point_left = np.matmul(P, point_left)
    #             if point_left[2, 0] != 0:
    #                 point_left = point_left / point_left[2, 0]
    #                 x_new = int(point_left[0, 0])
    #                 y_new = int(point_left[1, 0])
    #                 if x_new >= 0 and x_new < w and y_new >= 0 and y_new < h:
    #                     reprojection_left[y_new, x_new, :] = left[y, x, :]
    #                     # reprojection_left[y, x, :] = left[y_new, x_new, :]
    #     print("Finished synthesizing synth_{}.jpg".format(str(i).zfill(2)))
    #     cv.imwrite(f'results/{set}/synth_' + str(i).zfill(2) + '.jpg', reprojection_left)
"""

if __name__ == "__main__":
    # time the execution
    start = time.default_timer()
    main()
    stop = time.default_timer()
    print('Total time: ', stop - start)
