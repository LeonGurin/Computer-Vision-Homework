import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import timeit as time
import cv2.ximgproc as xip

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
    depth_left = depth_left / np.min(np.max(depth_left), 10) * 255
    depth_left[depth_left > 255] = 255
    depth_right = depth_right / np.min(np.max(depth_right), 10) * 255
    depth_right[depth_right > 255] = 255
    cv.imwrite(str(dir / 'depth_left.jpg'), (depth_left).astype(np.uint8))
    cv.imwrite(str(dir / 'depth_right.jpg'), (depth_right).astype(np.uint8))

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

        ltr_cost_volume[:, :, d] = hamming_distance(left_census, right_shifted)
        rtl_cost_volume[:, :, d] = hamming_distance(right_census, left_shifted)

    return ltr_cost_volume, rtl_cost_volume

def aggregate_cost_volume(left_image, right_image, left_cost, right_cost):
    sigma_r, sigma_s = 4, 11
    wndw_size = -1 # calculate window size from spatial kernel
    left_image = np.float32(left_image)
    right_image = np.float32(right_image)
    for d in range(MAX_DISP):
        # a joint bilateral filter is applied to each disparity level
        left_cost[:, :, d] = xip.jointBilateralFilter(left_image, left_cost[:, :, d], wndw_size, sigma_r, sigma_s)
        right_cost[:, :, d] = xip.jointBilateralFilter(right_image, right_cost[:, :, d], wndw_size, sigma_r, sigma_s)
    return left_cost, right_cost

def consistency_test(left_disparity, right_disparity, direction, threshold=1):
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
    # mask2 = (l_disp == r_disp) # check if DL(x,y) = DR(x-DL(x,y)) or DR(x,y) = DL(x+DR(x,y),y)
    mask2 = (np.abs(l_disp - r_disp) < threshold)
    lr_check[y[mask1][mask2], x[mask1][mask2]] = left_disparity[mask1][mask2]
    # create mask for pixels that passed the consistency check
    consistent_mask = np.zeros((h, w), dtype=np.uint8)
    consistent_mask[y[mask1][mask2], x[mask1][mask2]] = 1
    inconsistent_mask = np.logical_not(consistent_mask)
    return lr_check, inconsistent_mask

def hole_filling(image, disparity, hole_mask):
    h, w = disparity.shape
    # pad maximum disparity for the holes in boundary
    lr_check_pad = cv.copyMakeBorder(disparity, 0,0,1,1, cv.BORDER_CONSTANT, value=MAX_DISP)
    l_labels = np.zeros((h, w), dtype=np.float32)
    r_labels = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            idx_L, idx_R = 0, 0
            # 𝐹𝐿, the disparity map filled by closest valid disparity from left
            while lr_check_pad[y, x+1-idx_L] == 0:
                idx_L += 1
            l_labels[y, x] = lr_check_pad[y, x+1-idx_L]
            # 𝐹𝑅, the disparity map filled by closest valid disparity from right
            while lr_check_pad[y, x+1+idx_R] == 0:
                idx_R += 1
            r_labels[y, x] = lr_check_pad[y, x+1+idx_R]
    # Final filled disparity map 𝐷 = min(𝐹𝐿 , 𝐹𝑅) (pixel-wise minimum)
    labels = np.min((l_labels, r_labels), axis=0)

    # weighted median filter
    WMF_r = 11
    labels = xip.weightedMedianFilter(image.astype(np.uint8), labels, WMF_r)
    return labels

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

def synthesize_image(i, left, pixel_3d_points, set, direction="left"):
    h, w, _ = left.shape
    if direction == "left":
        t[0, 0] = -i * 0.01
    else:
        t[0, 0] = i * 0.01
    extrinsic_matrix_shifted = np.hstack((R, t))
    P = np.dot(K, extrinsic_matrix_shifted)
    # print(extrinsic_matrix_shifted)
    
    synth_img = np.zeros_like(left)
    filled_mask = np.zeros((h, w), dtype=np.uint8)

    for v in range(h):
        for u in range(w):
            pixel_3d_cam_shifted = pixel_3d_points[v, u, :]
            # Project 3D point onto the camera
            pixel_3d_cam_shifted_homogeneous = np.hstack((pixel_3d_cam_shifted, 1))
            pixel_2d_cam_shifted = np.dot(P, pixel_3d_cam_shifted_homogeneous.T)
            
            if pixel_2d_cam_shifted[2] != 0:
                pixel_2d_cam_shifted = pixel_2d_cam_shifted / pixel_2d_cam_shifted[2]
                point_2d = np.round(pixel_2d_cam_shifted[:2]).astype(np.uint16)

                if 0 <= point_2d[1] < h and 0 <= point_2d[0] < w:
                    # synth_img[v, u, :] = left[point_2d[1], point_2d[0], :]
                    synth_img[point_2d[1], point_2d[0], :] = left[v, u, :]
                    filled_mask[point_2d[1], point_2d[0]] = 1
    holes_mask = np.logical_not(filled_mask)
                
    return synth_img, holes_mask
    
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
    leftToRight_aggregated_cost_vol, rightToLeft_aggregated_cost_vol = aggregate_cost_volume(left, right, leftToRight_cost_vol, rightToLeft_cost_vol)
    # find minimum cost
    leftToRight_disparity = np.argmin(leftToRight_aggregated_cost_vol, axis=2)
    rightToLeft_disparity = np.argmin(rightToLeft_aggregated_cost_vol, axis=2)


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
    right_colored = cv.imread(f'{dir}/im_right.jpg', cv.IMREAD_COLOR)

    pixel_3d_points = create_3d_points(depth_left)

    for i in range(0,11):
        synth_img, _ = synthesize_image(i, left_colored, pixel_3d_points, set)
        # Save the synthesized image
        cv.imwrite(str(results_dir / f"synth_{str(i+1).zfill(2)}.jpg"), synth_img)
        print(f"Finished synthesizing synth_{i+1}.jpg")

    # bonus
    print("Calculating bonus")
    # hole filling
    bonus_ltr_disparity = hole_filling(left, leftToRight_disparity, left_inconsistent_mask)
    bonus_rtl_disparity = hole_filling(right, rightToLeft_disparity, right_inconsistent_mask)

    # create depth map
    bonus_depth_left = create_depth_map(bonus_ltr_disparity)
    bonus_depth_right = create_depth_map(bonus_rtl_disparity)

    # save results
    bonus_results_dir = results_dir / "bonus"
    os.makedirs(bonus_results_dir, exist_ok=True)
    save_disparity_results(bonus_results_dir, bonus_ltr_disparity, bonus_rtl_disparity)
    save_depth_results(bonus_results_dir, bonus_depth_left, bonus_depth_right)

    # calculate reprojections
    bonus_left_pixel_3d_points = create_3d_points(bonus_depth_left)
    bonus_right_pixel_3d_points = create_3d_points(bonus_depth_right)

    for i in range(0,11):
        left_synth_img, left_holes_mask = synthesize_image(i, left_colored, bonus_left_pixel_3d_points, set, direction='left')
        right_synth_img, right_holes_mask = synthesize_image(10-i, right_colored, bonus_right_pixel_3d_points, set, direction='right')
        left_holes_mask_3ch = np.stack((left_holes_mask, left_holes_mask, left_holes_mask), axis=-1)
        synth_img = left_synth_img + right_synth_img * left_holes_mask_3ch
        # apply median filter on pixels that are holes
        synth_img = left_synth_img + cv.medianBlur(synth_img, 3) * left_holes_mask_3ch
        # Save the synthesized image
        cv.imwrite(str(bonus_results_dir / f"synth_{str(i+1).zfill(2)}.jpg"), synth_img)
        print(f"Finished synthesizing synth_{i+1}.jpg")
        if args.debug:
            # visualize holes mask
            cv.imwrite(f'debug/{set}/left_holes_mask_{i+1}.jpg', left_holes_mask * 255)
            cv.imwrite(f'debug/{set}/right_holes_mask_{i+1}.jpg', right_holes_mask * 255)


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

    
if __name__ == "__main__":
    # time the execution
    start = time.default_timer()
    main()
    stop = time.default_timer()
    print('Total time: ', stop - start)
