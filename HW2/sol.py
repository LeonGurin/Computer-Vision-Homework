import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

WINDOW_SIZE = 5
BASELINE = 0.1

MAX_DISP = None
K = None
FOCAL_LENGTH = None

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

# def census_transform_cv(img):
#     h, w = img.shape
#     census = cv.stereo.censusTransform(img)
#     return census

def census_transform(img):
    h, w = img.shape
    half_window_size = WINDOW_SIZE // 2
    img = np.pad(img, half_window_size, 'constant', constant_values=0)
    census = np.zeros((h, w), dtype=np.uint64)


    for i in range(half_window_size, h - half_window_size):
        for j in range(half_window_size, w - half_window_size):
            binary = ''
            for k in range(i - half_window_size, i + half_window_size + 1):
                for l in range(j - half_window_size, j + half_window_size + 1):
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
    cost_volume1 = np.zeros((h, w, MAX_DISP), dtype=np.uint64)
    cost_volume2 = np.zeros((h, w, MAX_DISP), dtype=np.uint64)

    # right_shifted = right_census.copy()
    # left_shifted = left_census.copy()

    for d in range(MAX_DISP):
        for i in range(h):
            for j in range(w):
                if j - d >= 0:
                    cost_volume1[i, j, d] = bin(left_census[i, j] ^ right_census[i, j - d]).count('1')
                else:
                    cost_volume1[i, j, d] = bin(left_census[i, j] ^ right_census[i, 0]).count('1')
                if j + d < w:
                    cost_volume2[i, j, d] = bin(right_census[i, j] ^ left_census[i, j + d]).count('1')
                else:
                    cost_volume2[i, j, d] = bin(right_census[i, j] ^ left_census[i, w - 1]).count('1')

    # for i in range(MAX_DISP):
    #     right_shifted = np.roll(right_shifted, 1, axis=1)
    #     left_shifted = np.roll(left_shifted, -1, axis=1)
    #     cost_volume1[:, :, i] = hamming_distance(left_census, right_shifted)
    #     cost_volume2[:, :, i] = hamming_distance(right_census, left_shifted)

    return cost_volume1, cost_volume2

def aggregate_cost_volume(cost_volume1, cost_volume2, mask_size = (3, 3)):
    # perform uniform averaging with openCV filter
    for i in range(MAX_DISP):
        cost_volume1[:, :, i] = cv.blur(cost_volume1[:, :, i], mask_size)
        cost_volume2[:, :, i] = cv.blur(cost_volume2[:, :, i], mask_size)
    
    return cost_volume1, cost_volume2

def find_minimum_cost(aggregated_cost_volume):
    h, w, d = aggregated_cost_volume.shape
    disparity = np.zeros((h, w), dtype=np.uint64)

    for i in range(h):
        for j in range(w):
            disparity[i, j] = np.argmin(aggregated_cost_volume[i, j, :])

    with open('test.txt', 'w') as f:
        for i in range(disparity.shape[0]):
            for j in range(disparity.shape[1]):
                f.write(str(disparity[i, j]) + ',')
            f.write('\n')
        f.close()

    # disparity = np.argmin(aggregated_cost_volume, axis=2)

    return disparity

def consistency_test(left_disparity, right_disparity):
    h, w = left_disparity.shape
    threshold = 60
    
    for i in range(h):
        for j in range(w):
            mapped_val = j - left_disparity[i, j]
            if mapped_val >= 0 and mapped_val < w:
                if abs(left_disparity[i, j] - right_disparity[i, mapped_val]) > threshold:
                    left_disparity[i, j] = 0
                    right_disparity[i, mapped_val] = 0
            
            mapped_val = j - right_disparity[i, j]
            if mapped_val >= 0 and mapped_val < w:
                if abs(right_disparity[i, j] - left_disparity[i, mapped_val]) > threshold:
                    right_disparity[i, j] = 0
                    left_disparity[i, mapped_val] = 0
    
    return left_disparity, right_disparity

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

def arg_parser():
    parser = argparse.ArgumentParser(description="HW2 of Computer Vision 2023")
    parser.add_argument("-p", "--path", type=str, default=None, help="Path to the direcroty of a specific set. Default is all sets.")
    parser.add_argument("-d", "--debug", action="store_true", default=False, help="Plot extra useful images.")
    return parser.parse_args()

def main():
    args = arg_parser()
    if args.path is None:
        # to be supported
        exit(1)

    set = os.path.basename(args.path)
    dir = Path(args.path)
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

    # aggregate cost volume
    leftToRight_aggregated_cost_vol, rightToLeft_aggregated_cost_vol = aggregate_cost_volume(leftToRight_cost_vol, rightToLeft_cost_vol)

    # find minimum cost
    leftToRight_disparity = np.argmin(leftToRight_aggregated_cost_vol, axis=2)
    rightToLeft_disparity = np.argmin(rightToLeft_aggregated_cost_vol, axis=2)
    # leftToRight_disparity = find_minimum_cost(leftToRight_aggregated_cost_vol)
    # rightToLeft_disparity = find_minimum_cost(rightToLeft_aggregated_cost_vol)
    # leftToRight_disparity = find_minimum_cost(leftToRight_cost_vol)
    # rightToLeft_disparity = find_minimum_cost(rightToLeft_cost_vol)

    # filter with consistency test
    # leftToRight_disparity, rightToLeft_disparity = consistency_test(leftToRight_disparity, rightToLeft_disparity)
    
    print("finished calculating disparity map")

    # create depth map
    depth_left = create_depth_map(leftToRight_disparity)
    depth_right = create_depth_map(rightToLeft_disparity)

    print("finished calculating depth map")

    # save images
    os.makedirs(f'results/{set}', exist_ok=True)
    cv.imwrite(f'results/{set}/disp_left.jpg', leftToRight_disparity)
    cv.imwrite(f'results/{set}/disp_right.jpg', rightToLeft_disparity)
    cv.imwrite(f'results/{set}/depth_left.jpg', depth_left)
    cv.imwrite(f'results/{set}/depth_right.jpg', depth_right)

    np.savetxt(f'results/{set}/disp_left.txt', leftToRight_disparity, delimiter=',')
    np.savetxt(f'results/{set}/disp_right.txt', rightToLeft_disparity, delimiter=',')
    np.savetxt(f'results/{set}/depth_left.txt', depth_left, delimiter=',')
    np.savetxt(f'results/{set}/depth_right.txt', depth_right, delimiter=',')

    # calculate reprojection using depth map D and intrinsics matrix K
    
    R = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    t = np.array([[0.0], [0.0], [0.0]])

    h, w = depth_left.shape
    reprojection_left = np.zeros((h, w, 3), dtype=np.float32)

    reprojection_points_left = np.zeros((h, w, 3), dtype=np.float32)

    left = cv.imread(str(dir / 'im_left.jpg'), cv.IMREAD_COLOR)

    for y in range(h):
        for x in range(w):
            depth_value_left = depth_left[y, x]

            # Convert pixel coordinates to homogeneous coordinates
            homogeneous_point = np.array([x, y, 1])

            # calculate the 3d point in camera coordinates
            point_left = depth_value_left * np.matmul(np.linalg.inv(K), homogeneous_point)

            # append the 3d point to the list of points
            reprojection_points_left[y, x, :] = point_left

    # synthesize new images by projecting the 3d points back to the image plane 
    # using the camera matrix K and the rotation matrix R
    # translate each time 1 cm along the x-axis
    # also get the RGB values from the original image and assign them to the new image

    for i in range(12):
        t[0, 0] = i * 0.01
        for y in range(h):
            for x in range(w):
                point_left = reprojection_points_left[y, x, :]
                point_left = point_left.reshape((3, 1))
                point_left = np.matmul(R, point_left) + t
                point_left = np.matmul(K, point_left)
                point_left = point_left / point_left[2, 0]
                try:
                    x_new = int(point_left[0, 0])
                    y_new = int(point_left[1, 0])
                except:
                    x_new = 0
                    y_new = 0
                if x_new >= 0 and x_new < w and y_new >= 0 and y_new < h:
                    reprojection_left[y_new, x_new, :] = left[y, x, :]

        cv.imwrite(f'results/{set}/synth_' + str(i) + '.png', reprojection_left)
    
if __name__ == "__main__":
    main()
