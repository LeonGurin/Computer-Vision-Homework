import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import timeit as time

WINDOW_SIZE = 9
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

# def hamming_distance2(census, shifted_census):
#     distance = np.bitwise_xor(census, shifted_census).astype(np.uint8)
#     distance = np.sum(np.unpackbits(distance, axis=-1), axis=-1)
#     return distance

def hamming_distance(left, right):
    h, w = right.shape
    hamming = np.zeros((h, w), dtype=np.uint16)
    for i in range(h):
        for j in range(w):
            hamming[i, j] = bin(left[i, j] ^ right[i, j]).count('1')
    return hamming

def cost_volume(left_census, right_census):
    h, w = left_census.shape
    rtl_cost_volume = np.zeros((h, w, MAX_DISP), dtype=np.uint64)
    ltr_cost_volume = np.zeros((h, w, MAX_DISP), dtype=np.uint64)

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

def aggregate_cost_volume(cost_volume1, cost_volume2, mask_size = (3, 3), sigma = 1):
    for i in range(MAX_DISP):
        cost_volume1[:, :, i] = cv.GaussianBlur(cost_volume1[:, :, i], mask_size, sigma)
        cost_volume2[:, :, i] = cv.GaussianBlur(cost_volume2[:, :, i], mask_size, sigma)
    
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

def consistency_test(left_disparity, right_disparity, threshold = 10):
    h, w = left_disparity.shape
    original_left = left_disparity.copy()
    original_right = right_disparity.copy()
    left_consistency_mask = np.zeros((h, w), dtype=np.uint8)
    right_consistency_mask = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            mapped_val = j - original_left[i, j]
            if mapped_val >= 0 and mapped_val < w:
                if abs(original_left[i, j] - original_right[i, mapped_val]) > threshold:
                    left_disparity[i, j] = 0
                    right_disparity[i, mapped_val] = 0
            
            mapped_val = j + original_right[i, j]
            if mapped_val >= 0 and mapped_val < w:
                if abs(original_right[i, j] - original_left[i, mapped_val]) > threshold:
                    right_disparity[i, j] = 0
                    left_disparity[i, mapped_val] = 0
    
    return left_disparity, right_disparity

def consistency_test2(left_disparity, right_disparity, direction):
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
    return lr_check

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
    # depth_map[disparity == 0] = 0.0
    return depth_map

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
                
    # Save the resulting image with the corresponding shift index
    cv.imwrite(f"results/{set}/synth_{str(i).zfill(2)}.jpg", synth_img)
    print("Finished synthesizing synth_{}.jpg".format(i))
    
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
    left_census = census_transform2(left)
    right_census = census_transform2(right)

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
        leftToRight_cost_vol = cv.normalize(leftToRight_cost_vol, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        rightToLeft_cost_vol = cv.normalize(rightToLeft_cost_vol, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        # save cost volume
        cv.imwrite(f'debug/{set}/leftToRight_cost_vol_{MAX_DISP-1}.jpg', leftToRight_cost_vol[:, :, MAX_DISP-1])
        cv.imwrite(f'debug/{set}/rightToLeft_cost_vol_{MAX_DISP-1}.jpg', rightToLeft_cost_vol[:, :, MAX_DISP-1])

    # aggregate cost volume
    leftToRight_aggregated_cost_vol, rightToLeft_aggregated_cost_vol = aggregate_cost_volume(leftToRight_cost_vol, rightToLeft_cost_vol)

    # find minimum cost
    leftToRight_disparity = np.argmin(leftToRight_aggregated_cost_vol, axis=2)
    rightToLeft_disparity = np.argmin(rightToLeft_aggregated_cost_vol, axis=2)
    # leftToRight_disparity = find_minimum_cost(leftToRight_aggregated_cost_vol)
    # rightToLeft_disparity = find_minimum_cost(rightToLeft_aggregated_cost_vol)

    print("finished calculating disparity map")

    # filter with consistency test
    #leftToRight_disparity, rightToLeft_disparity = consistency_test(leftToRight_disparity, rightToLeft_disparity)
    leftToRight_disparity = consistency_test2(leftToRight_disparity, rightToLeft_disparity, direction='left')
    rightToLeft_disparity = consistency_test2(rightToLeft_disparity, leftToRight_disparity, direction='right')
    print("finished consistency test")
    
    # create depth map
    depth_left = create_depth_map(leftToRight_disparity)
    depth_right = create_depth_map(rightToLeft_disparity)

    print("finished calculating depth map")

    np.savetxt(f'results/{set}/disp_left.txt', leftToRight_disparity, delimiter=',', fmt='%d', newline='\n')
    np.savetxt(f'results/{set}/disp_right.txt', rightToLeft_disparity, delimiter=',', fmt='%d', newline='\n')
    with open(f'results/{set}/depth_left.txt', 'w') as f:
        for i in range(depth_left.shape[0]):
            for j in range(depth_left.shape[1]):
                if j == depth_left.shape[1] - 1:
                    f.write(str(depth_left[i, j]))
                else:
                    f.write(str(depth_left[i, j]) + ',')
            f.write('\n')
        f.close()
    with open(f'results/{set}/depth_right.txt', 'w') as f:
        for i in range(depth_right.shape[0]):
            for j in range(depth_right.shape[1]):
                if j == depth_right.shape[1] - 1:
                    f.write(str(depth_right[i, j]))
                else:
                    f.write(str(depth_right[i, j]) + ',')
            f.write('\n')
        f.close()

    # normalize disparity map for visualization
    leftToRight_disparity = cv.normalize(leftToRight_disparity, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    rightToLeft_disparity = cv.normalize(rightToLeft_disparity, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    # normalize depth map for visualization
    depth_left = cv.normalize(depth_left, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    depth_right = cv.normalize(depth_right, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

    # save images
    os.makedirs(f'results/{set}', exist_ok=True)
    cv.imwrite(f'results/{set}/disp_left.jpg', leftToRight_disparity)
    cv.imwrite(f'results/{set}/disp_right.jpg', rightToLeft_disparity)
    cv.imwrite(f'results/{set}/depth_left.jpg', depth_left)
    cv.imwrite(f'results/{set}/depth_right.jpg', depth_right)

    # np.savetxt(f'results/{set}/depth_left.txt', depth_left, delimiter=',', fmt='%f', newline='\n')
    # np.savetxt(f'results/{set}/depth_right.txt', depth_right, delimiter=',', fmt='%f', newline='\n')

    # calculate reprojection using depth map D and intrinsics matrix K
    left_colored = cv.imread(f'{set}/im_left.jpg', cv.IMREAD_COLOR)
    h, w = left_colored.shape[:2]

    pixel_3d_points = np.zeros((h, w, 3), dtype=np.float32)

    # Synthesize 3D points onto the camera with the shifted extrinsic matrix
    for v in range(h):
        for u in range(w):
            depth = depth_left[v, u]

            pixel_homogeneous = np.array([[u, v, 1]])
            pixel_3d_cam_shifted = np.dot(np.linalg.inv(K), pixel_homogeneous.T) * depth

            pixel_3d_points[v, u, :] = pixel_3d_cam_shifted.T

    for i in range(0,12):
        synthesize_image(i, left_colored, pixel_3d_points, set)

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
    
if __name__ == "__main__":
    # time the execution
    start = time.default_timer()
    main()
    stop = time.default_timer()
    print('Time: ', stop - start)
