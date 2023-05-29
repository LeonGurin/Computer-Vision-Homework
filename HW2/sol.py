import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

DIR = 'set_1'

with open(DIR + '/max_disp.txt', 'r') as f:
    MAX_DISP = int(f.read())
    f.close()

with open(DIR + '/K.txt', 'r') as f:
    K = np.array([[float(num) for num in line.split()] for line in f])
    f.close()

WINDOW_SIZE_X = 5
WINDOW_SIZE_Y = 5
BASELINE = 0.1
FOCAL_LENGTH = K[0, 0]

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

def main():
    left  = cv.imread(DIR + '/im_left.jpg', cv.IMREAD_GRAYSCALE)
    right = cv.imread(DIR + '/im_right.jpg', cv.IMREAD_GRAYSCALE)

    # calculate census transform
    left_census = census_transform(left)
    right_census = census_transform(right)
    
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
    
    # create depth map
    depth_left = create_depth_map(leftToRight_disparity)
    depth_right = create_depth_map(rightToLeft_disparity)

    cv.imwrite('disp_left.png', leftToRight_disparity)
    cv.imwrite('disp_right.png', rightToLeft_disparity)
    cv.imwrite('depth_left.png', depth_left)
    cv.imwrite('depth_right.png', depth_right)

    np.savetxt('disp_left.txt', leftToRight_disparity, delimiter=',')
    np.savetxt('disp_right.txt', rightToLeft_disparity, delimiter=',')
    np.savetxt('depth_left.txt', depth_left, delimiter=',')
    np.savetxt('depth_right.txt', depth_right, delimiter=',')

def main2():
    # calculate reprojection using depth map D and intrinsics matrix K
    # depth_left = np.loadtxt(DIR + '/depth_left.txt', delimiter=',')
    depth_left = np.loadtxt('depth_left.txt', delimiter=',')
    
    R = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    t = np.array([[0.0], [0.0], [0.0]])

    h, w = depth_left.shape
    reprojection_left = np.zeros((h, w, 3), dtype=np.float32)

    reprojection_points_left = np.zeros((h, w, 3), dtype=np.float32)

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

    left = cv.imread(DIR + '/im_left.jpg', cv.IMREAD_COLOR)

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

        cv.imwrite('synth_' + str(i) + '.png', reprojection_left)
    
if __name__ == "__main__":
    main()
    main2()