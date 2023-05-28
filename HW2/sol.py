import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


with open('set_1/max_disp.txt', 'r') as f:
    MAX_DISP = int(f.read())
    f.close()

with open('set_1/K.txt', 'r') as f:
    K = np.array([[float(num) for num in line.split()] for line in f])
    f.close()

WINDOW_SIZE_X = 11
WINDOW_SIZE_Y = 5
BASELINE = 1
FOCAL_LENGTH = K[0, 0]

def census_transform(img):
    h, w = img.shape
    census = np.zeros((h, w), dtype=np.uint64)

    start_x = WINDOW_SIZE_X // 2
    start_y = WINDOW_SIZE_Y // 2

    for i in range(start_x, h - start_x):
        for j in range(start_y, w - start_y):
            binary = ''
            for k in range(i - start_x, i + start_x + 1):
                for l in range(j - start_y, j + start_y + 1):
                    if img[k, l] >= img[i, j]:
                        binary += '1'
                    else:
                        binary += '0'
            census[i, j] = int(binary, 2)
    return census

def hamming_distance(left, right):
    h, w = left.shape
    hamming = np.zeros((h, w), dtype=np.uint64)
    for i in range(h):
        for j in range(w):
            hamming[i, j] = bin(left[i, j] ^ right[i, j]).count('1')
    return hamming

def cost_volume(left_census, right_census):
    h, w = left_census.shape
    cost_volume1 = np.zeros((h, w, MAX_DISP), dtype=np.uint64)
    cost_volume2 = np.zeros((h, w, MAX_DISP), dtype=np.uint64)

    right_shifted = right_census.copy()
    left_shifted = left_census.copy()

    for i in range(MAX_DISP):
            right_shifted = np.roll(right_census, i, axis=1)
            hamming = hamming_distance(left_census, right_shifted)
            cost_volume1[:, :, i] = hamming

            left_shifted = np.roll(left_census, -i, axis=1)
            hamming = hamming_distance(right_census, left_shifted)
            cost_volume2[:, :, i] = hamming

    return cost_volume1, cost_volume2

def aggregate_cost_volume(cost_volume1, cost_volume2):
    mask_size = (7, 7)
    # perform uniform averaging with openCV filter
    for i in range(MAX_DISP):
        cost_volume1[:, :, i] = cv.blur(cost_volume1[:, :, i], mask_size)
        cost_volume2[:, :, i] = cv.blur(cost_volume2[:, :, i], mask_size)
    
    return cost_volume1, cost_volume2

def find_minimum_cost(aggregated_cost_volume):
    h, w, _ = aggregated_cost_volume.shape
    disparity = np.zeros((h, w), dtype=np.uint64)
    
    disparity = np.argmin(aggregated_cost_volume, axis=2)

    # for i in range(h):
    #     for j in range(w):
    #         min_val = aggregated_cost_volume[i, j, 0]
    #         for d in range(MAX_DISP):
    #             if aggregated_cost_volume[i, j, d] < min_val:
    #                 min_val = aggregated_cost_volume[i, j, d]

    #         # disparity[i, j] = np.argmin(aggregated_cost_volume[i, j, :])
    #         disparity[i, j] = min_val
    
    return disparity

def consistency_test(left_disparity, right_disparity):
    h, w = left_disparity.shape
    threshold = 30
    
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
    # depth_map = np.where(disparity != 0, (BASELINE * FOCAL_LENGTH) / disparity, 0.0)
    # return depth_map

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
    left  = cv.imread('example/im_left.jpg', cv.IMREAD_GRAYSCALE)
    right = cv.imread('example/im_right.jpg', cv.IMREAD_GRAYSCALE)

    # calculate census transform
    left_census = census_transform(left)
    right_census = census_transform(right)
    
    # calculate cost volumes
    leftToRight_cost_vol, rightToLeft_cost_vol = cost_volume(left_census, right_census)

    # aggregate cost volume
    leftToRight_aggregated_cost_vol, rightToLeft_aggregated_cost_vol = aggregate_cost_volume(leftToRight_cost_vol, rightToLeft_cost_vol)

    # find minimum cost
    # leftToRight_disparity = np.argmin(leftToRight_aggregated_cost_vol, axis=2)
    # rightToLeft_disparity = np.argmin(rightToLeft_aggregated_cost_vol, axis=2)
    leftToRight_disparity = find_minimum_cost(leftToRight_aggregated_cost_vol)
    rightToLeft_disparity = find_minimum_cost(rightToLeft_aggregated_cost_vol)

    # filter with consistency test
    leftToRight_disparity, rightToLeft_disparity = consistency_test(leftToRight_disparity, rightToLeft_disparity)
    
    # create depth map
    depth_left = create_depth_map(leftToRight_disparity).astype(np.uint8)
    depth_right = create_depth_map(rightToLeft_disparity).astype(np.uint8)
    # depth_left = np.where(leftToRight_disparity != 0, (BASELINE * FOCAL_LENGTH) / leftToRight_disparity, 0)
    # depth_right = np.where(rightToLeft_disparity != 0, (BASELINE * FOCAL_LENGTH) / rightToLeft_disparity, 0)

    # depth_left = cv.applyColorMap(depth_left, cv.COLORMAP_JET)
    # depth_right = cv.applyColorMap(depth_right, cv.COLORMAP_JET)

    cv.imwrite('leftToRight_disparity.png', leftToRight_disparity)
    cv.imwrite('rightToLeft_disparity.png', rightToLeft_disparity)
    cv.imwrite('depth_left.png', depth_left)
    cv.imwrite('depth_right.png', depth_right)

    # display results
    # cv.imshow('left', left)
    # cv.imshow('right', right)
    # cv.imshow('leftToRight_disparity', leftToRight_disparity)
    # cv.imshow('rightToLeft_disparity', rightToLeft_disparity)
    cv.imshow('depth_left', depth_left)
    cv.imshow('depth_right', depth_right)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()