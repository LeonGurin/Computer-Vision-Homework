import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


with open('set_1/max_disp.txt', 'r') as f:
    MAX_DISP = int(f.read())
    f.close()

WINDOW_SIZE = 3 # window size for census transform


def census_transform(img):
    h, w = img.shape
    census = np.zeros((h, w), dtype=np.uint64)

    start = WINDOW_SIZE // 2

    for i in range(start, h - start):
        for j in range(start, w - start):
            binary = ''
            # iterate over window
            for k in range(-start, start + 1):
                for l in range(-start, start + 1):
                    if img[i, j] >= img[i + k, j + l]:
                        binary += '1'
                    else:
                        binary += '0'
            census[i, j] = int(binary, 2)
    return census

# def hamming_distance(left, right):
    h, w = left.shape
    hamming = np.zeros((h, w), dtype=np.uint8)
    for i in range(1, h-1):
        for j in range(1, w-1):
            hamming[i, j] = bin(left[i, j] ^ right[i, j]).count('1')
    return hamming

def cost_volume(left_census, right_census, d=MAX_DISP):
    h, w = left_census.shape
    cost_volume = np.zeros((h, w, d), dtype=np.uint64)

    for i in range(d):
        shifted_right_census = np.roll(right_census, i, axis=1)
        diff = np.count_nonzero(left_census != shifted_right_census)
        cost_volume[:, :, i] = diff

    return cost_volume

def aggregate_cost_volume(cost_volume):
    h, w, d = cost_volume.shape
    aggregated_cost_volume = np.zeros((h, w, d), dtype=np.uint64)
    
    mask_size = (3, 3)
    # perform uniform averaging with openCV filter
    for i in range(d):
        aggregated_cost_volume[:, :, i] = cv.blur(cost_volume[:, :, i], mask_size)
    
    return aggregated_cost_volume

def find_minimum_cost(aggregated_cost_volume):
    h, w, d = aggregated_cost_volume.shape
    disparity = np.zeros((h, w), dtype=np.uint64)
    for i in range(1, h-1):
        for j in range(1, w-1):
            # find minimum cost
            disparity[i, j] = np.argmin(aggregated_cost_volume[i, j, :])
    
    return disparity

def consistency_test(disparity, aggregated_cost_volume):
    h, w, _ = aggregated_cost_volume.shape

    for y in range(h):
        for x in range(w):
            # Find the corresponding pixel position in the right image
            corresponding_x = x - disparity[y, x]

            # Retrieve the disparity value from the right image
            if 0 <= corresponding_x < w:
                corresponding_disparity = disparity[y, corresponding_x]
            else:
                corresponding_disparity = 0

            # Compare the retrieved disparity value with the initial disparity value
            diff = abs(disparity[y, x] - corresponding_disparity)

            # Perform the consistency test
            if diff <= 5:
                disparity[y, x] = 255  # Mark as consistent
    return disparity                

def main():
    left = cv.imread('set_1/im_left.jpg', cv.IMREAD_GRAYSCALE)
    right = cv.imread('set_1/im_right.jpg', cv.IMREAD_GRAYSCALE)
    
    # calculate census transform
    left_census = census_transform(left)
    right_census = census_transform(right)
    
    # calculate cost volume
    cost_vol = cost_volume(left_census, right_census)
    
    # aggregate cost volume
    aggregated_cost_volume = aggregate_cost_volume(cost_vol)
    
    # find minimum cost
    disparity = np.argmin(aggregated_cost_volume, axis=2)
    # disparity = find_minimum_cost(aggregated_cost_volume)
    
    # filter with consistency test
    # disparity = consistency_test(disparity, aggregated_cost_volume)
    
    cv.imwrite('disparity_map.png', disparity)
    # plt.imshow(disparity, cmap='gray')
    # plt.imshow(left_census, cmap='gray')
    # plt.imshow(right_census, cmap='gray')
    plt.show()

if __name__ == "__main__":
    main()