import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import timeit as time

DIR = 'example'

with open(DIR + '/max_disp.txt', 'r') as f:
    MAX_DISP = int(f.read())
    f.close()

with open(DIR + '/K.txt', 'r') as f:
    K = np.array([[float(num) for num in line.split()] for line in f])
    f.close()

WINDOW_SIZE_X = 9
WINDOW_SIZE_Y = 7
BASELINE = 0.1
FOCAL_LENGTH = K[0, 0]

R = np.array([[1.0, 0.0, 0.0], 
              [0.0, 1.0, 0.0], 
              [0.0, 0.0, 1.0]])
t = np.array([[0.0], 
              [0.0], 
              [0.0]])

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
        cost_volume1[:, :, i] = hamming_distance(left_census, right_shifted)
        cost_volume2[:, :, i] = hamming_distance(right_census, left_shifted)
        right_shifted = np.roll(right_shifted, 1, axis=1)
        left_shifted = np.roll(left_shifted, -1, axis=1)

    return cost_volume1, cost_volume2

def cost_volume2(left_census, right_census):
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

        ltr_cost_volume[:, :, d] = hamming_distance(left_census, right_shifted)
        rtl_cost_volume[:, :, d] = hamming_distance(right_census, left_shifted)

    return ltr_cost_volume, rtl_cost_volume

def aggregate_cost_volume(cost_volume1, cost_volume2, mask_size = (3, 3), sigma = 1):
    for i in range(MAX_DISP):
        cost_volume1[:, :, i] = cv.blur(cost_volume1[:, :, i], mask_size)
        cost_volume2[:, :, i] = cv.blur(cost_volume2[:, :, i], mask_size)
        # cost_volume1[:, :, i] = cv.GaussianBlur(cost_volume1[:, :, i], mask_size, sigma)
        # cost_volume2[:, :, i] = cv.GaussianBlur(cost_volume2[:, :, i], mask_size, sigma)
    
    return cost_volume1, cost_volume2

def find_minimum_cost(aggregated_cost_volume):
    h, w, d = aggregated_cost_volume.shape
    disparity = np.zeros((h, w), dtype=np.uint64)

    for i in range(h):
        for j in range(w):
            disparity[i, j] = np.argmin(aggregated_cost_volume[i, j, :])

    # disparity = np.argmin(aggregated_cost_volume, axis=2)

    return disparity

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
    
    inconsistent_mask = np.logical_not(mask2)  # Get mask for inconsistent pixels

    # Get the coordinates of inconsistent pixels
    inconsistent_y = y[mask1][inconsistent_mask]
    inconsistent_x = x[mask1][inconsistent_mask]

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
        neighbors = lr_check[ymin:ymax, xmin:xmax]

        # Find the most common value among the neighbors
        unique, counts = np.unique(neighbors, return_counts=True)
        most_common_value = unique[np.argmax(counts)]

        # Set the inconsistent pixel to the most common value
        lr_check[y_coord, x_coord] = most_common_value

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

def main():
    left  = cv.imread(DIR + '/im_left.jpg', cv.IMREAD_GRAYSCALE)
    right = cv.imread(DIR + '/im_right.jpg', cv.IMREAD_GRAYSCALE)

    # left = cv.cvtColor(left, cv.COLOR_BGR2GRAY)
    # right = cv.cvtColor(right, cv.COLOR_BGR2GRAY)

    # calculate census transform
    left_census = census_transform(left)
    right_census = census_transform(right)
    print("Finished calculating census transform")

    # calculate cost volumes
    leftToRight_cost_vol, rightToLeft_cost_vol = cost_volume2(left_census, right_census)
    print("Finished calculating cost volumes")

    # aggregate cost volume
    leftToRight_aggregated_cost_vol, rightToLeft_aggregated_cost_vol = aggregate_cost_volume(leftToRight_cost_vol, rightToLeft_cost_vol)
    print("Finished aggregating cost volumes")

    # find minimum cost
    leftToRight_disparity = np.argmin(leftToRight_aggregated_cost_vol, axis=2)
    rightToLeft_disparity = np.argmin(rightToLeft_aggregated_cost_vol, axis=2)
    print("Finished finding minimum cost")

    # filter with consistency test
    leftToRight_disparity = consistency_test2(leftToRight_disparity, rightToLeft_disparity, direction='left')
    rightToLeft_disparity = consistency_test2(rightToLeft_disparity, leftToRight_disparity, direction='right')
    
    # create depth map
    depth_left = create_depth_map(leftToRight_disparity)
    depth_right = create_depth_map(rightToLeft_disparity)
    print("Finished creating depth maps")

    np.savetxt('disp_left.txt', leftToRight_disparity, delimiter=',', fmt='%d', newline='\n')
    np.savetxt('disp_right.txt', rightToLeft_disparity, delimiter=',', fmt='%d', newline='\n')
    with open('depth_left.txt', 'w') as f:
        for i in range(depth_left.shape[0]):
            for j in range(depth_left.shape[1]):
                if j == depth_left.shape[1] - 1:
                    f.write(str(depth_left[i, j]))
                else:
                    f.write(str(depth_left[i, j]) + ',')
            f.write('\n')
        f.close()
    with open('depth_right.txt', 'w') as f:
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
    
    cv.imwrite('disp_left.png', leftToRight_disparity)
    cv.imwrite('disp_right.png', rightToLeft_disparity)
    cv.imwrite('depth_left.png', depth_left)
    cv.imwrite('depth_right.png', depth_right)

    print("Finished saving disparity and depth maps")

def synthesize_image(i, left, pixel_3d_points):
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
                
            # else:
            #     synth_img[v, u, :] = left[v, u, :]

    # Save the resulting image with the corresponding shift index
    cv.imwrite("synth_" + str(i).zfill(2) + ".jpg", synth_img)
    print("Finished synthesizing synth_{}.jpg".format(i))
    
def main2():
    # depth_left = np.loadtxt(DIR + '/depth_left.txt', delimiter=',')
    depth_left = np.loadtxt('depth_left.txt', delimiter=',')

    left = cv.imread(DIR + '/im_left.jpg', cv.IMREAD_COLOR)
    h, w = left.shape[:2]

    pixel_3d_points = np.zeros((h, w, 3), dtype=np.float32)

    # Synthesize 3D points onto the camera with the shifted extrinsic matrix
    for v in range(h):
        for u in range(w):
            depth = depth_left[v, u]

            pixel_homogeneous = np.array([[u, v, 1]])
            pixel_3d_cam_shifted = np.dot(np.linalg.inv(K), pixel_homogeneous.T) * depth

            pixel_3d_points[v, u, :] = pixel_3d_cam_shifted.T
    print("Finished reprojecting 3D points")

    for i in range(0,12):
        synthesize_image(i, left, pixel_3d_points)

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

    # left = cv.imread(DIR + '/im_left.jpg', cv.IMREAD_COLOR)
    # # left = cv.cvtColor(left, cv.COLOR_BGR2RGB)
    # reprojection_left = np.zeros((h, w, 3), dtype=np.float32)

    # for i in range(12):
    #     t[0, 0] = i * 0.01
    #     print(t)
    #     for y in range(h):
    #         for x in range(w):
    #             point_left = reprojection_points_left[y, x, :]
    #             point_left = point_left.reshape((3, 1))
    #             # convert to camera coordinates
    #             point_left = np.matmul(R, point_left) + t
    #             point_left = np.matmul(K, point_left)
    #             point_left = point_left / point_left[2, 0]
    #             try:
    #                 x_new = int(point_left[0, 0])
    #                 y_new = int(point_left[1, 0])
    #             except:
    #                 x_new = x
    #                 y_new = y
    #             if x_new >= 0 and x_new < w and y_new >= 0 and y_new < h:
    #                 reprojection_left[y_new, x_new, :] = left[y, x, :]

    #     cv.imwrite('synth_' + str(i) + '.png', reprojection_left)
    #     # plt.imsave('synth_' + str(i) + '.png', reprojection_left)
    
if __name__ == "__main__":
    # time the execution
    start = time.default_timer()
    main()
    main2()
    stop = time.default_timer()
    print('Time: ', stop - start)