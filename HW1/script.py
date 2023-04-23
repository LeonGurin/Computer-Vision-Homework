import cv2 as cv
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg
import matplotlib.pyplot as plt
import os
import random
import time
from numba import njit, prange
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
import concurrent.futures




# functions
# def euclidean_distance(p1, p2):
#     return np.linalg.norm(p1 - p2)

@njit
def euclidean_distance(des1, des2):
    return np.sqrt(np.sum((des1 - des2) ** 2))

def calculate_distances_matrix(descriptors1, descriptors2):
    diff = descriptors1[:, np.newaxis, :] - descriptors2[np.newaxis, :, :]
    squared_diff = diff ** 2
    distances = np.sqrt(np.sum(squared_diff, axis=-1))
    return distances

@njit
def calculate_distances_matrix_numba(descriptors1, descriptors2):
    n_rows, n_cols = len(descriptors1), len(descriptors2)
    distances = np.empty((n_rows, n_cols))
    for i in range(n_rows):
        for j in range(n_cols):
            distances[i, j] = euclidean_distance(descriptors1[i], descriptors2[j])

    return distances

def calculate_distances_matrix_scipy(descriptors1, descriptors2):
    return cdist(descriptors1, descriptors2, metric='euclidean')


@njit
def find_best_matches_numba(distances, threshold=0.8):
    n_rows, n_cols = distances.shape
    best_matches = []

    for i in range(n_rows):
        # Initialize the smallest and second smallest values and their indices
        min_val, second_min_val = np.inf, np.inf
        min_idx, second_min_idx = -1, -1

        # Find the two smallest values and their indices in the row
        for j in range(n_cols):
            if distances[i, j] < min_val:
                second_min_val = min_val
                min_val = distances[i, j]
                min_idx = j
            elif distances[i, j] < second_min_val:
                second_min_val = distances[i, j]

        # Check the condition and append to the best_matches list
        if min_val < threshold * second_min_val:
            best_matches.append((i, min_idx))

    return best_matches

def find_best_matches(distances, threshold=0.8):
    best_matches = []
    for i, row in enumerate(distances):
        smallest_indices = np.argpartition(row, 2)[:2]
        smallest_values = sorted((row[j], j) for j in smallest_indices)
        min_value, idx = smallest_values[0]
        second_min_value = smallest_values[1][0]
        
        if min_value < threshold * second_min_value:
            best_matches.append((i, idx))
            
    return best_matches

def read_sorted_pieces(pieces_dir):
    # Custom sorting function to extract the serial number from the filename
    def sort_key(filename):
        basename = os.path.splitext(filename)[0]  # remove extension
        serial_number = int(basename.split("_")[-1])  # get the last part after the last underscore
        return serial_number

    # List and sort the filenames based on the serial number
    sorted_filenames = sorted(os.listdir(pieces_dir), key=sort_key)

    pieces = []
    for filename in sorted_filenames:
        print("Reading piece: ", filename)
        img = cv.imread(os.path.join(pieces_dir, filename))
        pieces.append(img)
    
    return pieces

def plot_best_matches(piece1, piece2, best_matches, kp1, kp2):
    dm_best_matches = [cv.DMatch(_queryIdx=m[0], _trainIdx=m[1], _imgIdx=0, _distance=0) for m in best_matches]
    result = cv.drawMatches(piece1, kp1, piece2, kp2, dm_best_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(result), plt.show()

def plot_keypoints(piece, kp):
    result = cv.drawKeypoints(piece, kp, None)
    plt.imshow(result), plt.show()


def get_piece_transform(piece, target, warp_type="affine"):
    #global distances_time, best_matches_time
    # find keypoints and descriptors for the piece and the target
    sift = cv.SIFT_create()
    target_kp, target_des = sift.detectAndCompute(target, None)
    kp, des = sift.detectAndCompute(piece, None)
    #plot_keypoints(piece, kp)
    print("Number of keypoints in first piece: ", len(kp))

    start_time = time.time()
    distances = calculate_distances_matrix_scipy(des, target_des)
    middle_time = time.time()
    best_matches = find_best_matches_numba(distances, threshold=0.75)
    end_time = time.time()
    #distances_time += middle_time - start_time
    #best_matches_time += end_time - middle_time
    print("Number of best matches: ", len(best_matches))
    #plot_best_matches(pieces[1], target, best_matches, kp, target_kp)

    # implement RANSAC to find the best transformation matrix
    best_inlier_ratio = 0
    all_src_pts = np.float32([kp[match[0]].pt for match in best_matches]).reshape(-1, 1, 2)
    all_dst_pts = np.float32([target_kp[match[1]].pt for match in best_matches]).reshape(-1, 1, 2)
    for _ in range(10000):
        # select 3 or 4 random points
        k = 3 if warp_type == "affine" else 4
        random_matches = random.sample(best_matches, k=k)
        # calculate transformation matrix
        src_pts = np.float32([kp[match[0]].pt for match in random_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([target_kp[match[1]].pt for match in random_matches]).reshape(-1, 1, 2)
        if warp_type == "affine":
            M = cv.getAffineTransform(src_pts, dst_pts)
            transformed_points = cv.transform(all_src_pts, M)
        else:
            M = cv.getPerspectiveTransform(src_pts, dst_pts)
            transformed_points = cv.perspectiveTransform(all_src_pts, M)
        # calculate the residuals for all the points
        
        # Calculate the distance between transformed points and their corresponding points in the target image
        residuals = np.linalg.norm(transformed_points - all_dst_pts, axis=2)
        # if the number of inliers is larger than the current best, update the best
        inliers = residuals < 5
        inlier_ratio = np.sum(inliers) / len(residuals)
        if inlier_ratio > best_inlier_ratio:
            best_inlier_ratio = inlier_ratio
            transform = M
    print("best_inlier_ratio: ", best_inlier_ratio)
    if warp_type == "affine":
        transform = np.vstack((transform, np.array([0, 0, 1])))
    return transform, best_inlier_ratio > 0.1


def solve_puzzle(directory):
    print("Solving puzzle: ", os.path.basename(directory))
    warp_type = os.path.basename(directory).split('_')[1]
    pieces_dir = os.path.join(directory, 'pieces')
    # find the txt file in the directory using a one-liner given there is only one
    txt_file = [f for f in os.listdir(directory) if f.endswith('.txt')][0]
    # txt file is in the format: warp_mat_1__H_537__W_735_.txt
    # extract the height and width of the target image
    target_height = int(txt_file.split('__')[1].split('_')[1])
    target_width = int(txt_file.split('__')[2].split('_')[1])
    with open(os.path.join(directory, txt_file), 'r') as f:
        # read the transformation matrix
        global_transform = np.array([line.split() for line in f.readlines()], dtype=np.float32)

    # read pieces
    pieces = read_sorted_pieces(pieces_dir)
    gray_pieces = [cv.cvtColor(piece, cv.COLOR_BGR2GRAY) for piece in pieces]

    gray_target = cv.warpPerspective(gray_pieces[0], global_transform, (target_width, target_height), flags=cv.INTER_CUBIC)
    target = cv.warpPerspective(pieces[0], global_transform, (target_width, target_height), flags=cv.INTER_CUBIC)
    target_mask = cv.warpPerspective(np.ones_like(gray_pieces[0]), global_transform, (target_width, target_height), flags=cv.INTER_CUBIC)
    
    transformed_pieces = [target.copy()]
    total_successes = 1
    for i in range(1, len(pieces)):
        print(f"Processing piece: {i} of {len(pieces)}")
        # find the transformation matrix for the piece
        transform, success = get_piece_transform(gray_pieces[i], gray_target, warp_type=warp_type)

        if not success:
            continue
        total_successes += 1

        # apply the transformation matrix to the test piece
        transformed_piece = cv.warpPerspective(pieces[i], transform, (target_width, target_height), flags=cv.INTER_CUBIC)
        gray_transformed_piece = cv.warpPerspective(gray_pieces[i], transform, (target_width, target_height), flags=cv.INTER_CUBIC)
        #cv.imshow("Transformed Piece", transformed_piece)
        target[gray_target == 0] = transformed_piece[gray_target == 0]
        gray_target[gray_target == 0] = gray_transformed_piece[gray_target == 0]
        target_mask[gray_transformed_piece != 0] += 1

        transformed_pieces.append(transformed_piece)

    #print("Time to calculate distances: ", distances_time)
    #print("Time to find best matches: ", best_matches_time)

    saving_dir = os.path.join("results", os.path.basename(directory))
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    for i, piece in enumerate(transformed_pieces):
        cv.imwrite(os.path.join(saving_dir, f"piece_{i+1}_relative.jpeg"), piece)
    
    cv.imwrite(os.path.join(saving_dir, f"solution_{total_successes}_{len(pieces)}.jpeg"), target)

    # Plot the coverage mask
    plt.figure()
    plt.imshow(target_mask, vmin=0, vmax=np.max(target_mask))
    plt.colorbar()
    plt.title("coverage count")
    # Save the plot as a JPEG file
    plt.savefig(os.path.join(saving_dir, "coverage_count.jpeg"), format='jpeg')
    # Clear the current figure
    plt.clf()

    print("Done solving puzzle: ", os.path.basename(directory))



distances_time = 0
best_matches_time = 0

def solve_all_puzzles():
    with concurrent.futures.ThreadPoolExecutor() as executor:
        puzzle_dirs = [os.path.join('puzzles', puzzle_path) for puzzle_path in os.listdir('puzzles')]
        print(os.listdir('puzzles'))
        executor.map(solve_puzzle, puzzle_dirs)
    
    print('done')

def main():
    solve_all_puzzles()
    #solve_puzzle('puzzles/puzzle_homography_3')
    #cv.waitKey(0)
    print('done')



if __name__ == '__main__':
    main()