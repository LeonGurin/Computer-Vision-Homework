import cv2 as cv
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg
import matplotlib.pyplot as plt
import os
import random
#from numba import njit, prange
#from scipy.spatial.distance import cdist
#import concurrent.futures
import sys
import argparse
import csv

def read_csv(file_path):
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        return {row['']: row for row in reader}

def write_csv(file_path, data):
    with open(file_path, 'w') as csvfile:
        fieldnames = ['', 'total pieces', 'solved pieces']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for key, row in data.items():
            writer.writerow(row)

class Args:
    def __init__(self, path="puzzle_affine_1", best_match_threshold=0.7, distance_threshold=5.0, 
                 max_iterations=3000, success_ratio=0.5, min_required_matches=9):     
        # global   
        self.puzzle_dir = f'puzzles/{path}'
        self.success_ratio = success_ratio
        self.save_results = True

        # optimizable
        self.max_iterations = max_iterations
        self.distance_threshold = distance_threshold
        self.best_match_threshold = best_match_threshold
        self.min_required_matches = min_required_matches

        # self.nfeatures = 1000
        # self.nOctaveLayers = 3
        # self.contrastThreshold = 0.04
        # self.edgeThreshold = 10
        # self.sigma = 1.6
        # self.nOctaves = 4

global_parameter_dict = {
    # name : argument_class
    "puzzle_affine_1" : Args(path="puzzle_affine_1"),
    "puzzle_affine_2" : Args(path="puzzle_affine_2"),
    "puzzle_affine_3" : Args(path="puzzle_affine_3"),
    "puzzle_affine_4" : Args(path="puzzle_affine_4"),
    "puzzle_affine_5" : Args(path="puzzle_affine_5",
                                best_match_threshold=0.6953002546492293,
                                distance_threshold=0.5043536464906101,max_iterations=2082,
                                min_required_matches=21),
    "puzzle_affine_6" : Args(path="puzzle_affine_6"),
    "puzzle_affine_7" : Args(path="puzzle_affine_7",
                                best_match_threshold=0.608,
                                distance_threshold=0.729,
                                min_required_matches=15,
                                success_ratio=0.5),
    "puzzle_affine_8" : Args(path="puzzle_affine_8",
                                best_match_threshold=0.7088669478457845,
                                distance_threshold=6.433027284503128, max_iterations=2000,
                                min_required_matches=8),
    "puzzle_affine_9" : Args(path="puzzle_affine_9",
                                best_match_threshold=0.5752,
                                distance_threshold=9.0664, max_iterations=3000,
                                min_required_matches=5, success_ratio=1),
    "puzzle_affine_10" : Args(path="puzzle_affine_10",
                                best_match_threshold=0.7902,
                                distance_threshold=1.9544, max_iterations=3675,
                                min_required_matches=8, success_ratio=1),

    "puzzle_homography_1" : Args(path="puzzle_affine_11"),
    "puzzle_homography_2" : Args(path="puzzle_homography_2"),
    "puzzle_homography_3" : Args(path="puzzle_homography_3"),
    "puzzle_homography_4" : Args(path="puzzle_homography_4"),
    "puzzle_homography_5" : Args(path="puzzle_homography_5",
                                    best_match_threshold=0.564,
                                    distance_threshold=4.399,
                                    min_required_matches=21),
    "puzzle_homography_6" : Args(path="puzzle_homography_6",
                                    best_match_threshold=0.559,
                                    distance_threshold=8.9, 
                                    min_required_matches=12),
    "puzzle_homography_7" : Args(path="puzzle_homography_7",
                                    best_match_threshold=0.6884,
                                    distance_threshold=2.8796, max_iterations=2335,
                                    min_required_matches=12, success_ratio=1),
    "puzzle_homography_8" : Args(path="puzzle_homography_8",
                                    best_match_threshold=0.6390466644494099,
                                    distance_threshold=1.0519974371695389, max_iterations=2296,
                                    min_required_matches=11, success_ratio=1),
    "puzzle_homography_9" : Args(path="puzzle_homography_9"),
    "puzzle_homography_10" : Args(path="puzzle_homography_10",
                                    best_match_threshold=0.7683,
                                    distance_threshold=1.5568, max_iterations=4000,
                                    min_required_matches=19, success_ratio=1),
}


# @njit
# def euclidean_distance(des1, des2):
#     return np.sqrt(np.sum((des1 - des2) ** 2))

def calculate_distances_matrix(descriptors1, descriptors2):
    diff = descriptors1[:, np.newaxis, :] - descriptors2[np.newaxis, :, :]
    squared_diff = diff ** 2
    distances = np.sqrt(np.sum(squared_diff, axis=-1))
    return distances

# @njit
# def calculate_distances_matrix_numba(descriptors1, descriptors2):
#     n_rows, n_cols = len(descriptors1), len(descriptors2)
#     distances = np.empty((n_rows, n_cols))
#     for i in range(n_rows):
#         for j in range(n_cols):
#             distances[i, j] = euclidean_distance(descriptors1[i], descriptors2[j])

#     return distances

# def calculate_distances_matrix_scipy(descriptors1, descriptors2):
#     return cdist(descriptors1, descriptors2, metric='euclidean')


# @njit
# def find_best_matches_numba(distances, threshold=0.8):
#     n_rows, n_cols = distances.shape
#     best_matches = []

#     for i in range(n_rows):
#         # Initialize the smallest and second smallest values and their indices
#         min_val, second_min_val = np.inf, np.inf
#         min_idx, second_min_idx = -1, -1

#         # Find the two smallest values and their indices in the row
#         for j in range(n_cols):
#             if distances[i, j] < min_val:
#                 second_min_val = min_val
#                 min_val = distances[i, j]
#                 min_idx = j
#             elif distances[i, j] < second_min_val:
#                 second_min_val = distances[i, j]

#         # Check the condition and append to the best_matches list
#         if min_val < threshold * second_min_val:
#             best_matches.append((i, min_idx))

#     return best_matches

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
        # print("Reading piece: ", filename)
        img = cv.imread(os.path.join(pieces_dir, filename))
        pieces.append(img)
    
    return pieces

def save_results(output_dir, pieces, target, target_mask, transformed_pieces):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, piece in enumerate(transformed_pieces):
        cv.imwrite(os.path.join(output_dir, f"piece_{i+1}_relative.jpeg"), piece)
    
    cv.imwrite(os.path.join(output_dir, f"solution_{len(transformed_pieces)}_{len(pieces)}.jpeg"), target)

    # Plot the coverage mask
    plt.figure()
    plt.imshow(target_mask, vmin=0, vmax=np.max(target_mask))
    plt.colorbar()
    plt.title("coverage count")
    # Save the plot as a JPEG file
    plt.savefig(os.path.join(output_dir, "coverage_count.jpeg"), format='jpeg')
    # Clear the current figure
    plt.clf()

def plot_best_matches(piece1, piece2, best_matches, kp1, kp2):
    dm_best_matches = [cv.DMatch(_queryIdx=m[0], _trainIdx=m[1], _imgIdx=0, _distance=0) for m in best_matches]
    result = cv.drawMatches(piece1, kp1, piece2, kp2, dm_best_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(result), plt.show()

def plot_keypoints(piece, kp):
    result = cv.drawKeypoints(piece, kp, None)
    plt.imshow(result), plt.show()

def find_features(piece, target, args):
    sift = cv.SIFT_create()
    # Detect keypoints and compute descriptors for the piece
    kp, des = sift.detectAndCompute(piece, None)
    #plot_keypoints(piece, kp)

    keypoints, descriptors = sift.detectAndCompute(target, None)
    print(f"Found {len(keypoints)} keypoints in target image. Wello")

    distances = calculate_distances_matrix(des, descriptors)
    best_matches = find_best_matches(distances, threshold=args.best_match_threshold)
    
    return kp, keypoints, best_matches

def get_piece_transform(piece, target, warp_type, args):
    # find keypoints and descriptors
    kp, target_kp, best_matches = find_features(piece, target, args)
    print(f"Found {len(best_matches)} matches")

    if len(best_matches) < args.min_required_matches:
        return None, False

    # implement RANSAC to find the best transformation matrix
    best_cost = float('inf')
    all_src_pts = np.float32([kp[match[0]].pt for match in best_matches]).reshape(-1, 1, 2)
    all_dst_pts = np.float32([target_kp[match[1]].pt for match in best_matches]).reshape(-1, 1, 2)
    for j in range(args.max_iterations):
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

        # Calculate the distance between transformed points and their corresponding points in the target image
        residuals = np.linalg.norm(transformed_points - all_dst_pts, axis=2)
        cost = np.sum(np.minimum(residuals ** 2, args.distance_threshold ** 2))
    
        # if the number of inliers is larger than the current best, update the best
        #inliers = residuals < args.distance_threshold
        #inlier_ratio = np.sum(inliers) / len(residuals)
        if cost < best_cost:
            best_cost = cost
            transform = M
            #if best_cost > 0.99 and j > 100:
            #    break
    average_cost = best_cost / len(residuals)
    print("best average cost: ", average_cost)
    if warp_type == "affine":
        transform = np.vstack((transform, np.array([0, 0, 1])))
    return transform, average_cost < args.distance_threshold * args.success_ratio






def solve_puzzle(directory, args):
    random.seed(42)
    np.random.seed(42)
    print("Solving puzzle: ", os.path.basename(directory))
    warp_type = os.path.basename(directory).split('_')[1] # affine or homography
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
    unsuccessful_pieces_idxs = list(range(1, len(pieces)))

    progress = True
    while not unsuccessful_pieces_idxs == [] and progress:
        progress = False
        for i in unsuccessful_pieces_idxs:
            print(f"Processing piece: {i+1} of {len(pieces)}")
            # find the transformation matrix for the piece
            transform, success = get_piece_transform(gray_pieces[i], gray_target, warp_type, args)

            if not success:
                print(f"Failed to find a good transformation matrix for piece {i}")
                continue
            progress = True
            unsuccessful_pieces_idxs.remove(i)

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

    if args.save_results:
        output_dir = os.path.join("results", os.path.basename(directory))
        save_results(output_dir, pieces, target, target_mask, transformed_pieces)

    print(f"Done solving puzzle {os.path.basename(directory)}, with {len(transformed_pieces)} pieces out of {len(pieces)}")
    return len(transformed_pieces) / len(pieces), len(transformed_pieces)

# def solve_all_puzzles_concurrent(args):
#     # Read the CSV file into a dictionary
#     csv_data = read_csv('piece_count.csv')

#     # solve all puzzles in parallel
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         puzzle_dirs = [os.path.join('puzzles', puzzle_path) for puzzle_path in os.listdir(args.puzzle_dir)]
#         args = [global_parameter_dict[puzzle_path] for puzzle_path in os.listdir(args.puzzle_dir)]
#         # Update the dictionary with the results of the solve_puzzle function
#         results = executor.map(solve_puzzle, puzzle_dirs, args)
#         print(results)
#         for puzzle_path, (_, solved_pieces) in zip(puzzle_dirs, results):
#             puzzle_name = os.path.basename(puzzle_path)
#             csv_data[puzzle_name]['solved pieces'] = str(solved_pieces)

#     # Write the updated dictionary back to the CSV file
#     write_csv('piece_count.csv', csv_data)

def solve_all_puzzles(global_args):
    # Read the CSV file into a dictionary
    csv_data = read_csv('piece_count.csv')

    # solve all puzzles
    for puzzle_path in os.listdir(global_args.puzzle_dir):
        puzzle_name = os.path.basename(puzzle_path)
        args = global_parameter_dict[puzzle_path]
        success_ratio, solved_pieces = solve_puzzle(os.path.join(global_args.puzzle_dir, puzzle_path), args)
        csv_data[puzzle_name]['solved pieces'] = str(solved_pieces)

    # Write the updated dictionary back to the CSV file
    write_csv('piece_count.csv', csv_data)


    
def parse_args(argv):
    parser = argparse.ArgumentParser(description='Solve a puzzle')
    parser.add_argument('--puzzle_dir', type=str, default='puzzles', help='path to the puzzle directory. If not specified, all puzzles in the puzzles directory will be solved')
    parser.add_argument('--max_iterations', type=int, default=3000, help='maximum number of iterations for RANSAC')
    parser.add_argument('--distance_threshold', type=float, default=5.0, help='maximum distance between two features to be considered a match')
    parser.add_argument('--min_required_matches', type=int, default=9, help='minimum number of matches required to calculate the transformation matrix')
    # parser.add_argument('--nfeatures', type=int, default=0, help='number of features to detect')
    # parser.add_argument('--nOctaveLayers', type=int, default=3, help='number of octave layers')
    # parser.add_argument('--contrastThreshold', type=float, default=0.04, help='contrast threshold')
    # parser.add_argument('--edgeThreshold', type=int, default=10, help='edge threshold')
    # parser.add_argument('--sigma', type=float, default=1.6, help='sigma for the gaussian blur in the sift detector')
    # parser.add_argument('--nOctaves', type=int, default=4, help='number of octaves for the gaussian blur in the sift detector')
    parser.add_argument('--best_match_threshold', type=float, default=0.7, help='best match threshold for the ratio test in the sift matcher')
    parser.add_argument('--success_ratio', type=float, default=0.5, help='minimum ratio of cost that must be achieved to successfully match a piece')
    parser.add_argument('--save_results', type=bool, default=True, help='save the results of the puzzle solving')
    args = parser.parse_args(argv)
    return args

def main():
    args = parse_args(sys.argv[1:])
    print(args)
    if os.path.exists(os.path.join(args.puzzle_dir, 'pieces')):
        solve_puzzle(args.puzzle_dir, args)
    else:
        solve_all_puzzles(args)
    #cv.waitKey(0)
    # print('done')

if __name__ == '__main__':
    main()