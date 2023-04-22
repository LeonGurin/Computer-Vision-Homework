import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import random


# functions

def euclidean_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def calculate_distances_matrix(descriptors1, descriptors2):
    distances = np.zeros((len(descriptors1), len(descriptors2)))
    for i, des1 in enumerate(descriptors1):
        for j, des2 in enumerate(descriptors2):
            distances[i, j] = euclidean_distance(des1, des2)
    return distances

def find_best_matches(distances, threshold=0.8):
    best_matches = []
    for i, dist in enumerate(distances):
        sorted_dist = np.argsort(dist)
        min_value = dist[sorted_dist[0]]
        second_min_value = dist[sorted_dist[1]]
        if min_value < threshold * second_min_value:
            best_matches.append((i, sorted_dist[0]))
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

def solve_puzzle(directory):
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
    
    for i in range(1, len(pieces)):

        # find keypoints and descriptors for all pieces
        sift = cv.SIFT_create()
        #kp, des = zip(*[sift.detectAndCompute(piece, None) for piece in gray_pieces])
        target_kp, target_des = sift.detectAndCompute(gray_target, None)
        kp, des = sift.detectAndCompute(gray_pieces[i], None)

        # plot keypoints for a single test piece
        test_piece = cv.drawKeypoints(pieces[i], kp, None)
        cv.imshow("First Piece Keypoints", test_piece)
        print("Number of keypoints in first piece: ", len(kp))

        distances = calculate_distances_matrix(des, target_des)
        best_matches = find_best_matches(distances, threshold=0.75)
        print("Number of best matches: ", len(best_matches))
        #plot_best_matches(pieces[1], target, best_matches, kp, target_kp)

        # implement RANSAC to find the best transformation matrix
        best_error = np.inf
        all_src_pts = np.float32([kp[match[0]].pt for match in best_matches]).reshape(-1, 1, 2)
        all_dst_pts = np.float32([target_kp[match[1]].pt for match in best_matches]).reshape(-1, 1, 2)
        for _ in range(1000):
            # select 3 random points
            random_matches = random.sample(best_matches, k=3)
            # calculate transformation matrix
            src_pts = np.float32([kp[match[0]].pt for match in random_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([target_kp[match[1]].pt for match in random_matches]).reshape(-1, 1, 2)
            M = cv.getAffineTransform(src_pts, dst_pts)
            # calculate the residuals for all the points
            transformed_points = cv.transform(all_src_pts, M)
            residuals = [euclidean_distance(tp, dp) for tp, dp in zip(transformed_points, all_dst_pts)]
            error = np.mean(residuals)
            # if the number of inliers is larger than the current best, update the best
            if error < best_error:
                best_error = error
                transform = M
            


        # apply the transformation matrix to the test piece
        transform = np.vstack((transform, np.array([0, 0, 1])))
        print("Transformation matrix: ", transform)
        transformed_piece = cv.warpPerspective(pieces[i], transform, (target_width, target_height), flags=cv.INTER_CUBIC)
        gray_transformed_piece = cv.warpPerspective(gray_pieces[i], transform, (target_width, target_height), flags=cv.INTER_CUBIC)
        #cv.imshow("Transformed Piece", transformed_piece)
        target[gray_target == 0] = transformed_piece[gray_target == 0]
        gray_target[gray_target == 0] = gray_transformed_piece[gray_target == 0]
        #target = cv.add(target, transformed_piece)

    cv.imshow("Final Image", target)






def main():
    solve_puzzle('puzzles/puzzle_affine_5')
    cv.waitKey(0)
    print('done')



if __name__ == '__main__':
    main()