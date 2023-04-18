import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os



# functions
def solve_puzzle(directory):
    pieces_dir = os.path.join(directory, 'pieces')
    # read pieces
    pieces = []
    gray_pieces = []
    for filename in os.listdir(pieces_dir):
        img = cv.imread(os.path.join(pieces_dir, filename))
        pieces.append(img)
        gray_pieces.append(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
    
    sift = cv.SIFT_create()
    # find keypoints and descriptors
    kp = []
    des = []
    for piece in gray_pieces:
        k, d = sift.detectAndCompute(piece, None)
        kp.append(k)
        des.append(d)

    # plot keypoints for a single test piece
    test_piece = cv.drawKeypoints(pieces[0], kp[0], None)
    plt.imshow(test_piece)
    plt.show()


def main():
    solve_puzzle('puzzles/puzzle_affine_4')
    print('done')



if __name__ == '__main__':
    main()