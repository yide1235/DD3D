import cv2 as cv
import numpy as np
import os
import glob
import argparse

def calibrateFromImages(images, outputdir, fisheye=False):
    # Defining the dimensions of checkerboard
    CHECKERBOARD = (4,6)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 40, 0.001)

    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = [] 
    
    winSize = (5, 5)
    zeroZone = (-1, -1)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_COUNT, 40, 0.001)
    
    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    imgoutdir = os.path.join(outputdir, 'points')
    os.makedirs(imgoutdir, exist_ok=True)
    # Extracting path of individual image stored in a given directory
    # images = glob.glob(os.path.join(basedir, '*.jpg'))
    print(f"Finding corners for {len(images)} images")
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, 
                                        cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)
        
        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display 
        them on the images of checker board
        """
        if ret == True:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            # Calculate the refined corner locations
            corners2 = cv.cornerSubPix(gray, corners, winSize, zeroZone, criteria)
            imgpoints.append(corners2)
            
            # Draw and display the corners
            img = cv.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        
        # cv.imshow('img',img)
        # cv.waitKey(0)
        cv.imwrite(os.path.join(imgoutdir, os.path.basename(fname)),img)

    # cv.destroyAllWindows()

    h,w = img.shape[:2]
    #Parameters for fisheye
    calibration_flags = cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv.fisheye.CALIB_CHECK_COND+cv.fisheye.CALIB_FIX_SKEW
    """
    Performing camera calibration by 
    passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the 
    detected corners (imgpoints)
    """
    print(f"Calibrating from {len(objpoints)} images")
    if not fisheye:
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
            mean_error += error
        print( "total error: {}".format(mean_error/len(objpoints)) )
    else:
        mtx = np.zeros((3, 3))
        dist = np.zeros((4, 1))
        count = len(objpoints)
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(count)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(count)]
        rms, _, _, _, _ = \
            cv.fisheye.calibrate(
                objpoints,
                imgpoints,
                gray.shape[::-1],
                mtx,
                dist,
                rvecs,
                tvecs,
                calibration_flags,
                # (cv.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
                (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 50, 1e-2)
            )
    #cv.solvePnPRansac(objectPoints, imagePoints, cameraMatrix, distCoeffs[, rvec[, tvec[, 
    # useExtrinsicGuess[, iterationsCount[, reprojectionError[, minInliersCount[, inliers[, flags]]]]]]]]) → rvec, tvec, inliers
    for i in range(0, len(objpoints)):
        ret, rvec, tvec, inliers = cv.solvePnPRansac(objpoints[i], imgpoints[i], mtx, dist, rvecs[i], tvecs[i], True)
        if(ret):
            print('---------------------------')
            print(rvec)
            print(tvec/max(tvec))
            print('---------------------------')


    rvecs = np.array(rvecs).squeeze()
    tvecs = np.array(tvecs).squeeze()
    dataoutput = os.path.join(outputdir, 'data')
    os.makedirs(dataoutput, exist_ok=True)

    print("Writing camera matrix")
    # print(mtx)
    with (open(os.path.join(dataoutput, 'cameramatrix.txt'), 'w')) as f:
        np.savetxt(f, mtx, header=f" camera matrix")
    print("Writing dist")
    # print(dist)
    with (open(os.path.join(dataoutput, 'dist.txt'), 'w')) as f:
        np.savetxt(f, dist, header=f"{dist.shape[0]} dist")
    
    print("Writing rvecs")
    # print(rvecs)
    with (open(os.path.join(dataoutput, 'rvecs.txt'), 'w')) as f:
        np.savetxt(f, rvecs, header=f"{rvecs.shape[0]}-by-{rvecs.shape[1]} rvecs")
    print("Writing tvecs")
    # print(tvecs)
    with (open(os.path.join(dataoutput, 'tvecs.txt'), 'w')) as f:
        np.savetxt(f, tvecs, header=f"{tvecs.shape[0]}-by-{tvecs.shape[1]} tvecs")


if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--pathIn", help="path to video")
    a.add_argument("--pathOut", help="path to images", required=False)
    a.add_argument("--fisheye", help="Whether to use fish eye calibration", 
                   action='store_true', required=False)
    args = a.parse_args()
    print(args)
    if args.pathIn is None:
        args.pathIn = os.getcwd()
    if args.pathOut is None:
        args.pathOut = args.pathIn
    os.makedirs(args.pathOut, exist_ok=True)
    images = glob.glob(os.path.join(args.pathIn,'**/*.jpg'))
    if len(images) == 0:
        print(f"No images found at {args.pathIn}")
        exit(1)
    calibrateFromImages(images, args.pathOut, args.fisheye)
    print("done")