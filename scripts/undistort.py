#!/usr/bin/env python
import cv2
import numpy as np
import os
import glob
import argparse


def readNpy(target):
    try:
        with(open(target, 'r')) as f:
            return np.loadtxt(f)
    except:
        return None

def loadModel(inputdir):
    kpath = os.path.join(inputdir, 'cameramatrix.txt')
    K = readNpy(kpath)
    dpath = os.path.join(inputdir, 'dist.txt')
    D = readNpy(dpath)
    return K,D

def undistort(images, datadir, inputdir, outputdir, fisheye):
    if datadir is None:
        d = os.path.join(inputdir, 'data')
    else:
        d = datadir
    K,D = loadModel(d)

    for imgpath in images:
        img = cv2.imread(imgpath)
        h,w = img.shape[:2]
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(K, D, (w,h), 0, (w,h))
        if fisheye:
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), newcameramtx, (w,h), cv2.CV_16SC2)
            undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        else:
            undistorted_img = cv2.undistort(img, K, D, None, newcameramtx)

        imgpath = os.path.join(outputdir, "undistorted_" + os.path.basename(os.path.dirname(imgpath)), os.path.basename(imgpath))
        os.makedirs(os.path.dirname(imgpath), exist_ok=True)
        cv2.imwrite(imgpath, undistorted_img)

if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--pathIn", help="path to video")
    a.add_argument("--pathOut", help="path to images", required=False)
    a.add_argument("--dataIn", help="path to data dir", required=False)
    a.add_argument("--fisheye", help="Whether to use fish eye calibration", 
                   action='store_true', required=False)
    args = a.parse_args()
    print(args)
    if args.pathIn is None:
        args.pathIn = os.getcwd()
    if args.pathOut is None:
        args.pathOut = args.pathIn
    os.makedirs(args.pathOut, exist_ok=True)
    images = glob.glob(os.path.join(args.pathIn,'*.png'))
    if len(images) == 0:
        print(f"No images found at {args.pathIn}")
        exit(1)
    undistort(images, args.dataIn, args.pathIn, args.pathOut, args.fisheye)
    print("done")