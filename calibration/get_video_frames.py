import sys
import argparse
import glob
import cv2
import os

def extractImages(pathIn, pathOut, name):
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    success = True
    while success:
        try:
            vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))    # added this line 
            success,image = vidcap.read()
            print (f'Frame at second {count}')
            cv2.imwrite(os.path.join(pathOut,  name + f"_frame_{count}.jpg"), image)
            count = count + 1
        except cv2.error as e:
            print(e)
            if e.err == "!_src.empty()":
                break # break the while loop

if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--pathIn", help="path to video")
    a.add_argument("--pathOut", help="path to images", required=False)
    args = a.parse_args()
    print(args)
    if args.pathIn is None:
        args.pathIn = os.getcwd()
    if args.pathOut is None:
        args.pathOut = args.pathIn
    os.makedirs(args.pathOut, exist_ok=True)
    videos = glob.glob(os.path.join(args.pathIn,'*.mp4'))
    for videofile in videos:
        print(f"Processing {videofile}")
        name = os.path.basename(videofile).replace('.mp4','')
        outdir = os.path.join(args.pathOut, name)
        os.makedirs(outdir, exist_ok=True)
        extractImages(videofile, outdir, name)
    print("done")