
'''

CUDA_DEVICES_VISIBLE=0 python comparer.py \
    --dir1 NEW_RUNS/OG_model/mediacorp \
    --dir2 NEW_RUNS/VM_model/mediacorp \
    --out_dir NEW_RUNS/Differences/Test_Mediacorp/OGdiffVM/diffs

CUDA_DEVICES_VISIBLE=2 python comparer.py \
    --dir1 NEW_RUNS/jitter_aug_model/mediacorp \
    --dir2 NEW_RUNS/augBgrBase_model_s1/mediacorp \
    --out_dir NEW_RUNS/Differences/Test_Mediacorp/JITTERAUGvsBGRBASE_s1/diffs

CUDA_DEVICES_VISIBLE=1 python comparer.py \
    --dir1 NEW_RUNS/OG_model/mediacorp \
    --dir2 NEW_RUNS/IM_model/mediacorp \
    --out_dir NEW_RUNS/Differences/Test_Mediacorp/OGdiffIM/diffs

CUDA_DEVICES_VISIBLE=2 python comparer.py \
    --dir1 NEW_RUNS/VM_model/mediacorp \
    --dir2 NEW_RUNS/IM_model/mediacorp \
    --out_dir NEW_RUNS/Differences/Test_Mediacorp/VMdiffIM/diffs

CUDA_DEVICES_VISIBLE=3 python comparer.py \
    --dir1 NEW_RUNS/OG_model/panasonic \
    --dir2 NEW_RUNS/VM_model/panasonic \
    --out_dir NEW_RUNS/Differences/Train_Panasonic/OGdiffVM/diffs

CUDA_DEVICES_VISIBLE=0 python comparer.py \
    --dir1 NEW_RUNS/OG_model/panasonic \
    --dir2 NEW_RUNS/IM_model/panasonic \
    --out_dir NEW_RUNS/Differences/Train_Panasonic/OGdiffIM/diffs

CUDA_DEVICES_VISIBLE=1 python comparer.py \
    --dir1 NEW_RUNS/VM_model/panasonic \
    --dir2 NEW_RUNS/IM_model/panasonic \
    --out_dir NEW_RUNS/Differences/Train_Panasonic/VMdiffIM/diffs
'''

import argparse
import cv2
import os
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--dir1", type=str, required=True, help="Path to the first directory containing video files")
parser.add_argument("--dir2", type=str, required=True, help="Path to the second directory containing video files")
parser.add_argument("--out_dir", type=str, required=True, help="Path to the output directory")
args = parser.parse_args()

dir1 = args.dir1
dir2 = args.dir2
out_dir = args.out_dir

# Get a list of all the video files in the directories
video_files1 = [f for f in os.listdir(dir1)]
video_files2 = [f for f in os.listdir(dir2)]

video_files1.sort()
video_files2.sort()
print(video_files1[:5],video_files2[:5])
# Create the output directory if it does not exist
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for i, file in enumerate(tqdm(video_files2,desc= "Differentiating files")):
    if i == min(len(video_files1),len(video_files2)): break
    if file in os.listdir(out_dir):
        print("Skipping",file,"since it already exists")
        continue
    print("Current File:",file)

    file1_path = os.path.join(dir1, file)
    file2_path = os.path.join(dir2, file)
    out_file_path = os.path.join(out_dir, f"{file.split('.')[0]}.mp4")
    cap1 = cv2.VideoCapture(file1_path)
    cap2 = cv2.VideoCapture(file2_path)

    fps = cap1.get(cv2.CAP_PROP_FPS)
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # create the VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_file_path, fourcc, fps, (width, height))

    # iterate through the frames of the two input videos
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        # break the loop if any of the videos ends
        if not ret1 or not ret2: break
        diff = cv2.absdiff(frame1, frame2)
        out.write(diff)
    
    cap1.release()
    cap2.release()
    out.release()

print("Done!")


#-------------Specific files version-------------------------------------------------------------------------------------------------------------
# import cv2
# import os
# import numpy as np
# from tqdm import tqdm

# video1 = 'runs/OG_model/arbitrary09.mp4'
# video2 = 'runs/OG_model/09_composition.mp4'
# out_dir = 'runs/difference'

# out_file_path = os.path.join(out_dir, "arbitrary09_diff.mp4")
# cap1 = cv2.VideoCapture(video1)
# cap2 = cv2.VideoCapture(video2)

# fps = cap1.get(cv2.CAP_PROP_FPS)
# width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # create the VideoWriter object to save the output video
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(out_file_path, fourcc, fps, (width, height))

# # iterate through the frames of the two input videos
# while True:
#     ret1, frame1 = cap1.read()
#     ret2, frame2 = cap2.read()

#     # break the loop if any of the videos ends
#     if not ret1 or not ret2: break
#     diff = cv2.absdiff(frame1, frame2)
#     out.write(diff)

# cap1.release()
# cap2.release()
# out.release()

# print("Done!")
