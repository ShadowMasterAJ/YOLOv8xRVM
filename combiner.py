'''
CUDA_VISIBLE_DEVICES=0 python combiner.py \
    --dir1 NEW_RUNS/OG_model/mediacorp \
    --diff_dir NEW_RUNS/Differences/Test_Mediacorp/OGdiffVM/diffs \
    --dir2 NEW_RUNS/VM_model/mediacorp \
    --all3 NEW_RUNS/Differences/Test_Mediacorp/OGdiffVM/merged

CUDA_VISIBLE_DEVICES=0 python combiner.py \
    --dir1 NEW_RUNS/jitter_aug_model/mediacorp \
    --dir2 NEW_RUNS/augBgrBase_model_s1/mediacorp \
    --diff_dir NEW_RUNS/Differences/Test_Mediacorp/JITTERAUGvsBGRBASE_s1/diffs \
    --all3 NEW_RUNS/Differences/Test_Mediacorp/JITTERAUGvsBGRBASE_s1/merged

CUDA_VISIBLE_DEVICES=1 python combiner.py \
    --dir1 NEW_RUNS/OG_model/mediacorp \
    --diff_dir NEW_RUNS/Differences/Test_Mediacorp/OGdiffIM/diffs \
    --dir2 NEW_RUNS/IM_model/mediacorp \
    --all3 NEW_RUNS/Differences/Test_Mediacorp/OGdiffIM/merged
    
CUDA_VISIBLE_DEVICES=2 python combiner.py \
    --dir1 NEW_RUNS/VM_model/mediacorp \
    --diff_dir NEW_RUNS/Differences/Test_Mediacorp/VMdiffIM/diffs \
    --dir2 NEW_RUNS/IM_model/mediacorp \
    --all3 NEW_RUNS/Differences/Test_Mediacorp/VMdiffIM/merged

CUDA_VISIBLE_DEVICES=3 python combiner.py \
    --dir1 NEW_RUNS/OG_model/panasonic \
    --diff_dir NEW_RUNS/Differences/Train_Panasonic/OGdiffVM/diffs \
    --dir2 NEW_RUNS/VM_model/panasonic \
    --all3 NEW_RUNS/Differences/Train_Panasonic/OGdiffVM/merged

CUDA_VISIBLE_DEVICES=0 python combiner.py \
    --dir1 NEW_RUNS/OG_model/panasonic \
    --diff_dir NEW_RUNS/Differences/Train_Panasonic/OGdiffIM/diffs \
    --dir2 NEW_RUNS/IM_model/panasonic \
    --all3 NEW_RUNS/Differences/Train_Panasonic/OGdiffIM/merged

CUDA_VISIBLE_DEVICES=1 python combiner.py \
    --dir1 NEW_RUNS/VM_model/panasonic \
    --diff_dir NEW_RUNS/Differences/Train_Panasonic/VMdiffIM/diffs \
    --dir2 NEW_RUNS/IM_model/panasonic \
    --all3 NEW_RUNS/Differences/Train_Panasonic/VMdiffIM/merged
'''

import argparse
import subprocess
import os
import cv2
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Merge video files from three directories.')
parser.add_argument('--dir1', type=str, help='Directory containing video files from the first model')
parser.add_argument('--diff_dir', type=str, help='Directory containing video files of the difference between the first and second models')
parser.add_argument('--dir2', type=str, help='Directory containing video files from the second model')
parser.add_argument('--all3', type=str, help='Output directory for merged video files')
args = parser.parse_args()

# Path of the directories
dir1 = args.dir1
diff_dir = args.diff_dir
dir2 = args.dir2
all3 = args.all3

# Get the list of video files in the directories
og_vids = sorted([os.path.join("Rotated/MEDIACORP", f.replace("_composition", "")) for f in os.listdir(dir2)])
dir1_vids = sorted([os.path.join(dir1, f) for f in os.listdir(dir2)])
diff_vids = sorted([os.path.join(diff_dir, f) for f in os.listdir(diff_dir)])
dir2_vids = sorted([os.path.join(dir2, f) for f in os.listdir(dir2)])

print('\n',og_vids[:5])
print('\n',dir1_vids[:5])
print('\n',diff_vids[:5])
print('\n',dir2_vids[:5])


# Create the output directory if it does not exist
if not os.path.exists(all3):
    os.makedirs(all3)

# Merge the videos using ffmpeg
for og_vid, video_a, video_b, video_c in tqdm(zip(og_vids, dir1_vids, diff_vids, dir2_vids),desc="Merging video"):
    
    if '_' in os.path.splitext(os.path.basename(video_a))[0][:2]:
        output_file_name = os.path.splitext(os.path.basename(video_a))[0][:2] + 'merged.mp4'
    else:
        if 'I' not in os.path.splitext(os.path.basename(video_a))[0][:2]:
            output_file_name = os.path.splitext(os.path.basename(video_a))[0][:2] + '_merged.mp4'
        else:
            output_file_name = os.path.splitext(os.path.basename(video_a))[0] + '_merged.mp4'

            
    output_file = os.path.join(all3, output_file_name)
    if output_file_name in os.listdir(all3): 
        print(f"{output_file_name} already exists. Skipping...")
        continue
    print("Now processing:",output_file_name)

    output_file = os.path.join(all3, output_file_name)
    command = f"ffmpeg -i {video_a} -i {video_b} -i {video_c} -filter_complex \"[1:v][0:v]scale2ref=oh*mdar:ih[1v][0v];[2:v][0v]scale2ref=oh*mdar:ih[2v][0v];[0v][1v][2v]hstack=3,scale=iw/2:-2:force_original_aspect_ratio=decrease\" -r 30 {output_file}"
    command2 = f"ffmpeg -i {og_vid} -i {video_a} -i {video_b} -i {video_c} -filter_complex \"[1:v][0:v]scale2ref=oh*mdar:ih[1v][0v];[2:v][0v]scale2ref=oh*mdar:ih[2v][0v];[3:v][0v]scale2ref=oh*mdar:ih[3v][0v];[0v][1v][2v][3v]hstack=4\" -r 60 {output_file}"
    subprocess.call(command2, shell=True)

