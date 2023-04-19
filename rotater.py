'''
<Filter for videos that need rotation i.e. exiftool -rotate [video] does not result in 0>
1. Use 
    ffmpeg -i test_vids/01.mp4 -metadata:s:v rotate=0 -codec copy -y  01.mp4
   to remove metadata
2. Use 
    ffmpeg -i 2.mp4 -vf "transpose=1" 2Rot.mp4
   to rotate the video
3. Use cv2, exiftool and ffmpeg to confirm the orientation
4. Overwrite original files or create a new folder for the correctly oriented videos
'''

import json
import subprocess
from tqdm import tqdm
import cv2
import os

input_dir = "mediacorp_vids" #"panasonic_vids"
rotated_output_directory = "Rotated/MEDIACORP"

def check_rotation(video_path):
    metadata_output = subprocess.check_output(['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', video_path])
    metadata = json.loads(metadata_output)
    rotation = None
    for stream in metadata['streams']:
        if stream['codec_type'] == 'video' and 'tags' in stream and 'rotate' in stream['tags']:
            rotation = int(stream['tags']['rotate'])
    return rotation

for file in tqdm(sorted(os.listdir(input_dir)),"Videos Progress"):
    filename, _ = os.path.splitext(file)
    if file in os.listdir(rotated_output_directory): 
        print("Skipping",file)
        continue
    print("\n---------------------------------------\nCurrent file:",file)
    
    video_path = os.path.join(input_dir, file)
    rotation=check_rotation(video_path)
    print(f"Rotation value for {file}: {rotation}\n---------------------------------------\n")
    if rotation is None:
        os.rename(os.path.join(input_dir, file), os.path.join(rotated_output_directory, file))
        print("############################")
        print("No rotation was needed")
        print("############################")
    else:
        subprocess.check_call(['ffmpeg', '-i', video_path, '-metadata:s:v', 'rotate=0', '-codec', 'copy', '-y', "temp.mp4"])
        rotatedVidPath = os.path.join(rotated_output_directory,file)
        rotation = check_rotation(video_path)
        
        if rotation==90:
            subprocess.check_call(['ffmpeg', '-i', "temp.mp4", '-vf', "transpose=1", rotatedVidPath])
        elif rotation==180:
            subprocess.check_call(['ffmpeg', '-i', "temp.mp4", '-vf', "vflip", rotatedVidPath])
        else:
            subprocess.check_call(['ffmpeg', '-i', "temp.mp4", '-vf', "transpose=2", rotatedVidPath])
        
        print("############################")
        print("Video rotated from {} to {}".format(rotation,check_rotation(rotatedVidPath)))
        print("############################")

        cap = cv2.VideoCapture(rotatedVidPath)
        ret, frame = cap.read()
        
        cv2.imshow("Frame", frame)
        cv2.waitKey(0)
        cap.release()
        cv2.destroyAllWindows()