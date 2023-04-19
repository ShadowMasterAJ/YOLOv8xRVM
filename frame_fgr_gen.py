import cv2
import os

from tqdm import tqdm

# directory paths
dir1 = 'Rotated/MEDIACORP'
dir2 = 'mediacorp_new_videos_frames_pha/pha_vids'

# output directories
out_dir1 = 'mediacorp_new_videos_frames_pha/frames'
out_dir2 = 'mediacorp_new_videos_frames_pha/pha'

# get video filenames from directories
videos1 = [f for f in os.listdir(dir1) if len(f)>6]
videos2 = [f for f in os.listdir(dir2)]
videos1.sort()
videos2.sort()
print(videos1,len(videos1))
print('')
print(videos2,len(videos2))

# loop through videos and save frames as images
for i in tqdm(range(len(videos1)), desc="Video"):
    # create directory with video name in output directory
    video_name = os.path.splitext(videos1[i])[0]
    video_out_dir1 = os.path.join(out_dir1, video_name)
    video_out_dir2 = os.path.join(out_dir2, video_name)
    os.makedirs(video_out_dir1, exist_ok=True)
    os.makedirs(video_out_dir2, exist_ok=True)

    # read video from dir1
    cap1 = cv2.VideoCapture(os.path.join(dir1, videos1[i]))
    # read video from dir2
    cap2 = cv2.VideoCapture(os.path.join(dir2, videos2[i]))
    frame = 1

    # loop through frames of both videos
    while True:
        # read frame from video1
        ret1, frame1 = cap1.read()
        # read frame from video2
        ret2, frame2 = cap2.read()

        # check if frames were successfully read
        if not ret1 or not ret2:
            break

        # write frame1 to out_dir1 with same video name
        cv2.imwrite(os.path.join(video_out_dir1, f'{frame}.jpg'), frame1)
        # write frame2 to out_dir2 with same video name
        cv2.imwrite(os.path.join(video_out_dir2, f'{frame}.jpg'), frame2)
        frame += 1

    # release video capture
    cap1.release()
    cap2.release()