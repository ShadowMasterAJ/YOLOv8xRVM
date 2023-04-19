import os

# Define global variables for easy, medium and hard directories
EASY_DIR = '/home/jazz/Matting/matting-data/Mediacorp/November PTCs raw/easy'
MEDIUM_DIR = '/home/jazz/Matting/matting-data/Mediacorp/November PTCs raw/medium'
HARD_DIR = '/home/jazz/Matting/matting-data/Mediacorp/November PTCs raw/difficult'


easy_videos = [video.split('.')[0] for video in os.listdir(os.path.join(EASY_DIR))]
medium_videos = [video.split('.')[0] for video in os.listdir(os.path.join(MEDIUM_DIR))]
hard_videos = [video.split('.')[0] for video in os.listdir(os.path.join(HARD_DIR))]



def create_folders(another_dir_path):
    os.makedirs(os.path.join(another_dir_path, 'easy'), exist_ok=True)
    os.makedirs(os.path.join(another_dir_path, 'medium'), exist_ok=True)
    os.makedirs(os.path.join(another_dir_path, 'hard'), exist_ok=True)
    
    for video in os.listdir(another_dir_path):
        video_name = video.split('.')[0]
        if video_name in easy_videos:
            os.rename(os.path.join(another_dir_path, video), os.path.join(another_dir_path, 'easy', video))
        elif video_name in medium_videos:
            os.rename(os.path.join(another_dir_path, video), os.path.join(another_dir_path, 'medium', video))
        elif video_name in hard_videos:
            os.rename(os.path.join(another_dir_path, video), os.path.join(another_dir_path, 'hard', video))

create_folders('ALL_MEDIACORP/MOBILENETV3')
create_folders('ALL_MEDIACORP/MOTION_JITTER')
create_folders('ALL_MEDIACORP/RESNET50')