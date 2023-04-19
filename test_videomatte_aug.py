from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.augmentation_aug import TrainFrameSampler
from dataset.videomatte_aug import VideoMatteDataset, VideoMatteTrainAugmentation
from train_config import DATA_PATHS

# Define the dataset and the data loader
dataset = VideoMatteDataset(
                            videomatte_dir="../matting-data/PanasonicMatte/train_aug",
                            background_image_dir=DATA_PATHS['background_images']['train'],
                            background_video_dir=DATA_PATHS['background_videos']['train'],
                            size=2048,
                            seq_length=6,
                            seq_sampler=TrainFrameSampler(),
                            transform=VideoMatteTrainAugmentation(2048))

dataloader = DataLoader(
                dataset=dataset,
                batch_size=1,
                num_workers=1,
                pin_memory=True)

# Test the data loader
for fgrs, phas, bgrs in tqdm(dataloader,delay=10):
    continue
