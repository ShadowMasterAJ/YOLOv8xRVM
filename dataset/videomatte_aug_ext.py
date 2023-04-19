import os
import random
from torch.utils.data import Dataset
from PIL import Image
from .augmentation_aug_ext import MotionAugmentation

import matplotlib.pyplot as plt

class VideoMatteDataset(Dataset):
    def __init__(self,
                 videomatte_dir,
                 background_image_dir,
                 background_video_dir,
                 size,
                 seq_length,
                 seq_sampler,
                 transform=None):
        self.background_image_dir = background_image_dir
        self.background_image_files = os.listdir(background_image_dir)
        self.background_video_dir = background_video_dir
        self.background_video_clips = sorted(os.listdir(background_video_dir))
        self.background_video_frames = [sorted(os.listdir(os.path.join(background_video_dir, clip)))
                                        for clip in self.background_video_clips]
        
        self.videomatte_dir = videomatte_dir
        self.videomatte_clips = sorted(os.listdir(os.path.join(videomatte_dir, 'fgr')))[:1]
        self.videomatte_frames = [sorted(os.listdir(os.path.join(videomatte_dir, 'fgr', clip))) 
                                  for clip in self.videomatte_clips]
        self.videomatte_idx = [(clip_idx, frame_idx) 
                                for clip_idx in sorted(range(len(self.videomatte_clips))) 
                                for frame_idx in range(0, len(self.videomatte_frames[clip_idx]), seq_length)]

        self.size = size
        self.seq_length = seq_length
        self.seq_sampler = seq_sampler
        self.transform = transform
        self.frameToBgrDict = {
                                '0017':'durian1',
                                '0018':'durian2',
                                '0019':'durian3'
                                }
        self.special_clip=None

    def __len__(self):
        return len(self.videomatte_idx)
    
    def __getitem__(self, idx):
        fgrs, phas, files = self._get_videomatte(idx)
        if random.random() < 0.5:  bgrs, bases = self._get_random_image_background()
        else: bgrs,bases,paths = self._get_random_video_background(files)               

        if self.transform:
            fgrs, phas, bgrs, bases = self.transform(fgrs, phas, bgrs, bases, self.special_clip)
            
            fgrs = [fgr.permute(1,2,0) for fgr in fgrs]
            phas = [pha.permute(1,2,0) for pha in phas]
            bgrs = [bgr.permute(1,2,0) for bgr in bgrs]
            bases = [base.permute(1,2,0) for base in bases]
            
            print(fgrs[0].shape,phas[0].shape,bgrs[0].shape,bases[0].shape)
            
            for i in range(len(bgrs)):
                # create a boolean mask for the black background pixels in the bgr image
                mask = (bgrs[i] == 0).all(axis=-1)
                # replace the black background pixels with corresponding pixels from the base image
                bgrs[i][mask] = bases[i][mask]
            
            comps = [bgrs[i] * phas[i] for i in range(len(phas))]
            _, ax = plt.subplots(4, len(fgrs), figsize=(15,10))
            for i in range(len(fgrs)):
                
                # display the first image in the first subplot
                ax[0,i].imshow(fgrs[i])
                ax[0,i].set_title(f'Fgr {i+1}')
                ax[0,i].axis('off')

                # display the second image in the second subplot
                ax[1,i].imshow(phas[i])
                ax[1,i].set_title(f'Pha {i+1}')
                ax[1,i].axis('off')

                ax[2,i].imshow(bgrs[i])
                ax[2,i].set_title(f'Bgr {i+1}')
                ax[2,i].axis('off')
    
                ax[3,i].imshow(comps[i])
                ax[3,i].set_title(f'Comp {i+1}')
                ax[3,i].axis('off')
                # plt.savefig('aug_plots/augmentations {}.jpg'.format(i+1))
 
            plt.subplots_adjust(
                                top=0.94,
                                bottom=0.01,
                                left=0.008,
                                right=0.992,
                                hspace=0.2,
                                wspace=0.015
                            )
            plt.suptitle("Augmentations")

            plt.show()
            plt.close()


            return fgrs, phas, bgrs

        # return fgrs, phas, bgrs
    
    def _get_random_image_background(self):
        with Image.open(os.path.join(self.background_image_dir, random.choice(self.background_image_files))) as bgr:
            bgr = self._downsample_if_needed(bgr.convert('RGB'))
        bgrs = [bgr] * self.seq_length
        
        with Image.open(os.path.join(self.background_image_dir, random.choice(self.background_image_files))) as base:
            base = self._downsample_if_needed(base.convert('RGB'))
        bases = [base] * self.seq_length
        
        return bgrs, bases
    def _get_random_video_background(self,files):
        if self.special_clip:
            if random.random() < 1:
                # print(self.background_video_clips,len(self.background_video_clips),'durian1' in self.background_video_clips)
                clip_idx = self.background_video_clips.index(self.frameToBgrDict[self.special_clip])
                print("USING SPECIAL BGR FOR SPECIAL VID: {} | CLIP: {}".format(self.special_clip,self.background_video_clips[clip_idx]))
            else:
                available_clips = [clip for clip in self.background_video_clips[:-3] if clip not in ['durian1', 'durian2', 'durian3']]
                clip_idx = random.choice(range(len(available_clips)))
                print("USING NORMAL BGR FOR SPECIAL VID: {} | CLIP: {}".format(self.special_clip,self.background_video_clips[clip_idx]))

        else:
            available_clips = [clip for clip in self.background_video_clips[:-3] if clip not in ['durian1', 'durian2', 'durian3']]
            clip_idx = random.choice(range(len(available_clips)))
            clip_idx_base = random.choice(range(len(available_clips)))
            
        frame_count = len(self.background_video_frames[clip_idx])
        
        frame_idx = random.choice(range(max(1, frame_count - self.seq_length)))
        frame_idx_base = random.choice(range(max(1, frame_count - self.seq_length)))
        clip = self.background_video_clips[clip_idx]
        clip_base = self.background_video_clips[clip_idx_base]
        
        if not self.special_clip and clip in ['durian1', 'durian2', 'durian3']:
            print("\n=========================================\nERROR: CHOSEN SPECIAL BGR FOR NORMAL VID\n=========================================\n")
            return
        bgrs = []
        paths = []
        bases = []
        
        if self.special_clip and clip in self.frameToBgrDict.values():
            for f in files:
                paths.append(f)            
                file_name = os.path.splitext(os.path.basename(f))[0]
                frame_idx_t = int(file_name) % frame_count
                frame = self.background_video_frames[clip_idx][frame_idx_t]
                bgrs.append(self._downsample_if_needed(Image.open(os.path.join(self.background_video_dir, clip, frame)).convert('RGB')))

        else:
            for i in self.seq_sampler(self.seq_length):
                frame_idx_t = frame_idx + i
                frame = self.background_video_frames[clip_idx][frame_idx_t % frame_count]
                with Image.open(os.path.join(self.background_video_dir, clip, frame)) as bgr:
                    bgr = self._downsample_if_needed(bgr.convert('RGB'))
                bgrs.append(bgr)
        
        for i in self.seq_sampler(self.seq_length):
                frame_idx_t = frame_idx_base + i
                frame = self.background_video_frames[clip_idx_base][frame_idx_t % frame_count]
                with Image.open(os.path.join(self.background_video_dir, clip_base, frame)) as bgr:
                    base = self._downsample_if_needed(bgr.convert('RGB'))
                bases.append(base)
        return bgrs, bases, paths
    
    def _get_videomatte(self, idx):
        clip_idx, frame_idx = self.videomatte_idx[idx]
        clip = self.videomatte_clips[clip_idx]
        #print("Chosen Video Clip",clip)
        if clip in self.frameToBgrDict:
            self.special_clip = clip
            print("WORKING WITH SPECIAL VID:",self.special_clip)
        # else:
        #     print("WORKING WITH NORMAL VID")

        frame_count = len(self.videomatte_frames[clip_idx])
        fgrs, phas, paths = [], [], []
        for i in self.seq_sampler(self.seq_length):
            frame = self.videomatte_frames[clip_idx][(frame_idx + i) % frame_count]
            paths.append(frame)
            #print("FGR PATH {}: {}".format(frame,os.path.join(self.videomatte_dir, 'fgr', clip, frame)))
            with Image.open(os.path.join(self.videomatte_dir, 'fgr', clip, frame)) as fgr, \
                 Image.open(os.path.join(self.videomatte_dir, 'pha', clip, frame)) as pha:
                    fgr = self._downsample_if_needed(fgr.convert('RGB'))
                    pha = self._downsample_if_needed(pha.convert('L'))
            fgrs.append(fgr)
            phas.append(pha)
        return fgrs, phas, paths
    
    def _downsample_if_needed(self, img):
        w, h = img.size

        if min(w, h) > self.size:
            scale = self.size / min(w, h)
            w = int(scale * w)
            h = int(scale * h)
            img = img.resize((w, h))

        return img

class VideoMatteTrainAugmentation(MotionAugmentation):
    def __init__(self, size):
        super().__init__(
            size=size,
            prob_fgr_affine=0.3,
            prob_bgr_affine=0.3,
            prob_noise=0.1,
            prob_color_jitter=0.3,
            prob_grayscale=0.02,
            prob_sharpness=0.1,
            prob_blur=0.02,
            prob_hflip=0.5,
            prob_pause=0.03,
        )

class VideoMatteValidAugmentation(MotionAugmentation):
    def __init__(self, size):
        super().__init__(
            size=size,
            prob_fgr_affine=0,
            prob_bgr_affine=0,
            prob_noise=0,
            prob_color_jitter=0,
            prob_grayscale=0,
            prob_sharpness=0,
            prob_blur=0,
            prob_hflip=0,
            prob_pause=0,
        )
