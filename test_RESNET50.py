import os
import torch
from inference import convert_video
from model.model import MattingNetwork
from tqdm import tqdm

input_dir = '/home/jazz/Matting/matting-data/Mediacorp/cut_vids'
final_name = 'ALL_MEDIACORP/RESNET50'

def run_model_on_directory(input_dir, final_name):
    model = MattingNetwork("resnet50").eval().cuda()  # or "resnet50"

    model.load_state_dict(torch.load("checkpoint/rvm_resnet50.pth"))
    inputList = sorted(os.listdir(input_dir))
    selected = inputList #[:13]+inputList[21:22]+inputList[38:39]
    
    print(inputList,selected)
    for file in tqdm(selected,desc="Video Progress"):
        filename= os.path.splitext(file)[0]
        if f'{filename}_composition.mp4' in os.listdir(final_name):
            print("Skipping",file)
            continue
        output_composition = f'{final_name}/{filename}.mp4'
        print("Current Video:",file)

        convert_video(
            model,                                          # The loaded model, can be on any device (cpu or cuda).
            input_source= os.path.join(input_dir, file),    # A video file or an image sequence directory.
            downsample_ratio=0.25,                          # [Optional] If None, make downsampled max size be 512px.
            output_type='video',                            # Choose "video" or "png_sequence"
            output_composition=output_composition,          # File path if video; directory path if png sequence.
            # output_alpha=output_composition,              # [Optional] Output the raw alpha prediction.
            # output_foreground="runs/foreground2.mp4",     # [Optional] Output the raw foreground prediction.
            output_video_mbps=10,                           # Output video mbps. Not needed for png sequence.
            seq_chunk=12,                                   # Process n frames at once for better parallelism.
            # num_workers=1,                                # Only for image sequence input. Reader threads.
            progress=True,                                  # Print conversion progress.
            )
        print("-----------------------------------------------------------\nFINISHED:",file)

run_model_on_directory(input_dir, final_name)
