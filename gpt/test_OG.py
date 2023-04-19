import os
import warnings
import torch
from inference import convert_video
from model.model_test import MattingNetwork
from tqdm import tqdm


def run_model_on_directory(input_dir, final_name):
    model = MattingNetwork("mobilenetv3").eval().cuda()  # or "resnet50"

    model.load_state_dict(torch.load("checkpoint/stage1/epoch-19.pth"))

    # model = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3")
    print(os.listdir(input_dir))
    for file in tqdm(sorted(os.listdir(input_dir)),desc="Video Progress"):
        filename, _ = os.path.splitext(file)
        # if f'{filename}_composition.mp4' in os.listdir(final_name):
        #     print("Skipping",file)
        #     continue
        output_composition = f'{final_name}/{filename}_composition.mp4'
        print("Current Video:",file)

        convert_video(
            model,                                    # The loaded model, can be on any device (cpu or cuda).
            input_source= os.path.join(input_dir, file), # A video file or an image sequence directory.
            downsample_ratio=0.25,                    # [Optional] If None, make downsampled max size be 512px.
            output_type='video',                      # Choose "video" or "png_sequence"
            output_composition=output_composition,    # File path if video; directory path if png sequence.
            # output_alpha=output_composition,                # [Optional] Output the raw alpha prediction.
            # output_foreground="runs/foreground2.mp4",      # [Optional] Output the raw foreground prediction.
            output_video_mbps=10,                     # Output video mbps. Not needed for png sequence.
            seq_chunk=12,                             # Process n frames at once for better parallelism.
            # num_workers=1,                            # Only for image sequence input. Reader threads.
            progress=True,                            # Print conversion progress.
            )
        print("-----------------------------------------------------------\nFINISHED:",file)

# input_dir = 'Rotated/PANASONIC'
# final_name = 'NEW_RUNS/OG_model/panasonic'
# run_model_on_directory(input_dir, final_name)

input_dir = 'Rotated/MEDIACORP'
final_name = 'testing'
run_model_on_directory(input_dir, final_name)
