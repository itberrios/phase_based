"""
PyTorch implementation of Phase Based Motion Magnification

Isaac Berrios
January 2024

Current approach loads all frames into memory so this won't work for large videos
This is just a demo.

For the Batch size selection, we try to avoid the need to zero pad batchsize, since
it leads to undesireable behavior in the processing. It is easier to taylor both the 
batchsize and scale factor such that we don't need zero padding.

TODO: ensure selected Batch Size is compatible with the number of filters

 Sources:
    Papers: 
        - http://people.csail.mit.edu/nwadhwa/phase-video/phase-video.pdf
        - https://www.cns.nyu.edu/pub/eero/simoncelli95b.pdf
        - http://www.cns.nyu.edu/pub/eero/portilla99-reprint.pdf
    Code: 
        - http://people.csail.mit.edu/nwadhwa/phase-video/
        - https://github.com/LabForComputationalVision/matlabPyrTools
        - https://github.com/LabForComputationalVision/pyrtools
    Misc:
        - https://rafat.github.io/sites/wavebook/advanced/steer.html
        - http://www.cns.nyu.edu/~eero/steerpyr/
        - https://www.cns.nyu.edu/pub/lcv/simoncelli90.pdf
        - http://www.cns.nyu.edu/~eero/imrep-course/Slides/07-multiScale.pdf
"""

import os
import sys
import datetime
import re
import argparse
import numpy as np
from PIL import Image
import cv2
import torch

from steerable_pyramid import SteerablePyramid, SuboctaveSP
from phase_based_processing import PhaseBased
from phase_utils import *

## ==========================================================================================
## constants
EPS = 1e-6 # factor to avoid division by 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


## ==========================================================================================
## construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# Basic Args
ap.add_argument("-v", "--video_path", type=str, required=True,
	help="path to input video")
ap.add_argument("-a", "--phase_mag", type=float, default=25.0, required=True,
    help="Phase Magnification Factor")
ap.add_argument("-lo", "--freq_lo", type=float, required=True,
    help="Low Frequency cutoff for Temporal Filter")
ap.add_argument("-hi", "--freq_hi", type=float, required=True,
    help="High Frequency cutoff for Temporal Filter")
ap.add_argument("-n", "--colorspace", type=str, default="luma1",
    choices={"luma1", "luma3", "gray", "yiq", "rgb"},
    help="Defines the Colorspace that the processing will take place in")

# Pyramid Args
ap.add_argument("-p", "--pyramid_type", type=str, default="half_octave",
    choices={"full_octave", 
             "half_octave", 
             "smooth_half_octave", 
             "smooth_quarter_octave"},
    help="Complex Steerable Pyramid Type")

# Phase Processing Args
ap.add_argument("-s", "--sigma", type=float, default=0.0,
    help="Guassian Kernel Std Dev for amplitude weighted filtering, \n" \
         "If 0, then amplitude weighted filtering will not be performed")
ap.add_argument("-t", "--attenuate", type=bool, default=False,
	help="Attenuates other frequencies if True")
ap.add_argument("-fs", "--sample_frequency", type=float, default=-1.0,
	help="Video sample frequency, defaults to sample frequency from input "  \
         "video if input is less than or equal to zero. Video is " \
         "reconstructed with detected sample frequency")

# Misc Args
ap.add_argument("-r", "--reference_index", type=int, default=0,
    help="Reference Index for DC frame \
         (i.e. reference frame for phase changes)")
ap.add_argument("-c", "--scale_factor", type=float, default=1.0,
    help="Scales down image to rpeserve memory")
ap.add_argument("-b", "--batch_size", type=int, default=2,
    help="Batch size for CUDA parallelization")
ap.add_argument("-d", "--save_directory", type=str, default="",
    help="Save directory for output video or GIF, if False outputs \
          are placed in the same location as the input video")
ap.add_argument("-gif", "--save_gif", type=bool, default=False,
    help="Determines whether to save GIF of results")


## ==========================================================================================
## start main program

if __name__ == '__main__':

    ## Default use commandline args 
    ## --> Comment this out to manually input args in script
    # args = vars(ap.parse_args())

    # Optional: Pass arguments directly in script
    # --> Comment this out to receive args from commandline
    args = vars(ap.parse_args(
        ["--video_path",       "videos/guitar.avi", # "videos/eye.avi", # "videos/crane_crop.avi", 
         "--phase_mag",        "25.0", # "25.0", 
         "--freq_lo",          "72", # "30", # "0.20", 
         "--freq_hi",          "92", # "50", # "0.25", 
         "--colorspace",       "luma3",
         "--pyramid_type",     "half_octave",
         "--sigma",            "2.0", # "5.0"
         "--attenuate",        "True", # "False",
         "--sample_frequency", "600", # "500", # "-1.0", # This is generally not needed
         "--reference_index",  "0",
         "--scale_factor",     "0.75", # "1.0"
         "--batch_size",       "4",
         "--save_directory",   "",
         "--save_gif",         "True"
         ]))

    # args = vars(ap.parse_args(
    #     ["--video_path",       "videos/crane_crop.avi", 
    #      "--phase_mag",        "25.0", 
    #      "--freq_lo",          "0.20", 
    #      "--freq_hi",          "0.25", 
    #      "--colorspace",       "luma3",
    #      "--pyramid_type",     "half_octave",
    #      "--sigma",            "5.0",
    #      "--attenuate",        "True", # "False",
    #      "--sample_frequency", "-1.0", # This is generally not needed
    #      "--reference_index",  "0",
    #      "--scale_factor",     "1.0",
    #      "--batch_size",       "4",
    #      "--save_directory",   "",
    #      "--save_gif",         "False"
    #      ]))

    
    ## Parse Args   
    video_path       = args["video_path"]
    phase_mag        = args["phase_mag"]
    freq_lo          = args["freq_lo"]
    freq_hi          = args["freq_hi"]
    colorspace       = args["colorspace"]
    pyramid_type     = args["pyramid_type"]
    sigma            = args["sigma"]
    attenuate        = args["attenuate"]
    sample_frequency = args["sample_frequency"]
    ref_idx          = args["reference_index"]
    scale_factor     = args["scale_factor"]
    batch_size       = args["batch_size"]
    save_directory   = args["save_directory"]
    save_gif         = args["save_gif"]

    ## ======================================================================================
    ## start the clock once the args are received
    tic = cv2.getTickCount()

    ## ======================================================================================
    ## Process input filepaths
    if not os.path.exists(video_path):
        print(f"\nInput video path: {video_path} not found! exiting \n")
        sys.exit()
        
    if not save_directory:
        save_directory = os.path.dirname(video_path)
    elif not os.path.exists(save_directory):
        save_directory = os.path.dirname(video_path)
        print(f"\nSave Directory not found, " \
               "using default input video directory instead \n")
    
    video_name = re.search("\w*(?=\.\w*)", video_path).group()
    video_output = f"{video_name}_{colorspace}_{int(phase_mag)}x.mp4"
    video_save_path = os.path.join(save_directory, video_output)

    print(f"\nProcessing {video_name} " \
          f"and saving results to {video_save_path} \n")
    print(f"Device found: {DEVICE} \n")

    ## ======================================================================================
    ## Get frames and sample rate (fs) from input video

    # get forward and inverse colorspace functions
    # inverse colorspace obtains frames back in BGR representation
    if colorspace == "luma1":
        colorspace_func = lambda x: bgr2yiq(x)[:, :, 0]
        inv_colorspace = lambda x: cv2.cvtColor(
            cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1), 
            cv2.COLOR_GRAY2BGR) 
        
    elif colorspace == "luma3" or colorspace == "yiq":
        colorspace_func = bgr2yiq
        inv_colorspace = lambda x: cv2.cvtColor(
            cv2.normalize(yiq2rgb(x), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3), 
            cv2.COLOR_RGB2BGR)
        
    elif colorspace == "gray":
        colorspace_func = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        inv_colorspace = lambda x: cv2.cvtColor(
            cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1), 
            cv2.COLOR_GRAY2BGR)  
        
    elif colorspace == "rgb":
        colorspace_func = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        inv_colorspace = lambda x: cv2.cvtColor(
            cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3), 
            cv2.COLOR_RGB2BGR)
    
    # get scaled video frames in proper colorspace and sample frequency fs
    frames, video_fs = get_video(args["video_path"], 
                                 args["scale_factor"],
                                 colorspace_func)

    ## ======================================================================================
    ## Prepare for processing

    # get reference frame info
    ref_frame = frames[ref_idx]
    h, w = ref_frame.shape[:2]

    # video length
    num_frames = len(frames)

    # get sample frequency fs, use input sample freuqency if valid
    if sample_frequency > 0.0:
        fs = sample_frequency
        print(f"Detected Sample Frequency: {video_fs} \n")
        print(f"Sample Frequency overriden with input!: fs = {fs} \n")
    else:
        fs = video_fs
        print(f"Detected Sample Frequency: fs = {fs} \n")

    # Get Bandpass Filter Transfer function
    transfer_function = bandpass_filter(freq_lo, 
                                        freq_hi, 
                                        fs, 
                                        num_frames, 
                                        DEVICE)

    # Get Complex Steerable Pyramid Object
    max_depth = int(np.floor(np.log2(np.min(np.array((w, h))))) - 2)
    if pyramid_type == "full_octave":
        csp = SteerablePyramid(depth=max_depth, 
                               orientations=4, 
                               filters_per_octave=1, 
                               twidth=1.0, 
                               complex_pyr=True)
        
    elif pyramid_type == "half_octave":
        csp = SteerablePyramid(depth=max_depth, 
                               orientations=8, 
                               filters_per_octave=2, 
                               twidth=0.75, 
                               complex_pyr=True)
        
    elif pyramid_type == "smooth_half_octave":
        csp = SuboctaveSP(depth=max_depth, 
                          orientations=8, 
                          filters_per_octave=2, 
                          cos_order=6, 
                          complex_pyr=True) 
        
    elif pyramid_type == "smooth_quarter_octave":
        csp = SuboctaveSP(depth=max_depth, 
                          orientations=8, 
                          filters_per_octave=4, 
                          cos_order=6, 
                          complex_pyr=True) 

    # get Complex Steerabel Pyramid Filters
    filters, crops = csp.get_filters(h, w, cropped=False)
    filters_tensor = torch.tensor(np.array(filters)).type(torch.float32) \
                                                    .to(DEVICE)
    
    # TODO: ensure selected Batch Size is compatible with the number of filters
    # we don't want to rely on zero padding
    if (filters_tensor.shape[0] % batch_size) != 0:
        print(f"WARNING! Selected Batch size: {batch_size} might " \
              f"not be compatible with the number of " \
              f"Filters: {filters_tensor.shape[0]}! \n")


    # Compute DFT for all Video Frames
    frames_tensor = torch.tensor(np.array(frames)).type(torch.float32) \
                                                  .to(DEVICE)

    ## ======================================================================================
    ## Begin Motion Magnification processing

    print(f"Performing Phase Based Motion Magnification \n")

    pb = PhaseBased(sigma, transfer_function, phase_mag, attenuate, 
                    ref_idx, batch_size, DEVICE, EPS)

    if colorspace == "yiq" or colorspace == "rgb":
        # process each channel individually
        result_video = torch.zeros_like(frames_tensor).to(DEVICE)
        for c in range(frames_tensor.shape[-1]):
            video_dft = get_fft2_batch(frames_tensor[:, :, :, c]).to(DEVICE)
            result_video[:, :, :, c] = \
                pb.process_single_channel(frames_tensor[:, :, :, c], 
                                          filters_tensor, 
                                          video_dft)

    elif colorspace == "luma3":
        # process single Luma channel and add back to full color image
        result_video = frames_tensor.clone()
        video_dft = get_fft2_batch(frames_tensor[:, :, :, 0]).to(DEVICE)
        result_video[:, :, :, 0] = \
            pb.process_single_channel(frames_tensor[:, :, :, 0], 
                                      filters_tensor, 
                                      video_dft)

    else:
        # process single channel
        video_dft = get_fft2_batch(frames_tensor).to(DEVICE)
        result_video = pb.process_single_channel(frames_tensor, 
                                                 filters_tensor,
                                                 video_dft)

    # remove from CUDA and covnert to numpy
    result_video = result_video.cpu().numpy()

    ## ======================================================================================
    ## Process results

    ## get stacked side-by-side comparison frames
    og_h = int(h/scale_factor)
    og_w = int(w/scale_factor)
    middle = np.zeros((og_h, 3, 3)).astype(np.uint8)

    stacked_frames = []

    for vid_idx in range(num_frames):
        
        # get BGR frames
        bgr_frame = inv_colorspace(frames[vid_idx])
        bgr_processed = inv_colorspace(result_video[vid_idx])

        # resize to original shape
        bgr_frame = cv2.resize(bgr_frame, (og_w, og_h))
        bgr_processed = cv2.resize(bgr_processed, (og_w, og_h))

        # stack frames
        stacked = np.hstack((bgr_frame, 
                             middle, 
                             bgr_processed))

        stacked_frames.append(stacked)


    ## ======================================================================================
    ## make video
    # get width and height for stacked video frames
    sh, sw, _ = stacked_frames[-1].shape

    # save to mp4
    out = cv2.VideoWriter(video_save_path,
                          cv2.VideoWriter_fourcc(*'MP4V'), 
                          int(np.round(video_fs)), 
                          (sw, sh))
    
    for frame in stacked_frames:
        out.write(frame)

    out.release()
    del out

    print(f"Result video saved to: {video_save_path} \n")

    ## ======================================================================================
    ## make GIF if desired
    if save_gif:
        
        # replace video extension with ".gif"
        gif_save_path = re.sub("\.\w+(?<=\w)", ".gif", video_save_path)

        print(f"Saving GIF to: {gif_save_path} \n")

        # size back down for GIF
        sh = int(sh*scale_factor)
        sw = int(sw*scale_factor)

        # accumulate PIL image objects
        pil_images = []
        for img in stacked_frames:
            img = cv2.cvtColor(cv2.resize(img, (sw, sh)), cv2.COLOR_BGR2RGB)
            pil_images.append(Image.fromarray(img))

        # create GIF
        pil_images[0].save(gif_save_path, 
                           format="GIF", 
                           append_images=pil_images, 
                           save_all=True, 
                           duration=50, # duration that each frame is displayed
                           loop=0)

    ## ======================================================================================
    ## end of processing
        
    # get time elapsed in Hours : Minutes : Seconds
    toc = cv2.getTickCount()
    time_elapsed = (toc - tic) / cv2.getTickFrequency()
    time_elapsed = str(datetime.timedelta(seconds=time_elapsed))

    print("Motion Magnification processing complete! \n")
    print(f"Time Elapsed (HH:MM:SS): {time_elapsed} \n")
    