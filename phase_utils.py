"""
Utils for Phase Based Video Magnification

"""
import os
from PIL import Image
from glob import glob
import numpy as np
import cv2
import torch
from scipy import signal

## ==========================================================================================
## Color spaces
def rgb2yiq(rgb):
    """ Converts an RGB image to YIQ using FCC NTSC format.
        This is a numpy version of the colorsys implementation
        https://github.com/python/cpython/blob/main/Lib/colorsys.py
        Inputs:
            rgb - (N,M,3) rgb image
        Outputs
            yiq - (N,M,3) YIQ image
        """
    # compute Luma Channel
    y = rgb @ np.array([[0.30], [0.59], [0.11]])

    # subtract y channel from red and blue channels
    rby = rgb[:, :, (0,2)] - y

    i = np.sum(rby * np.array([[[0.74, -0.27]]]), axis=-1)
    q = np.sum(rby * np.array([[[0.48, 0.41]]]), axis=-1)

    yiq = np.dstack((y.squeeze(), i, q))
    
    return yiq


def bgr2yiq(bgr):
    """ Coverts a BGR image to float32 YIQ """
    # get normalized YIQ frame
    rgb = np.float32(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    yiq = rgb2yiq(rgb)

    return yiq


def yiq2rgb(yiq):
    """ Converts a YIQ image to RGB.
        Inputs:
            yiq - (N,M,3) YIQ image
        Outputs:
            rgb - (N,M,3) rgb image
        """
    r = yiq @ np.array([1.0, 0.9468822170900693, 0.6235565819861433])
    g = yiq @ np.array([1.0, -0.27478764629897834, -0.6356910791873801])
    b = yiq @ np.array([1.0, -1.1085450346420322, 1.7090069284064666])
    rgb = np.clip(np.dstack((r, g, b)), 0, 1)
    return rgb


## ==========================================================================================
## Video Utils
def get_video(video_path, scale_factor, colorspace_func=lambda x: x):
    """ Obtains frames from input video path 
        Inputs:
            video_path - path to video
            scale_factor - scale factor for frame sizes
            colorspace_func - function to map default BGR to 
        Outputs:
            frames - extracted video frames in desired colorspace and scale size
            fs - video sample rate
        """
    frames = [] # frames for processing
    cap = cv2.VideoCapture(video_path)

    # video sampling rate
    fs = cap.get(cv2.CAP_PROP_FPS)

    idx = 0

    while(cap.isOpened()):
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            break

        if idx == 0:
            og_h, og_w, _ = frame.shape
            w = int(og_w*scale_factor)
            h = int(og_h*scale_factor)

        # convert normalized uint8 BGR to the desired color space
        frame = colorspace_func(np.float32(frame/255))

        # append resized frame
        frames.append(cv2.resize(frame, (w, h)))

        idx += 1
        
        
    cap.release()
    cv2.destroyAllWindows()
    del cap

    return frames, fs

## ==========================================================================================
## GIF utils

def create_gif_from_images(save_path, image_path, ext):
    ''' creates a GIF from a folder of images
        Inputs:
            save_path (str) - path to save GIF
            image_path (str) - path where images are located
            ext (str) - extension of the images
        Outputs:
            None
        
        Update:
            Add functionality for multiple extensions
    '''
    image_paths = sorted(glob(os.path.join(image_path, f'*.{ext}')))
    pil_images = [Image.open(im_path ) for im_path in image_paths]
    pil_images[0].save(save_path, format='GIF', append_images=pil_images,
                       save_all=True, duration=45, loop=0)
    
def create_gif_from_numpy(save_path, images):
    ''' creates a GIF from numpy images
        Inputs:
            save_path (str) - path to save GIF
            image_path (str) - path where images are located
            ext (str) - extension of the images
        Outputs:
            None
        
        Update:
            Add functionality for multiple extensions
    '''
    pil_images = [Image.fromarray(img) for img in images]
    pil_images[0].save(save_path, format='GIF', append_images=pil_images,
                       save_all=True, duration=45, loop=0)

## ==========================================================================================
## Misc utils

def get_fft2_batch(tensor_in):
    return torch.fft.fftshift(torch.fft.fft2(tensor_in, dim=(1,2))).type(torch.complex64)


def bandpass_filter(freq_lo, freq_hi, fs, num_taps, device):
    """ Obtains Frequency Domain Transfer Function for Band pass filter 
        Inputs:
            freq_lo
            freq_hi
            fs
            num_frames - number of taps for filter
            device - CUDA or CPU
        Outputs:
            transfer_function - frequency domain transfer function
        """
    freq_lo = freq_lo / fs * 2
    freq_hi = freq_hi / fs * 2

    bandpass = signal.firwin(numtaps=num_taps, 
                             cutoff=[freq_lo, freq_hi], 
                             pass_zero=False)
    
    bandpass = torch.tensor(bandpass).to(device)
    transfer_function = torch.fft.fft(torch.fft.ifftshift(bandpass)).type(torch.complex64)
    transfer_function = torch.tile(transfer_function, [1, 1, 1, 1]).permute(0, 3, 1, 2)

    return transfer_function