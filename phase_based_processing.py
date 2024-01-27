"""
Phase Based Magnification Processing

Isaac Berrios
January 2024

"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pyramid_utils import build_level, \
                          build_level_batch, \
                          recon_level_batch


class PhaseBased():
     
    def __init__(self, 
                 sigma, 
                 transfer_function, 
                 phase_mag, 
                 attenuate, 
                 ref_idx, 
                 batch_size, 
                 device, 
                 eps=1e-6):
        """
            sigma - std dev of Amplitude Weighted Phase Blurring 
                    (use 0 for no blurring)
            transfer_function - Frequency Domain Bandpass Filter 
                                Transfer Function (array)
            phase_mag - Phase Magnification/Amplification factor
            attenuate - determines whether to attenuate other frequencies
            ref_idx - index of reference frame to compare local phase 
                      changes to (DC frame)
            batch_size - batch size for parrallelization
            device - "cuda" or "cpu"
            eps - offset to avoid division by zero
        """
        self.sigma = sigma
        self.transfer_function = transfer_function
        self.phase_mag = phase_mag
        self.attenuate = attenuate
        self.ref_idx = ref_idx
        self.batch_size = batch_size
        self.device = device
        self.eps = eps

        self.gauss_kernel = self.get_gauss_kernel()


    def get_gauss_kernel(self):
        """ Obtains Gaussian Kernel for Aplitude weighted Blurring 
            Inputs: None
            Outputs:
                gauss_kernel
            """
        # ensure ksize is odd or the filtering will take too long
        # see warnng in: https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html
        ksize = np.max((3, np.ceil(4*self.sigma) - 1)).astype(int)
        if ((ksize % 2) != 1):
            ksize += 1

        # get Gaussian Blur Kernel for reference only
        gk = cv2.getGaussianKernel(ksize=ksize, sigma=self.sigma)
        gauss_kernel = torch.tensor(gk @ gk.T).type(torch.float32) \
                                              .to(self.device) \
                                              .unsqueeze(0) \
                                              .unsqueeze(0)

        return gauss_kernel
    

    def process_single_channel(self, 
                               frames_tensor, 
                               filters_tensor, 
                               video_dft):
        """ Applies Phase Based Processing in the Frequency Domain 
            for single channel frames 
            Inputs:
                frames_tensor - tesnor of frames to process
                filters_tensor - tensor of Complex Steerable Filter components
                video_dft - tensor of DFT video frames
            Outputs:
                result_video - tensor of reconstructed video frames with 
                                amplified motion
            """
        num_frames, _, _ = frames_tensor.shape
        num_filters, h, w = filters_tensor.shape
        
        # allocate tensors for processing
        recon_dft = torch.zeros((num_frames, h, w), 
                                dtype=torch.complex64).to(self.device)
        phase_deltas = torch.zeros((self.batch_size, num_frames, h, w), 
                                   dtype=torch.complex64).to(self.device)

        for level in range(1, num_filters - 1, self.batch_size):

            # get batch indices
            idx1 = level
            idx2 = level + self.batch_size

            # get current filter batch
            filter_batch = filters_tensor[idx1:idx2]

            ## get reference frame pyramid and phase (DC)
            ref_pyr = build_level_batch(
                video_dft[self.ref_idx, :, :].unsqueeze(0), filter_batch)
            ref_phase = torch.angle(ref_pyr)

            ## Get Phase Deltas for each frame
            for vid_idx in range(num_frames):
                curr_pyr = build_level_batch(
                    video_dft[vid_idx, :, :].unsqueeze(0), filter_batch)

                # unwrapped phase delta
                _delta = torch.angle(curr_pyr) - ref_phase 

                # get phase delta wrapped to [-pi, pi]
                phase_deltas[:, vid_idx, :, :] = ((torch.pi + _delta) \
                                                  % 2*torch.pi) - torch.pi
                
            ## Temporally Filter the phase deltas
            # Filter in Frequency Domain and convert back to phase space
            phase_deltas = torch.fft.ifft(self.transfer_function \
                                          * torch.fft.fft(phase_deltas, dim=1),  
                                          dim=1).real

            ## Apply Motion Magnifications
            for vid_idx in range(num_frames):
                
                vid_dft = video_dft[vid_idx, :, :].unsqueeze(0)
                curr_pyr = build_level_batch(vid_dft, filter_batch)
                delta = phase_deltas[:, vid_idx, :, :].unsqueeze(1)

                ## Perform Amplitude Weighted Blurring
                if self.sigma != 0:
                    amplitude_weight = (torch.abs(curr_pyr) 
                                        + self.eps).unsqueeze(1)
                    
                    # Torch Functional Approach for convolutional filtering
                    weight = F.conv2d(input=amplitude_weight, 
                                      weight=self.gauss_kernel, 
                                      padding='same').squeeze(1)
                    
                    delta = F.conv2d(input=(amplitude_weight * delta), 
                                     weight=self.gauss_kernel, 
                                     padding='same').squeeze(1) 

                    # get weighted Phase Deltas
                    delta /= weight

                ## Modify phase variation
                modifed_phase = delta * self.phase_mag

                ## Attenuate other frequencies by scaling current magnitude  
                ## by normalized reference phase. This removed all phase
                ## changes except the banpdass filtered phases
                if self.attenuate:
                    curr_pyr = torch.abs(curr_pyr) \
                               * (ref_pyr/torch.abs(ref_pyr))

                ## apply modified phase to current level pyramid decomposition
                # if modified_phase = 0, then no change!
                curr_pyr = curr_pyr * torch.exp(1.0j*modifed_phase) 

                ## accumulate reconstruced levels
                recon_dft[vid_idx, :, :] += \
                    recon_level_batch(curr_pyr, filter_batch).sum(dim=0)


        ## add unchanged Low Pass Component for contrast
        # adding hipass seems to cause bad artifacts and leaving
        # it out doesn't seem to impact the overall quality
        
        # hipass = filters_tensor[0]
        lopass = filters_tensor[-1]

        ## add back lo and hi pass components
        for vid_idx in range(num_frames):
            # Get Pyramid Decompositions for Hi and Lo Pass Filters
            # curr_pyr_hi = build_level(video_dft[vid_idx, :, :], hipass)
            curr_pyr_lo = build_level(video_dft[vid_idx, :, :], lopass)

            # dft_hi = torch.fft.fftshift(torch.fft.fft2(curr_pyr_hi)) 
            dft_lo = torch.fft.fftshift(torch.fft.fft2(curr_pyr_lo))

            # accumulate reconstructed hi and lo components
            # recon_dft[vid_idx, :, :] += dft_hi*hipass
            recon_dft[vid_idx, :, :] += dft_lo*lopass


        ## Get Inverse DFT and remove from CUDA if applicable
        result_video = torch.fft.ifft2(
            torch.fft.ifftshift(recon_dft, dim=(1,2)), dim=(1,2)).real

        return result_video