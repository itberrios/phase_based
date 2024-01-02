import numpy as np
import torch

## ==========================================================================================

def get_polar_grid(h, w):
    """ Obtains Angle and Radius of Polar grid
        Inputs:
            h, w - input image height and width
        Outputs:
            angle - Angluar Component of Polar Grid
            radius - Radial component of Polar Grid
    """
    # Get grid for cosine ramp function
    h2 = h//2
    w2 = w//2

    # Get normalized frequencies (same as fftfreq) [-1, 1)
    # modulus remainders to account for odd numbers
    wx, wy = np.meshgrid(np.arange(-w2, w2 + (w % 2))/w2, 
                         np.arange(-h2, h2 + (h % 2))/h2)

    # angular component
    angle = np.arctan2(wy, wx)

    # radial component
    radius = np.sqrt(wx**2 + wy**2)
    radius[h2][w2] = radius[h2][w2 - 1] # remove zero component

    return angle, radius

## ==========================================================================================
## Filter Crop functions

def get_filter_crops(filter_in):
    """ Obtains indices that correspond to non-zero filter values and a
        180 degree rotated rotated copy of FILTER and all indices in between two
        non-zero indices
        Inputs:
            filter_in - input frequency domain filter
        Outputs:
            row_idx - index to crop along the rows (height)
            col_idx - index to crop along the cols (width)
        """
    h, w = filter_in.shape
    above_zero = filter_in > 1e-10

    # rows
    dim1 = np.sum(above_zero, axis=1)
    dim1 = np.where(dim1 > 0)[0]
    row_idx = np.clip([dim1.min() - 1, dim1.max() + 1], 0, h)

    # cols
    dim2 = np.sum(above_zero, axis=0)
    dim2 = np.where(dim2 > 0)[0]
    col_idx = np.clip([dim2.min() - 1, dim2.max() + 1], 0, w)

    return np.concatenate((row_idx, col_idx))


def get_cropped_filters(filters, crops):
    """ Obtains list of cropped filters 
        Inputs:
            filters - list of filters
            crops - list of crop indices
        Outputs:
            cropped_filters - list of cropped filters
        """
    cropped_filters = []
    for (filt, crop) in zip(filters, crops):
        cropped_filters.append(filt[crop[0]:crop[1], crop[2]:crop[3]])

    return cropped_filters

## ==========================================================================================
## Pyramid Level functions

def build_level(image_dft, filt):
    """ Builds a single level of the Pyramid Decomposition 
        Inputs:
            image_dft - Full scale Image DFT
            filt - Frequency Domain Filter
        Output:
            Pyramid decomposition as the level specified by filt
        """
    return torch.fft.ifft2(torch.fft.ifftshift(image_dft * filt))


def recon_level(pyr_level, filt):
    """ Reconstructs a single Level of the Pyramid Decomposition 
        Only Valid for Sub bands in Complex Pyramids (not lo or hi 
        pass filter bands)
        Inputs:
            pyr_level - Pyramid decomposition as the level specified by filt
            filt - Frequency Domain Filter (same shape as pyr_level)
        Outputs:
            recon_dft - Reconstructed DFT of the input Pyramid Level
        """
    return 2.0 * torch.fft.fftshift(torch.fft.fft2(pyr_level)) * filt


def build_level_batch(image_dft, filt):
    """ Builds a single level of the Pyramid Decomposition 
        Inputs:
            image_dft - Full scale Image DFT batch (b, n, m)
            filt - Frequency Domain Filter batch (b, n, m)
        Output:
            Pyramid decomposition as the level specified by filt
        """
    return torch.fft.ifft2(torch.fft.ifftshift(image_dft * filt, dim=(1,2)), dim=(1,2))


def recon_level_batch(pyr_level, filt):
    """ Reconstructs a single Level of the Pyramid Decomposition 
        Only Valid for Sub bands in Complex Pyramids (not lo or hi 
        pass filter bands)
        Inputs:
            pyr_level - Pyramid decomposition as the level specified by filt (b, n, m)
            filt - Frequency Domain Filter (same shape as pyr_level) (b, n, m)
        Outputs:
            recon_dft - Reconstructed DFT of the input Pyramid Level
        """
    return 2.0 * torch.fft.fftshift(torch.fft.fft2(pyr_level, dim=(1,2)), dim=(1,2)) * filt

