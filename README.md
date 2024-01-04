# phase_based
PyTorch implementation of [Phase Based Motion Magnification](http://people.csail.mit.edu/nwadhwa/phase-video/phase-video.pdf). It is based off of MATLAB source that can be found [here](http://people.csail.mit.edu/nwadhwa/phase-video/), the input videos can also be found at this location. The PyTorch implementation is much faster than a numpy implementation even without a GPU.

The [main notebook](https://github.com/itberrios/phase_based/blob/main/motion_amplification_pytorch.ipynb) contains a detailed hands-on overview of the Motion Magnification Algorithm. The main script is called motion_magnification.py and
can be called from the commandline. Or alternatively the arguments can be input directly in the script.

## Applying Motion Magnification

The following commandline arguments produce the following GIF: <br>
``` Python motion_magnification.py -v videos/crane_crop.avi -a 25 -lo 0.2 -hi 0.25 -n luma3 -p half_octave -s 5.0 -b 4 -c 0.7 -gif True ``` 

![crane_crop_luma3_25x](https://github.com/itberrios/phase_based/assets/60835780/83cebe8d-eafa-4342-b5c1-2a9cc13ea458)

### Arguments:

- "--video_path",       "videos/crane_crop.avi", 
- "--phase_mag",        "25.0", 
- "--freq_lo",          "0.20", 
- "--freq_hi",          "0.25", 
- "--colorspace",       "luma3",
- "--pyramid_type",     "half_octave",
- "--sigma",            "5.0",
- "--attenuate",        "True", # "False",
- "--sample_frequency", "-1.0", # This is generally not needed
- "--reference_index",  "0",
- "--scale_factor",     "1.0",
- "--batch_size",       "4",
- "--save_directory",   "",
- "--save_gif",         "False"

