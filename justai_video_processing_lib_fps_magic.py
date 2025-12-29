import math, cython

def lcm(a: cython.int, b: cython.int):
    return (a * b) // math.gcd(a, b)

def stats_data(a: cython.int, b: cython.int):
    smallest_divisible_number: cython.int = lcm(a, b)
    return (smallest_divisible_number,smallest_divisible_number/b - 1)

def performFpsMath(fps: cython.float, expected_fps: cython.float):
    """
    Rounded fps is needed to get a correct multiplier. There are smth like 23,97..... fps, we'll round that to 24
    """
    source_fps_rounded:         cython.int   = int(round(fps,0))
    dest_fps_rounded:           cython.int   = int(round(expected_fps,0))
    smallest_divisible_number:  cython.int   = dest_fps_rounded
    frames_to_remove:           cython.int   = 0
    frames_to_generate:         cython.int   = 0
    real_expected_fps:          cython.float = 0
    
    smallest_divisible_number, frames_to_remove = stats_data(source_fps_rounded, dest_fps_rounded)
    frames_to_generate  = int(smallest_divisible_number/source_fps_rounded - 1)
    real_expected_fps   = fps * (frames_to_generate + 1)
    return (source_fps_rounded, dest_fps_rounded, frames_to_generate, real_expected_fps, frames_to_remove)

def checkReasonableFPS(source_fps: cython.float,
        source_frames_amount_src: cython.int,
        threshold: cython.float = 0.05,
        strict_check: cython.bint = True,
        rnd_trigger: cython.int = 0):
    """
    @description: Checks if the source fps, which is supposed to be a float number,
        can be converted in the direct manner to the nearest integer number, so
        when additional frames are added in between - e.g. every X frames,
        the shift of video is not more than threshold (float number).
        Empirically I personally feel no difference when the frame floating is 0.05 seconds (1/20th of a second) -
        so that is the reason for a default value for threshold.
        Logic is as follows:
            1. Destination FPS is the nearest integer, rounded up, from original FPS (so called CEIL operation)
            2. We mutiply the amount of frames in a video by a Source FPS divided by Destination FPS number
               This way we get the Amount of Extra Frames which should be produced in order for the video
               to be the same time length as before. This time we just round the amount of frames -
               no need to round it explicitly up or down (well, if there is a need - set rnd_trigger appropriately).
            3. We divide the calculcated Amount of Extra Frames by the amount of frames in destination video.
               This is our resulting threshold - e.g. how differently the frames in the video might be positioned,
               they might be misplaced by this time in seconds in comparison to the original position.
               The comparison of this number to the provided threshold is returned as boolean.
                
    @params:
        source_fps: float - the source fps of a video.
        source_frames_amount_src: integer - the amount of frames in a video.
        threshold: float - the threshold of "frames floating" -
            e.g. a frame may be misplased by this amount of seconds later or earlier from original time of the frame
        rnd_trigger: integer - if zero then the amount of resulting frames in a video is just rounded to the nearest integer, otherwise -
            if negative number - rounded down, if positive number - rounded up.
        strict_check: boolean - if True, then the returned result is whether we are below threshold, if False - if we are below or equal.
    @return: boolean - are we within threshold or not.
    """
    source_frames_amount: cython.float  = float(source_frames_amount_src)
    dest_frames: cython.float           = source_frames_amount * float(math.ceil(source_fps)) / source_fps
    res_threshold: cython.float         = 0
    
    if rnd_trigger == 0:
        dest_frames = round(dest_frames)
    elif rnd_trigger > 0:
        dest_frames = math.ceil(dest_frames)
    elif rnd_trigger < 0:
        dest_frames = math.floor(dest_frames)
    dest_frames = float(dest_frames)
    res_threshold = (dest_frames - source_frames_amount) / dest_frames
    return res_threshold < threshold if strict_check else res_threshold <= threshold

def performFPSFloatingToIntegerMath(source_fps: cython.float, source_frames_amount_src: cython.int, rnd_trigger: cython.int = 0):
    """
    @description: Returns the amount of additional frames which have to be generated in order to convert float fps to the nearest integer (higher one).
    @params:
        source_fps: float - the source fps of a video.
        source_frames_amount_src: integer - the amount of frames in a video.
        rnd_trigger: integer - if zero then the amount of resulting frames in a video is just rounded to the nearest integer, otherwise -
            if negative number - rounded down, if positive number - rounded up.
    @return: integer - amount of frames to generate additionally.
    """
    source_frames_amount: cython.float  = float(source_frames_amount_src)
    dest_frames: cython.float           = source_frames_amount * float(math.ceil(source_fps)) / source_fps
    
    if rnd_trigger == 0:
        dest_frames = round(dest_frames)
    elif rnd_trigger > 0:
        dest_frames = math.ceil(dest_frames)
    elif rnd_trigger < 0:
        dest_frames = math.floor(dest_frames)
    return int(dest_frames - source_frames_amount)
