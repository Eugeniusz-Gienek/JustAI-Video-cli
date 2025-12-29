import cv2, cython, math, shutil, os, time, sys, traceback, subprocess, errno, glob, json, warnings
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from skimage import metrics
from SSIM_PIL import compare_ssim
from PIL import Image
import queue
from tqdm import tqdm
from queue import Queue

from justai_video_processor_logs import *

from justai_video_processor import video_info

def image_encoder(img, device, preprocess, model):
    img1 = Image.fromarray(img).convert('RGB')
    img1 = preprocess(img1).unsqueeze(0).to(device)
    img1 = model.encode_image(img1)
    return img1

def generateScore(image1, image2, device, preprocess, model):
    img1 = image_encoder(image1, device, preprocess, model)
    img2 = image_encoder(image2, device, preprocess, model)
    cos_scores = util.pytorch_cos_sim(img1, img2)
    score = round(float(cos_scores[0][0])*100, 2)
    return score

def init_clip(use_gpu: cython.bint=False, clip_model_name='ViT-B-16-plus-240', clip_pretrained='laion400m_e32'):
    device:          str = ''
    model                = None
    preprocess           = None
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=UserWarning
        )
    global util
    import open_clip
    from sentence_transformers import util
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    model, _, preprocess = open_clip.create_model_and_transforms(clip_model_name, pretrained=clip_pretrained)
    model.to(device)
    return (device, preprocess, model)

def build_frames_map(
        source_video_file: str,
        calc_method: str='ssim',
        use_gpu: cython.bint=False,
        dest_video_temp: str='',
        output_to: str='screen',
        output_to_file: str='',
        output_to_file_no_frame_numbers: cython.bint=False,
        output_to_file_json: str='',
        clip_model_name='ViT-B-16-plus-240',
        clip_pretrained='laion400m_e32',
        frames_map_show_progress:cython.bint=False,
        tqbar1=None,
        disable_logs: cython.bint=False,
        quiet: cython.bint=False,
        loud: cython.bint=False
        ):
    device:                 str          = ''
    output:                 str          = ''
    frames_map_log_file:    str          = ''
    use_gpu_txt:            str          = 'gpu' if use_gpu else 'cpu'
    frames:                 cython.int   = 0
    frame:                  cython.int   = 0
    width:                  cython.int   = 0
    height:                 cython.int   = 0
    pid1:                   cython.int   = 0
    pid2:                   cython.int   = 0
    fps:                    cython.float = 0
    success:                cython.bint  = True
    firstRun:               cython.bint  = True
    doTqbar:                cython.bint  = False
    output_data:            list         = []
    ssim_score:             cython.float = 0
    metric_val:             cython.float = 0
    clip_comp_val:          cython.float = 0
    res_val:                cython.float = 0
    videoCap                             = None
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=UserWarning
        )
    if quiet:
        loud = False
    if not disable_logs:
        frames_map_log_file = init_logs(os.path.join(dest_video_temp, 'logs'), 'frames_map')
    
    if loud or (not disable_logs):
        log_append_print(f'[ Frames Map ] Building frames map for the video file {source_video_file}.', log_file=frames_map_log_file, disable_logs=disable_logs, loud=loud)
        if output_to == 'file':
            if output_to_file != '':
                log_append_print(f'[ Frames Map ] Output will be stored in plain text format in a file {output_to_file}.', log_file=frames_map_log_file, disable_logs=disable_logs, loud=loud)
            if output_to_file_json != '':
                log_append_print(f'[ Frames Map ] Output will be stored in json format in a file {output_to_file_json}.', log_file=frames_map_log_file, disable_logs=disable_logs, loud=loud)
        elif output_to == 'screen':
            log_append_print(f'[ Frames Map ] Output will be output to a screen.', log_file=frames_map_log_file, disable_logs=disable_logs, loud=loud)
        else:
            log_append_print(f'[ Frames Map ] Output will be not created.', log_file=frames_map_log_file, disable_logs=disable_logs, loud=loud)
    try:
        if output_to == 'file':
            # Why here? In order to not perform the operation when there is no way to save a file. Let it break before the execution.
            # Sure, there is also a situation possible if accidentally started this operation and this will erase existing file.
            # Well, either that or that...
            if output_to_file != '':
                with open(output_to_file, "w") as f:
                    f.write('')
                    f.close()
            if output_to_file_json != '':
                with open(output_to_file_json, "w") as f:
                    f.write('')
                    f.close()
        (frames_in_source_video, source_video_width, source_video_height, source_video_fps) = video_info(source_video_file, False, False)
        
        videoCap    = cv2.VideoCapture(source_video_file)
        frames      = int(videoCap.get(cv2.CAP_PROP_FRAME_COUNT))
        width       = int(videoCap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height      = int(videoCap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps         = videoCap.get(cv2.CAP_PROP_FPS)
        
        src_image1  = None
        src_image2  = None
        image1_gray = None
        image2_gray = None
        
        hist_img1   = None
        hist_img2   = None
        
        img1        = None
        img2        = None
        
        preprocess  = None
        model       = None
        cos_scores  = None
        
        if not disable_logs:
            frames_map_log_file = init_logs(os.path.join(dest_video_temp, 'logs'), 'frames_map')
        
        if loud or (not disable_logs):
            log_append_print(f'Selected frames map technique is: {calc_method}', log_file=frames_map_log_file, disable_logs=disable_logs, loud=loud)
        if calc_method == 'clip':
            #device, preprocess, model = init_clip(use_gpu, clip_model_name, clip_pretrained)
            os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            import open_clip
            from sentence_transformers import util
            device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
            model, _, preprocess = open_clip.create_model_and_transforms(clip_model_name, pretrained=clip_pretrained)
            model.to(device)
        
        if frames_map_show_progress and (not loud) and (output_to != 'screen'):
            if tqbar1 is None:
                tqbar1 = tqdm(desc='Processed', unit='frame(s)', total=frames_in_source_video, leave=True)
            doTqbar = True
        
        while success:
            success, src_image = videoCap.read()
            if (loud or (not disable_logs)) and frames_map_show_progress:
                log_append_print(f'[ Frames Map ][{calc_method},{use_gpu_txt}] Processing frame nr. {frame} out of {frames_in_source_video}', log_file=frames_map_log_file, disable_logs=disable_logs, loud=loud)
            if success:
                frame += 1
                if firstRun:
                    if loud or (not disable_logs):
                        log_append_print(f'[ Frames Map ][{calc_method},{use_gpu_txt}]  First run.', log_file=frames_map_log_file, disable_logs=disable_logs, loud=loud)
                    src_image1  = src_image
                    src_image2  = src_image
                    firstRun    = False
                    ssim_score  = 1.0
                    output_data.append(ssim_score)
                    if output_to == 'screen':
                        print(f'{frame}: {ssim_score}')
                        log_append(f'{frame}: {ssim_score}', log_file=frames_map_log_file, disable_logs=disable_logs)
                    elif output_to != 'no':
                        if output_to_file_no_frame_numbers:
                            output += f"{ssim_score}\n"
                        else:
                            output += f"{frame}: {ssim_score}\n"
                    res_val = ssim_score
                    if doTqbar:
                        tqbar1.update(1)
                    if calc_method == 'clip':
                        img1        = Image.fromarray(src_image1).convert('RGB')
                        img1        = preprocess(img1).unsqueeze(0).to(device)
                        img1        = model.encode_image(img1)
                        img2        = img1
                    elif calc_method == 'ssim':
                        img1        = Image.fromarray(src_image1).convert('RGB')
                        img2        = img1
                    elif calc_method == 'ssim_':
                        image1_gray = cv2.cvtColor(src_image1, cv2.COLOR_BGR2GRAY)
                        image2_gray = image1_gray
                else:
                    src_image1  = src_image2
                    src_image2  = src_image
                    if calc_method == 'ssim':
                        img1        = img2
                        img2        = Image.fromarray(src_image2).convert('RGB')
                        ssim_score  = compare_ssim(img1, img2, GPU=use_gpu)
                        output_data.append(ssim_score)
                        res_val = ssim_score
                        if output_to == 'screen':
                            print(f'{frame}: {ssim_score}')
                            log_append(f'{frame}: {ssim_score}', log_file=frames_map_log_file, disable_logs=disable_logs)
                        elif output_to != 'no':
                            if output_to_file_no_frame_numbers:
                                output += f"{ssim_score}\n"
                            else:
                                output += f"{frame}: {ssim_score}\n"
                    elif calc_method == 'ssim_':
                        image1_gray = image2_gray
                        image2_gray = cv2.cvtColor(src_image2, cv2.COLOR_BGR2GRAY)
                        ssim_score  = metrics.structural_similarity(image1_gray, image2_gray, full=True)[0]
                        output_data.append(ssim_score)
                        res_val = ssim_score
                        if output_to == 'screen':
                            print(f'{frame}: {ssim_score}')
                            log_append(f'{frame}: {ssim_score}', log_file=frames_map_log_file, disable_logs=disable_logs)
                        elif output_to != 'no':
                            if output_to_file_no_frame_numbers:
                                output += f"{ssim_score}\n"
                            else:
                                output += f"{frame}: {ssim_score}\n"
                    elif calc_method == 'histogram':
                        hist_img1 = cv2.calcHist([src_image1], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
                        hist_img1[255, 255, 255] = 0  #ignore all white pixels
                        cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                        hist_img2 = cv2.calcHist([src_image2], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
                        hist_img2[255, 255, 255] = 0  #ignore all white pixels
                        cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                        metric_val = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_CORREL)
                        output_data.append(metric_val)
                        res_val = metric_val
                        if output_to == 'screen':
                            print(f'{frame}: {metric_val}')
                            log_append(f'{frame}: {metric_val}', log_file=frames_map_log_file, disable_logs=disable_logs)
                        elif output_to != 'no':
                            if output_to_file_no_frame_numbers:
                                output += f"{metric_val}\n"
                            else:
                                output += f"{frame}: {metric_val}\n"
                    elif calc_method == 'clip':
                        img1            = img2
                        img2            = Image.fromarray(src_image2).convert('RGB')
                        img2            = preprocess(img2).unsqueeze(0).to(device)
                        img2            = model.encode_image(img2)
                        cos_scores      = util.pytorch_cos_sim(img1, img2)
                        clip_comp_val   = float(cos_scores.detach()[0][0])
                        output_data.append(clip_comp_val)
                        res_val = clip_comp_val
                        if output_to == 'screen':
                            print(f'{frame}: {clip_comp_val}')
                            log_append(f'{frame}: {clip_comp_val}', log_file=frames_map_log_file, disable_logs=disable_logs)
                        elif output_to != 'no':
                            if output_to_file_no_frame_numbers:
                                output += f"{clip_comp_val}\n"
                            else:
                                output += f"{frame}: {clip_comp_val}\n"
                    if doTqbar:
                        tqbar1.update(1)
                if (loud or (not disable_logs)) and frames_map_show_progress:
                    log_append_print(f'[ Frames Map ][{calc_method},{use_gpu_txt}] Processed frame nr. {frame} out of {frames_in_source_video}. Similarity: {res_val}.', log_file=frames_map_log_file, disable_logs=disable_logs, loud=loud)
            else:
                if loud or (not disable_logs):
                    log_append_print(f'[ Frames Map ][{calc_method},{use_gpu_txt}] Unsuccessful attempt to receive frame.', log_file=frames_map_log_file, disable_logs=disable_logs, loud=loud)
        if doTqbar:
            tqbar1.close()
        if output_to == 'file':
            if output_to_file != '':
                if loud or (not disable_logs):
                    log_append_print(f'[ Frames Map ][{calc_method},{use_gpu_txt}] Saving output as a plain text format to file {output_to_file}', log_file=frames_map_log_file, disable_logs=disable_logs, loud=loud)
                with open(output_to_file, "w") as f:
                    f.write(output)
                    f.close()
            if output_to_file_json != '':
                if loud or (not disable_logs):
                    log_append_print(f'[ Frames Map ][{calc_method},{use_gpu_txt}] Saving output as a json to file {output_to_file_json}', log_file=frames_map_log_file, disable_logs=disable_logs, loud=loud)
                with open(output_to_file_json, "w") as f:
                    f.write(json.dumps(output_data))
                    f.close()
        if calc_method == 'clip':
            del model
    except KeyboardInterrupt:
        print('[ Frames Map ] Received keyboard interrupt. Stopping.')
        log_append('[ Frames Map ] Received keyboard interrupt. Stopping.', log_file=frames_map_log_file)
        output_data = None
    return output_data
