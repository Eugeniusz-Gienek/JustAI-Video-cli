import cv2, cython, math, shutil, os, time, sys, traceback, subprocess, errno, glob, json, warnings
import multiprocessing as mp
import queue
from tqdm import tqdm
from queue import Queue
from fractions import Fraction
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from skimage import metrics
from SSIM_PIL import compare_ssim
from PIL import Image
from tempfile import TemporaryFile, TemporaryDirectory

from justai_video_processor_logs import *
from justai_video_processing_lib_fps_magic import *

default_frames_stack_mem_limit: cython.int = 100
this_app_codename = 'justai_video'

def available_interpolators():
    return ('film',)

def available_nsfio_methods():
    return ('duplicate','prolong',)

def available_tasks():
    return ('interpolate','build_frames_map',)

def available_frame_comparison_techniques():
    return ('histogram', 'ssim', 'clip', )

def silentremove(filename):
    try:
        try:
            os.remove(filename)
        except IsADirectoryError as e:
            pass
    except OSError as e:
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred

def pad_batch2(batch, align):
    width: cython.int
    height: cython.int
    height_to_pad: cython.int
    width_to_pad: cython.int
    align: cython.int
    # Omitted: batch, crop_region
    
    height, width = batch.shape[1:3]
    height_to_pad = (align - height % align) if height % align != 0 else 0
    width_to_pad = (align - width % align) if width % align != 0 else 0

    crop_region = [height_to_pad >> 1, width_to_pad >> 1, height + (height_to_pad >> 1), width + (width_to_pad >> 1)]
    batch = np.pad(batch, ((0, 0), (height_to_pad >> 1, height_to_pad - (height_to_pad >> 1)),
                           (width_to_pad >> 1, width_to_pad - (width_to_pad >> 1)), (0, 0)), mode='constant')
    return batch, crop_region

def load_image2(img1, align: cython.int=64):
    image = img1.astype(np.float32) / np.float32(255)
    image_batch, crop_region = pad_batch2(np.expand_dims(image, axis=0), align)
    return image_batch, crop_region

def video_info(video_file: str, do_print: cython.bint=True, do_exit: cython.bint=True):
    frames:     cython.int      = 0
    width:      cython.int      = 0
    height:     cython.int      = 0
    fps:        cython.float    = 0
    fps2:       cython.int      = 0
    fps3:       str             = ""
    
    videoCap    = cv2.VideoCapture(video_file)
    frames      = int(videoCap.get(cv2.CAP_PROP_FRAME_COUNT))
    width       = int(videoCap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(videoCap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps         = videoCap.get(cv2.CAP_PROP_FPS)
    if do_print:
        s = '' if frames == 1 else 's'
        fps2 = math.ceil(fps)
        fps3 = f'{fps}' if fps2 == math.floor(fps) else f'approx. {fps2} (precisely: {fps})'
        print(f"Video info: {frames} frame{s} in video, width: {width}px, height: {height}px, FPS: {fps3}.")
    videoCap.release()
    if do_exit:
        try:
            cv2.destroyAllWindows()
        except:
            pass
    return (frames, width, height, fps)

def workerFramesSupplier(
        threads_comm:queue.Queue,
        data_queue:queue.Queue,
        processes:list,
        source_video_file: str,
        dest_video_temp: str,
        use_gpu: cython.bint=False,
        frames_supply_force_use_gpu: cython.bint=False,
        nsfio_params: dict={'enabled':False},
        tqbar=None,
        frames_stack_mem_limit: cython.int=0,
        unloadPermuteToCPU: cython.bint = False,
        start_from: cython.int=0,
        max_frames: cython.int=0,
        disable_logs: cython.bint=False,
        quiet: cython.bint=False,
        loud: cython.bint=False
        ):
    doStop:                 cython.bint  = False
    success:                cython.bint  = True
    doTqbar:                cython.bint  = (tqbar is not None)
    data_queue_is_empty:    cython.bint  = False
    nsfi_c:                 cython.bint  = True
    nsfi_prepared:          cython.bint  = False
    firstRun:               cython.bint  = True
    stack_size:             cython.int   = 0
    i:                      cython.int   = 0
    global_counter1:        cython.int   = 0
    th_tout:                cython.int   = 3
    frames:                 cython.int   = 0
    width:                  cython.int   = 0
    height:                 cython.int   = 0
    fps:                    cython.float = 0
    nsfi:                   cython.float = 1
    threads_signal_msg:     str          = ""
    frames_supplier_log_file: str        = ""
    scores:                 list         = []
    src_images:             queue.Queue
    src_images_temp:        queue.Queue
    frames_map_data:        dict         = {'enabled':False,'src_image':None,'src_image_temp':None,'frames_map_file_json':''}
    
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
    
    device:                 str          = ''
    ssim_score:             cython.float = 0
    metric_val:             cython.float = 0
    clip_comp_val:          cython.float = 0
    videoCap                             = None
    
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning
            )
        if not disable_logs:
            frames_supplier_log_file = init_logs(os.path.join(dest_video_temp, 'logs'), 'frames_supplier')
        
        if loud or (not disable_logs):
            log_append_print('Worker Frames Supplier', log_file=frames_supplier_log_file, disable_logs=disable_logs, loud=loud)
            log_append_print(f'module name:{__name__}', log_file=frames_supplier_log_file, disable_logs=disable_logs, loud=loud)
            log_append_print(f'parent process: {os.getppid()}', log_file=frames_supplier_log_file, disable_logs=disable_logs, loud=loud)
            log_append_print(f'process id: {os.getpid()}', log_file=frames_supplier_log_file, disable_logs=disable_logs, loud=loud)
            log_append_print(f"Frames supplier will be:\n"
                    f"* keeping maximum of this amount of items in memory: {frames_stack_mem_limit}\n"
                    f"* supplying frames from this file: {source_video_file}\n"
                    f"* starting from frame: {start_from}\n"
                    f"* with max frames set to (if zero then disabled): {max_frames}\n", log_file=frames_supplier_log_file, disable_logs=disable_logs, loud=loud)
        
        videoCap    = cv2.VideoCapture(source_video_file)
        frames      = int(videoCap.get(cv2.CAP_PROP_FRAME_COUNT))
        width       = int(videoCap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height      = int(videoCap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps         = videoCap.get(cv2.CAP_PROP_FPS)
        
        if nsfio_params['enabled']:
            if nsfio_params['frames_map_data'] is not None:
                if loud or (not disable_logs):
                    log_append_print(f'[ Frames Supplier ] Using already prepared JSON with frames map.', log_file=frames_supplier_log_file, disable_logs=disable_logs, loud=loud)
                scores = nsfio_params['frames_map_data']
                nsfi_prepared = True
            elif nsfio_params['frames_map_file_json'] != '':
                with open(nsfio_params['frames_map_file_json']) as f:
                    scores = json.load(f)
                    f.close()
                nsfi_prepared = True
                if loud or (not disable_logs):
                    log_append_print(f'[ Frames Supplier ] Using already prepared JSON file with frames map.', log_file=frames_supplier_log_file, disable_logs=disable_logs, loud=loud)
            else:
                if nsfio_params['frames_comparison_technique'] == 'clip':
                    if loud or (not disable_logs):
                        log_append_print(f'[ Frames Supplier ] CLIP methodology for frames comparison.', log_file=frames_supplier_log_file, disable_logs=disable_logs, loud=loud)
                    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
                    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
                    import open_clip
                    from sentence_transformers import util
                    device = 'cuda' if (frames_supply_force_use_gpu and use_gpu and torch.cuda.is_available()) else 'cpu'
                    model, _, preprocess = open_clip.create_model_and_transforms(nsfio_params['clip_model_name'], pretrained=nsfio_params['clip_pretrained'])
                    model.to(device)
        
        if start_from != 0:
            global_counter1 = max(start_from, 0)
            if loud or not disable_logs:
                log_append_print(f'[ Frames Supplier ] Starting from a frame nr. {start_from+1} - it is set as a current one. We will skip first {global_counter1} frames.', log_file=frames_supplier_log_file, disable_logs=disable_logs, loud=loud)
            # Moving to the needed position
            for i in range(global_counter1):
                success, src_image = videoCap.read()
                if not success:
                    if loud or (not disable_logs):
                        log_append_print(f'[ Frames Supplier ] There was a problem during reading a frame from a file during a resume. Perhaps, we ran out of frames.', log_file=frames_supplier_log_file, disable_logs=disable_logs, loud=loud)
                    break
            if doTqbar:
                tqbar.n = global_counter1
                tqbar.last_print_n = global_counter1
                tqbar.refresh()
            if (max_frames > 0) and (global_counter1 > start_from + max_frames):
                success = False
            if not success:
                # We ran out of frames...
                doStop = True
                if doTqbar:
                    tqbar.refresh()
                for i in range(len(processes)-1):
                    threads_comm.put('no_frames', timeout=th_tout)
        
        while not doStop:
            try:
                if not threads_comm.empty():
                    threads_signal_msg = threads_comm.get_nowait()
                    if threads_signal_msg == 'stop':
                        doStop = True
                        threads_comm.put('stop', timeout=th_tout)
                    if threads_signal_msg == 'faint':
                        time.sleep(1)
                        threads_comm.put('faint', timeout=th_tout)
                        raise Exception("[ Frames Supplier ] Received info about another process dying")
                    if threads_signal_msg == 'no_processing':
                        for i in range(len(processes)-1):
                            threads_comm.put('no_processing', timeout=th_tout)
            except queue.Empty:
                pass
            if not doStop:
                success, src_image = videoCap.read()
                stack_size = data_queue.qsize()
                if (max_frames > 0) and (global_counter1 > start_from + max_frames):
                    success = False
                if not success:
                    if loud or (not disable_logs):
                        log_append_print(f'[ Frames Supplier ] Ran out of frames. Stopping supply.', log_file=frames_supplier_log_file, disable_logs=disable_logs, loud=loud)
                    doStop = True
                    if doTqbar:
                        tqbar.refresh()
                    for i in range(len(processes)-1):
                        threads_comm.put('no_frames', timeout=th_tout)
                elif (stack_size >= frames_stack_mem_limit) or data_queue.full():
                    if loud or (not disable_logs):
                        reason = f'Reached the stack limit frames amount ({frames_stack_mem_limit}) - {stack_size}.'
                        if (stack_size < frames_stack_mem_limit) and data_queue.full():
                            reason = f'Reached the data queue limit.'
                        log_append_print(f'[ Frames Supplier ] {reason} Pausing while it is not processed.', log_file=frames_supplier_log_file, disable_logs=disable_logs, loud=loud)
                    while (not doStop) and ((stack_size >= frames_stack_mem_limit) or data_queue.full()):
                        stack_size = data_queue.qsize()
                        try:
                            threads_signal_msg = threads_comm.get_nowait()
                            if threads_signal_msg == 'stop':
                                doStop = True
                                threads_comm.put('stop', timeout=th_tout)
                            if threads_signal_msg == 'faint':
                                time.sleep(1)
                                threads_comm.put('faint', timeout=th_tout)
                                raise Exception("Received info about another process dying")
                            if threads_signal_msg == 'no_processing':
                                for i in range(len(processes)-1):
                                    threads_comm.put('no_processing', timeout=th_tout)
                        except queue.Empty:
                            pass
                        time.sleep(0.1)
                if success and (not doStop):
                    global_counter1 += 1
                    if nsfio_params['enabled']:
                        if nsfi_prepared:
                            nsfi = scores[global_counter1-1]
                            if loud or (not disable_logs):
                                log_append_print(f'[ Frames Supplier ] Frame {global_counter1}: Read score of {nsfi} from provided data.', log_file=frames_supplier_log_file, disable_logs=disable_logs, loud=loud)
                        elif firstRun:
                            src_image1  = src_image
                            src_image2  = src_image
                            firstRun    = False
                            ssim_score  = 1.0
                            nsfi = ssim_score
                            if loud or (not disable_logs):
                                log_append_print(f'[ Frames Supplier ] First frame: {global_counter1}: {ssim_score}', log_file=frames_supplier_log_file, disable_logs=disable_logs, loud=loud)
                            if nsfio_params['frames_comparison_technique'] == 'clip':
                                img1        = Image.fromarray(src_image1).convert('RGB')
                                img1        = preprocess(img1).unsqueeze(0).to(device)
                                img1        = model.encode_image(img1)
                                img2        = img1
                            elif nsfio_params['frames_comparison_technique'] == 'ssim':
                                img1        = Image.fromarray(src_image1).convert('RGB')
                                img2        = img1
                            elif nsfio_params['frames_comparison_technique'] == 'ssim_':
                                image1_gray = cv2.cvtColor(src_image1, cv2.COLOR_BGR2GRAY)
                                image2_gray = image1_gray
                        else:
                            # Not first run - perform check
                            src_image1  = src_image2
                            src_image2  = src_image
                            if nsfio_params['frames_comparison_technique'] == 'ssim':
                                img1        = img2
                                img2        = Image.fromarray(src_image2).convert('RGB')
                                ssim_score  = compare_ssim(img1, img2, GPU=use_gpu and frames_supply_force_use_gpu)
                                nsfi = ssim_score
                                if loud or (not disable_logs):
                                    log_append_print(f'[ Frames Supplier ] SSIM: {global_counter1}: {ssim_score}', log_file=frames_supplier_log_file, disable_logs=disable_logs, loud=loud)
                            elif nsfio_params['frames_comparison_technique'] == 'ssim_':
                                image1_gray = image2_gray
                                image2_gray = cv2.cvtColor(src_image2, cv2.COLOR_BGR2GRAY)
                                ssim_score  = metrics.structural_similarity(image1_gray, image2_gray, full=True)[0]
                                nsfi = ssim_score
                                if loud or (not disable_logs):
                                    log_append_print(f'[ Frames Supplier ] SSIM_: {global_counter1}: {ssim_score}', log_file=frames_supplier_log_file, disable_logs=disable_logs, loud=loud)
                            elif nsfio_params['frames_comparison_technique'] == 'histogram':
                                hist_img1 = cv2.calcHist([src_image1], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
                                hist_img1[255, 255, 255] = 0  #ignore all white pixels
                                cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                                hist_img2 = cv2.calcHist([src_image2], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
                                hist_img2[255, 255, 255] = 0  #ignore all white pixels
                                cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                                metric_val = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_CORREL)
                                nsfi = metric_val
                                if loud or (not disable_logs):
                                    log_append_print(f'[ Frames Supplier ] Histogram: {global_counter1}: {metric_val}', log_file=frames_supplier_log_file, disable_logs=disable_logs, loud=loud)
                            elif nsfio_params['frames_comparison_technique'] == 'clip':
                                img1            = img2
                                img2            = Image.fromarray(src_image2).convert('RGB')
                                img2            = preprocess(img2).unsqueeze(0).to(device)
                                img2            = model.encode_image(img2)
                                cos_scores      = util.pytorch_cos_sim(img1, img2)
                                clip_comp_val   = float(cos_scores.detach()[0][0])
                                nsfi = clip_comp_val
                                if loud or (not disable_logs):
                                    log_append_print(f'[ Frames Supplier ] Clip: {global_counter1}: {clip_comp_val}', log_file=frames_supplier_log_file, disable_logs=disable_logs, loud=loud)
                        nsfi_c = nsfi >= nsfio_params['threshold']
                    if unloadPermuteToCPU:
                        img_batch_1, crop_region_1 = load_image2(src_image)
                        if not (frames_supply_force_use_gpu and use_gpu):
                            img_batch_1 = torch.from_numpy(img_batch_1).cpu().permute(0, 3, 1, 2)
                        else:
                            img_batch_1 = torch.from_numpy(img_batch_1).permute(0, 3, 1, 2)
                        data_queue.put(((img_batch_1, crop_region_1),nsfi_c), timeout=th_tout)
                    else:
                        data_queue.put((load_image2(src_image),nsfi_c), timeout=th_tout)
                    if doTqbar:
                        tqbar.update(1)
                    stack_size = data_queue.qsize()
                    if loud or (not disable_logs):
                        log_append_print(f'[ Frames Supplier ] Supplied frame number {global_counter1}. Stack approx. size: {stack_size}.', log_file=frames_supplier_log_file, disable_logs=disable_logs, loud=loud)
        if doTqbar:
            tqbar.n = global_counter1
            tqbar.last_print_n = global_counter1
            tqbar.refresh()
            if tqbar.total != tqbar.n:
                raise Exception(f'[ Frames Supplier ] Not all frames were supplied: {tqbar.n} out of {tqbar.total}!')
        videoCap.release()
        if loud or (not disable_logs):
            log_append_print(f'[ Frames Supplier ] Finished supplying frames. Waiting for the queue to be read.', log_file=frames_supplier_log_file, disable_logs=disable_logs, loud=loud)
        while not data_queue_is_empty:
            data_queue_is_empty = data_queue.empty()
            try:
                threads_signal_msg = threads_comm.get_nowait()
                if threads_signal_msg == 'stop':
                    doStop = True
                    threads_comm.put('stop', timeout=th_tout)
                if threads_signal_msg == 'faint':
                    time.sleep(1)
                    threads_comm.put('faint', timeout=th_tout)
                    raise Exception("Received info about another process dying")
                if threads_signal_msg == 'no_frames':
                    threads_comm.put('no_frames', timeout=th_tout)
            except queue.Empty:
                pass
            time.sleep(0.1)
        if loud or (not disable_logs):
            log_append_print(f'[ Frames Supplier ] Finished waiting. Exiting.', log_file=frames_supplier_log_file, disable_logs=disable_logs, loud=loud)
        return True
    except Exception as e:
        s = str(e)
        if doTqbar:
            tqbar.clear()
        if loud or (not disable_logs):
            log_append_print(f'[ Frames Supplier ] Exception happened. Cleaning queue and exiting gracefully. Details: {s}. Traceback:', log_file=frames_supplier_log_file, disable_logs=disable_logs, loud=loud)
            log_append_print(traceback.format_exc(), log_file=frames_supplier_log_file, disable_logs=disable_logs, loud=loud)
        else:
            print("Error during supplying frames.")
            print(f'Technical details: {s}.')
        doStop = False
        while not doStop:
            try:
                msg = threads_comm.get(timeout=th_tout)
            except queue.Empty:
                doStop = True
        need_send_kill_signal = 0
        for c_proces in processes:
            try:
                if c_proces.is_alive():
                    need_send_kill_signal += need_send_kill_signal
            except AssertionError:
                need_send_kill_signal += 1
        if need_send_kill_signal > 1:
            if loud or (not disable_logs):
                log_append_print(f'[ Frames Supplier ] Have to send an exit signal to another threads.', log_file=frames_supplier_log_file, disable_logs=disable_logs, loud=loud)
            try:
                threads_comm.put('faint', timeout=th_tout)
            except:
                pass
        doStop = False
        while not doStop:
            try:
                msg = data_queue.get(timeout=th_tout)
            except queue.Empty:
                doStop = True
        if loud or (not disable_logs):
            log_append_print(f'[ Frames Supplier ] Finished exception handling operations.', log_file=frames_supplier_log_file, disable_logs=disable_logs, loud=loud)
            log_append_print(f'Data queue is empty: {data_queue.empty()}', log_file=frames_supplier_log_file, disable_logs=disable_logs, loud=loud)
            log_append_print(f'Threads communication queue is empty: {threads_comm.empty()}', log_file=frames_supplier_log_file, disable_logs=disable_logs, loud=loud)
        return False


def workerFramesToVideoWriter(
        threads_comm:queue.Queue,
        data_queue:queue.Queue, processes:list,
        frames_to_write_queue:queue.Queue,
        dest_video_file: str,
        dest_video_temp: str,
        source_fps: cython.float,
        expected_fps: cython.float,
        dest_video_width:cython.int = 0,
        dest_video_height:cython.int = 0,
        dest_video_pix_fmt: str = "",
        dest_video_vid_options: dict = {},
        frames_in_source_video: cython.int=0,
        tqbar=None,
        frames_stack_flush_limit: cython.int=0,
        keep_files: cython.bint=False,
        closest: cython.bint=False,
        start_from: cython.int=0,
        max_frames: cython.int=0,
        save_all_frames: cython.bint=False,
        disable_logs: cython.bint=False,
        quiet: cython.bint=False,
        loud: cython.bint=False
        ):
    import io, av, av.datasets, av.error
    global_counter3:        cython.int  = 0
    global_counter3a:       cython.int  = 0
    i:                      cython.int  = 0
    j:                      cython.int  = 0
    k:                      cython.int  = 0
    glob_k:                 cython.int  = 0
    glob_k2:                cython.int  = 0
    processing_stopped:     cython.bint = False
    first_file_saving:      cython.bint = True
    check_bad_process:      cython.bint = False
    th_tout:                cython.int  = 3
    chunk:                  cython.int  = 0
    doStop:                 cython.bint = False
    doTqbar:                cython.bint = (tqbar is not None)
    frames_to_write:        list        = []
    frames:                 list        = []
    glob_frames:            list        = []
    files:                  list        = []
    video_format:           str         = 'mp4'
    frame_format:           str         = 'rgb24'
    codec:                  str         = 'h264'
    video_folder:           str         = ''
    video_folder_temp:      str         = ''
    video_filename:         str         = ''
    video_filename_temp:    str         = ''
    frame_filename_temp:    str         = ''
    video_filepath_temp:    str         = ''
    frame_filepath_temp:    str         = ''
    frames_writer_log_file: str         = ''
    vid_options_str:        str         = json.dumps(dest_video_vid_options)
    videos_list_filepath_temp: str      = ''
    frames_multiplier:     cython.int   = 0
    frames_to_generate:    cython.int   = 0
    extra_frames:          cython.int   = 0
    frames_in_dest_video:  cython.int   = 0
    gen_additional_frames: cython.bint  = False
    extra_frames_ins_pos:  cython.int   = 0
    last_amount_of_frames_written:  cython.int   = 0
    extra_frames_ins_frames: cython.int = 1
    res_rate                            = 0
    chunks:                 cython.int  = math.ceil(frames_in_source_video/frames_stack_flush_limit) if frames_stack_flush_limit > 0 else 1
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning
            )
        if not disable_logs:
            frames_writer_log_file = init_logs(os.path.join(dest_video_temp, 'logs'), 'frames_writer')
        if loud or (not disable_logs):
            log_append_print('Worker Frames to Video Writer', log_file=frames_writer_log_file, disable_logs=disable_logs, loud=loud)
            log_append_print(f'module name: {__name__}', log_file=frames_writer_log_file, disable_logs=disable_logs, loud=loud)
            log_append_print(f'parent process: {os.getppid()}', log_file=frames_writer_log_file, disable_logs=disable_logs, loud=loud)
            log_append_print(f'process id: {os.getpid()}', log_file=frames_writer_log_file, disable_logs=disable_logs, loud=loud)
            log_append_print(f"Frames writer will be:\n"
                    f"* writing to this file: {dest_video_file}\n"
                    f"* using this temporary directory: {dest_video_temp}\n"
                    f"* using the closest-only FPS mode: {closest}\n"
                    f"* keeping the files in the temporary directory if they already were there: {keep_files}\n"
                    f"* knowing that source FPS is: {source_fps}\n"
                    f"* knowing that excepted FPS is: {expected_fps}\n"
                    f"* knowing that source video has this amount of frames: {frames_in_source_video}\n"
                    f"* knowing that destination video width is: {dest_video_width}\n"
                    f"* knowing that destination video height is: {dest_video_height}\n"
                    f"* knowing that destination video pixel format is: {dest_video_pix_fmt}\n"
                    f"* knowing that destination video options are: {vid_options_str}\n"
                    f"* flushing frames to a video file when the stack will have this amount of items: {frames_stack_flush_limit}\n", log_file=frames_writer_log_file, disable_logs=disable_logs, loud=loud)
        
        source_fps_rounded, dest_fps_rounded, frames_to_generate, real_expected_fps, frames_to_remove = performFpsMath(source_fps, expected_fps)
        frames_multiplier = frames_to_generate + 1
        frames_in_dest_video = frames_in_source_video * frames_multiplier
        extra_frames = performFPSFloatingToIntegerMath(expected_fps, frames_in_dest_video)
        if extra_frames > 0:
            extra_frames_ins_pos = math.floor(frames_in_dest_video / (extra_frames + 1))
            if extra_frames_ins_pos < 1:
                # frames_in_dest_video-1 = places - "holes" - to insert frames
                extra_frames_ins_pos = 1
                extra_frames_ins_frames = math.floor(extra_frames / (frames_in_dest_video-1))
            gen_additional_frames = True
        video_folder                = os.path.split(dest_video_file)[0]
        video_filename              = os.path.split(dest_video_file)[1]
        video_folder_temp           = dest_video_temp
        video_folder_temp_frames    = os.path.join(video_folder_temp, 'frames')
        videos_list_filepath_temp   = os.path.join(video_folder_temp, 'list.txt')
        if closest:
            res_rate                = Fraction(real_expected_fps).limit_denominator(65535)
        else:
            res_rate                = int(expected_fps)
        
        os.makedirs(video_folder, exist_ok=True)
        os.makedirs(video_folder_temp, exist_ok=True)
        if save_all_frames:
            os.makedirs(video_folder_temp_frames, exist_ok=True)
        
        if start_from != 0:
            chunk = math.ceil(start_from / frames_stack_flush_limit)
            global_counter3 = start_from
            global_counter3a = chunk * frames_stack_flush_limit
            glob_k2 = start_from
            glob_k = math.floor(global_counter3 * expected_fps / source_fps_rounded) + 1
            if gen_additional_frames:
                glob_k += (math.floor(global_counter3 / extra_frames_ins_pos) * extra_frames_ins_frames)
            #raise NotImplementedError("[ Frames Writer ] Global Counter change not implemented (yet) when starting not from zero frame.")
            if loud or (not disable_logs):
                log_append_print(f'[ Frames Writer ] Starting from chunk nr. {chunk} (counted from zero) - it is set as a current one. Started from frame: {start_from}.', log_file=frames_writer_log_file, disable_logs=disable_logs, loud=loud)
                log_append_print(f'[ Frames Writer ] Counters data: Frames already written counter: {glob_k}; Frames processed according to source video counter: {glob_k2}.', log_file=frames_writer_log_file, disable_logs=disable_logs, loud=loud)
            if doTqbar:
                tqbar.n = global_counter3a
                tqbar.last_print_n = global_counter3a
                tqbar.refresh()
            
        
        try:
            assert os.path.isfile(videos_list_filepath_temp) and os.access(videos_list_filepath_temp, os.R_OK), \
                    f"File {videos_list_filepath_temp} doesn't exist or isn't readable"
        except AssertionError as e:
            with open(videos_list_filepath_temp, "w") as f:
                f.write('')
                f.close()
        
        if not keep_files:
            with open(videos_list_filepath_temp, "w") as f:
                f.write('')
                f.close()
            files = sorted(glob.glob(f'{video_folder_temp}{os.path.sep}*'), reverse=False)
            for f in files:
                silentremove(f)
        
        while not doStop:
            try:
                if not threads_comm.empty():
                    threads_signal_msg = threads_comm.get_nowait()
                    if threads_signal_msg == 'stop':
                        doStop = True
                        threads_comm.put('stop', timeout=th_tout)
                    if threads_signal_msg == 'faint':
                        time.sleep(1)
                        threads_comm.put('faint', timeout=th_tout)
                        raise Exception("[ Frames Writer ] Received info about another process dying")
                    if threads_signal_msg == 'no_processing':
                        processing_stopped = True
                    if threads_signal_msg == 'no_frames':
                        for i in range(len(processes)-1):
                            threads_comm.put('no_frames', timeout=th_tout)
            except queue.Empty:
                pass
            if not doStop:
                last_amount_of_frames_written = 0
                try:
                    try:
                        nextFrame = frames_to_write_queue.get(timeout=th_tout)
                        global_counter3 += 1
                        if loud or (not disable_logs):
                            log_append_print(f'[ Frames Writer ] Received frame number {global_counter3}', log_file=frames_writer_log_file, disable_logs=disable_logs, loud=loud)
                        data_queue_is_empty = False
                    except queue.Empty:
                        data_queue_is_empty = True
                except FileNotFoundError:
                    data_queue_is_empty = True
                    check_bad_process = True
                # In case we've received a last frame and missed the signal from another thread
                if global_counter3 >= frames_in_source_video:
                    processing_stopped = True
                if data_queue_is_empty and (not processing_stopped):
                    while data_queue_is_empty and (not processing_stopped):
                        data_queue_is_empty = frames_to_write_queue.empty()
                        try:
                            threads_signal_msg = threads_comm.get_nowait()
                            if threads_signal_msg == 'stop':
                                doStop = True
                                threads_comm.put('stop', timeout=th_tout)
                            if threads_signal_msg == 'faint':
                                time.sleep(1)
                                threads_comm.put('faint', timeout=th_tout)
                                raise Exception("Received info about another process dying")
                            if threads_signal_msg == 'no_frames':
                                for i in range(len(processes)-1):
                                    threads_comm.put('no_frames', timeout=th_tout)
                            if threads_signal_msg == 'no_processing':
                                processing_stopped = True
                        except queue.Empty:
                            pass
                        if check_bad_process:
                            if not processing_stopped:
                                raise Exception('Another process already finished work but the signal was not received about that.')
                        time.sleep(0.1)
                elif (not data_queue_is_empty) or processing_stopped:
                    if not data_queue_is_empty:
                        if nextFrame is not None:
                            frames_to_write.append(nextFrame)
                    if (len(frames_to_write) >= frames_stack_flush_limit) or processing_stopped:
                        lftw = len(frames_to_write)
                        if (lftw > 0):
                            if loud or (not disable_logs):
                                log_append_print(f'[ Frames Writer ] Flushing {lftw} frame(s) (Chunk {chunk}).', log_file=frames_writer_log_file, disable_logs=disable_logs, loud=loud)
                            # We flush frames at this point to the file
                            output_memory_file  = io.BytesIO()
                            output              = av.open(output_memory_file, 'w', format=video_format)
                            stream              = output.add_stream(codec, rate=res_rate)
                            stream.width        = dest_video_width
                            stream.height       = dest_video_height
                            stream.pix_fmt      = dest_video_pix_fmt
                            stream.options      = dest_video_vid_options
                            j                   = 0
                            if save_all_frames:
                                byte_io             = io.BytesIO()
                            for fframe_raw in frames_to_write:
                                (fframes, y1, x1, y2, x2) = fframe_raw
                                frames = [(tensor[0] * 255).byte().flip(0).permute(1, 2, 0).numpy()[y1:y2, x1:x2].copy() for tensor in fframes]
                                # in "frames" we have a list of frames - including 1st and last one, which are the source video ones, used for generation
                                i = 0
                                if loud or (not disable_logs):
                                    log_append_print(f'[ Frames Writer ] Frame nr. {glob_k}. Frames to be processed: {len(frames)-1}.', log_file=frames_writer_log_file, disable_logs=disable_logs, loud=loud)
                                for fframe in frames:
                                    if (i < len(frames)-1) or (processing_stopped and (j == len(frames_to_write)-1) and (i == len(frames)-1)):
                                        frame   = av.VideoFrame.from_ndarray(fframe, format=frame_format)
                                        packet  = stream.encode(frame)
                                        output.mux(packet)
                                        last_amount_of_frames_written += 1
                                        if save_all_frames:
                                            file_num = str(glob_k)
                                            file_num = file_num.zfill(len(str(frames_in_source_video*(frames_to_generate+1))))
                                            file_num2 = str(glob_k2)
                                            file_num2 = file_num2.zfill(len(str(frames_in_source_video)))
                                            frame_filename_temp = f'frame{file_num}_src_frame_{file_num2}.jpg'
                                            frame_filepath_temp = os.path.join(video_folder_temp_frames, frame_filename_temp)
                                            frame.save(frame_filepath_temp)
                                        glob_k += 1
                                    i += 1
                                j += 1
                                glob_k2 += 1
                            packet = stream.encode(None)
                            output.mux(packet)
                            output.close()
                            
                            file_num = str(chunk)
                            file_num = file_num.zfill(len(str(chunks)))
                            
                            video_filename_temp = f'res{file_num}.mp4'
                            video_filepath_temp = os.path.join(video_folder_temp, video_filename_temp)
                            
                            if last_amount_of_frames_written == 0:
                                # This is NOT supposed to happen. Something is wrong with a frames processing logic.
                                raise Exception("Amount of frames to be written is ZERO. This should not happen - something went wrong."
                                    f"Counters data: Frames intended to be written amount: {lftw}; Frames written counter: {glob_k}; Frames written according to source video counter: {glob_k2}")
                            
                            if loud or (not disable_logs):
                                log_append_print(f'[ Frames Writer ] Last amount of frames to be written: {last_amount_of_frames_written}.', log_file=frames_writer_log_file, disable_logs=disable_logs, loud=loud)
                                log_append_print(f'[ Frames Writer ] Counters data: Frames intended to be written amount: {lftw}; Frames written counter: {glob_k}; Frames written according to source video counter: {glob_k2}', log_file=frames_writer_log_file, disable_logs=disable_logs, loud=loud)
                            
                            with open(video_filepath_temp, "wb") as f:
                                f.write(output_memory_file.getbuffer())
                                f.close()
                            
                            with open(videos_list_filepath_temp, "a") as f:
                                f.writelines(['file \'%s\'\n' % video_filename_temp, ])
                                f.close()
                            chunk += 1
                            global_counter3a += lftw
                        else:
                            if loud or (not disable_logs):
                                log_append_print(f'[ Frames Writer ] Amount of frames to be written is zero, thus not creating empty files.', log_file=frames_writer_log_file, disable_logs=disable_logs, loud=loud)
                        # End of flushing
                        frames_to_write = []
                        if doTqbar:
                            if lftw > 0:
                                tqbar.update(lftw)
                            if processing_stopped:
                                tqbar.refresh()
                else:
                    if loud or (not disable_logs):
                        log_append_print(f'[ Frames Writer ] How did you even get here?', log_file=frames_writer_log_file, disable_logs=disable_logs, loud=loud)
                    raise Exception(f'[ Frames Writer ] Unknown exception which is not supposed to happen.')
                if processing_stopped and (len(frames_to_write) == 0):
                    if loud or (not disable_logs):
                        log_append_print(f'[ Frames Writer ] Received signal about processing finished. Stopping writing.', log_file=frames_writer_log_file, disable_logs=disable_logs, loud=loud)
                    doStop = True
                    if doTqbar:
                        tqbar.refresh()
        if doTqbar:
            tqbar.n = global_counter3a
            tqbar.last_print_n = global_counter3a
            tqbar.refresh()
            if tqbar.n < tqbar.total - 1:
                if loud or (not disable_logs):
                    log_append_print(f'[ Frames Writer ] Warning - not all frames were written: {tqbar.n} out of {tqbar.total}!', log_file=frames_writer_log_file, disable_logs=disable_logs, loud=loud)
        if loud or (not disable_logs):
            log_append_print(f'[ Frames Writer ] Summary: Total frames written: {global_counter3a}. Total chunks: {chunk}.', log_file=frames_writer_log_file, disable_logs=disable_logs, loud=loud)
        return True
    except Exception as e:
        s = str(e)
        if doTqbar:
            tqbar.clear()
        if loud or (not disable_logs):
            log_append_print(f'[ Frames Writer ] Exception happened. Cleaning queue and exiting gracefully. Details: {s}. Traceback:', log_file=frames_writer_log_file, disable_logs=disable_logs, loud=loud)
            log_append_print(traceback.format_exc(), log_file=frames_writer_log_file, disable_logs=disable_logs, loud=loud)
        else:
            print("Error during writing frames.")
            print(f'Technical details: {s}.')
        doStop = False
        while not doStop:
            try:
                msg = threads_comm.get(timeout=th_tout)
            except queue.Empty:
                doStop = True
        need_send_kill_signal = 0
        for c_proces in processes:
            try:
                if c_proces.is_alive():
                    need_send_kill_signal += need_send_kill_signal
            except AssertionError:
                need_send_kill_signal += 1
        if need_send_kill_signal > 1:
            if loud or (not disable_logs):
                log_append_print(f'[ Frames Writer ] Have to send an exit signal to another threads.', log_file=frames_writer_log_file, disable_logs=disable_logs, loud=loud)
            try:
                threads_comm.put('faint', timeout=th_tout)
            except:
                pass
        doStop = False
        while not doStop:
            try:
                msg = data_queue.get(timeout=th_tout)
            except queue.Empty:
                doStop = True
        if loud or (not disable_logs):
            log_append_print(f'[ Frames Writer ] Finished exception handling operations.', log_file=frames_writer_log_file, disable_logs=disable_logs, loud=loud)
            log_append_print(f'Data queue is empty: {data_queue.empty()}', log_file=frames_writer_log_file, disable_logs=disable_logs, loud=loud)
            log_append_print(f'Threads communication queue is empty: {threads_comm.empty()}', log_file=frames_writer_log_file, disable_logs=disable_logs, loud=loud)
        return False


def dup_fr(
        permute_already_done: cython.bint = False,
        img_prev = None,
        img_curr = None,
        source_video_width: cython.int = 0,
        source_video_height: cython.int = 0,
        frames_to_generate_real: cython.int   = 0,
        quiet: cython.bint=False,
        loud: cython.bint=False
        ):
    img_batch_1:                np.ndarray
    crop_region_1:              list         = []
    img_prev_p                               = None
    img_curr_p                               = None
    generatedFramesList                      = None
    if loud:
        print(f"[ Frames Processor ] (In function) Found the frames which similarity is below threshold. "
                f"Instead of performing inference, duplicating previous frame.")
    # Performing permute for these frames.
    if not permute_already_done:
        img_batch_1, crop_region_1 = img_prev
        img_prev_p = torch.from_numpy(img_batch_1).cpu().permute(0, 3, 1, 2)
        img_batch_1, crop_region_1 = img_curr
        img_curr_p = torch.from_numpy(img_batch_1).cpu().permute(0, 3, 1, 2)
    else:
        img_prev_p = img_prev[0]
        img_curr_p = img_curr[0]
    generatedFramesList = [[img_prev_p],0,0,source_video_width,source_video_height]
    for i in range(frames_to_generate_real):
        generatedFramesList[0].append(img_prev_p)
    generatedFramesList[0].append(img_curr_p)
    return generatedFramesList

def do_fx_rem_frame(
        generatedFramesListDelayed            = None,
        generatedFramesList                   = None,
        frames_to_generate_real: cython.int   = 0,
        frames_to_generate:      cython.int   = 0,
        frames_to_remove:        cython.int   = 0,
        generated_frames_count:  cython.int   = 0,
        generated_frames_length: cython.int   = 0,
        global_counter2:         cython.int   = 0,
        frames_sent_to_writer:              cython.int   = 0,
        disable_logs:            cython.bint  = False,
        frames_processor_log_file:  str       = '',
        quiet:                   cython.bint  = False,
        loud:                    cython.bint  = False
        ):
    i:                      cython.int  = 0
    generatedFramesListPre: list        = generatedFramesList[0]
    generatedFramesListDelayed[0]       = generatedFramesListPre[0:(frames_to_generate_real+2)]
    generatedFramesListPre              = []
    if loud or (not disable_logs):
        log_append_print(f'[ Frames Processor ][DELAYED] Length of generated frames list: {len(generatedFramesListDelayed[0])-1}; gl.indx.: {generated_frames_count}; gen.len.:{generated_frames_length}.', log_file=frames_processor_log_file, disable_logs=disable_logs, loud=loud)
    for i in range(max(len(generatedFramesListDelayed[0])-1,0)): #-1 as last frame will be removed anyways
        if frames_to_remove > 0:
            if ((generated_frames_count + i) % (frames_to_remove+1)) != 1:
                generatedFramesListPre.append(generatedFramesListDelayed[0][i])
        else:
            generatedFramesListPre.append(generatedFramesListDelayed[0][i])
    # That's correct, 2 times. One replaced and one appended
    generatedFramesListPre[len(generatedFramesListPre)-1] = generatedFramesListDelayed[0][len(generatedFramesListDelayed[0])-1]
    generatedFramesListPre.append(generatedFramesListDelayed[0][len(generatedFramesListDelayed[0])-1])
    
    if loud or (not disable_logs):
        log_append_print(f'[ Frames Processor ][DELAYED] Leaving {len(generatedFramesListPre)} frames.', log_file=frames_processor_log_file, disable_logs=disable_logs, loud=loud)
    generatedFramesListDelayed[0]   = generatedFramesListPre
    if loud or (not disable_logs):
        log_append_print(f'[ Frames Processor ][DELAYED] Frame {global_counter2}: Resulting length of generated frames list: {len(generatedFramesListDelayed[0])-1}. Total expected sent frames length: {frames_sent_to_writer}.', log_file=frames_processor_log_file, disable_logs=disable_logs, loud=loud)
    generatedFramesListPre      = generatedFramesList[0]
    generatedFramesList[0]      = generatedFramesListPre[(frames_to_generate_real+2):]
    return (generatedFramesListDelayed, generatedFramesList)

def workerFramesProcessor(
        threads_comm:queue.Queue,
        data_queue:queue.Queue,
        processes:list,
        frames_to_write_queue:queue.Queue,
        dest_video_temp: str,
        source_fps: cython.float,
        expected_fps: cython.float,
        source_video_width:cython.int = 0,
        source_video_height:cython.int = 0,
        interpolation_model: str='',
        use_gpu: cython.bint=False,
        use_half_precision: cython.bint=False,
        frames_in_source_video: cython.int=0,
        nsfio_params: dict={'enabled':False},
        tqbar=None,
        closest: cython.bint=False,
        permute_already_done: cython.bint = False,
        start_from: cython.int=0,
        max_frames: cython.int=0,
        disable_logs: cython.bint=False,
        quiet: cython.bint=False,
        loud: cython.bint=False
        ):
    doStop:                     cython.bint  = False
    data_queue_is_empty:        cython.bint  = False
    out_of_frames:              cython.bint  = False
    first_frame:                cython.bint  = True
    gen_additional_frames:      cython.bint  = False
    first_attempt_dq_empty:     cython.bint  = True
    nsfi_c:                     cython.bint  = True
    fx_rem_frame:               cython.bint  = False
    i:                          cython.int   = 0
    j:                          cython.int   = 0
    generated_frames_count:     cython.int   = 0
    generated_frames_count_delayed:  cython.int   = 0
    generated_frames_length:         cython.int   = 0
    generated_frames_length_delayed: cython.int   = 0
    frames_sent_to_writer:           cython.int   = 0
    frames_sent_to_writer_delayed:   cython.int   = 0
    global_counter2:            cython.int   = 0
    th_tout:                    cython.int   = 3
    doTqbar:                    cython.bint  = (tqbar is not None)
    source_fps_rounded:         cython.int   = 0
    dest_fps_rounded:           cython.int   = 0
    frames_to_generate:         cython.int   = 0
    frames_to_generate_real:    cython.int   = 0
    real_expected_fps:          cython.float = 0
    frames_to_remove:           cython.int   = 0
    extra_frames:               cython.int   = 0
    extra_frames_ins_pos:       cython.int   = 0
    extra_frames_ins_pos_src:   cython.int   = 0
    extra_frames_ins_frames:    cython.int   = 1
    frames_multiplier:          cython.int   = 0
    frames_in_dest_video:       cython.int   = 0
    additional_frames_c:        cython.int   = 0
    temp_str1:                  str          = ''
    temp_str2:                  str          = ''
    frames_processor_log_file:  str          = ''
    stack_size:                 cython.int   = 0
    img_batch_1:                np.ndarray
    crop_region_1:              list         = []
    frames_to_remove_list:      list         = []
    frames_to_leave_list:       list         = []
    generatedFramesList                      = None
    generatedFramesListDelayed               = None
    generatedFramesListPre                   = None
    nextFramePre                             = None
    nextFrame                                = None
    
    from justai_video_processing_lib import filmIntLib
    
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning
            )
        if not disable_logs:
            frames_processor_log_file = init_logs(os.path.join(dest_video_temp, 'logs'), 'frames_processor')
        
        if loud or (not disable_logs):
            log_append_print('Worker Frames Processor', log_file=frames_processor_log_file, disable_logs=disable_logs, loud=loud)
            log_append_print(f'module name: {__name__}', log_file=frames_processor_log_file, disable_logs=disable_logs, loud=loud)
            log_append_print(f'parent process: {os.getppid()}', log_file=frames_processor_log_file, disable_logs=disable_logs, loud=loud)
            log_append_print(f'process id: {os.getpid()}', log_file=frames_processor_log_file, disable_logs=disable_logs, loud=loud)
            log_append_print(f"Frames processor will be:\n"
                    f"* using GPU: {use_gpu}\n"
                    f"* using Half Precision: {use_half_precision}\n"
                    f"* using this inpterpolation model: {interpolation_model}\n"
                    f"* using the closest-only FPS mode: {closest}\n"
                    f"* knowing that source FPS is: {source_fps}\n"
                    f"* knowing that excepted FPS is: {expected_fps}\n"
                    f"* knowing that source video has this amount of frames: {frames_in_source_video}\n", log_file=frames_processor_log_file, disable_logs=disable_logs, loud=loud)
        
        source_fps_rounded, dest_fps_rounded, frames_to_generate, real_expected_fps, frames_to_remove = performFpsMath(source_fps, expected_fps)
        frames_multiplier = frames_to_generate + 1
        if loud or (not disable_logs):
            log_append_print(f'[ Frames Processor ] Frames multiplier: {frames_multiplier}.', log_file=frames_processor_log_file, disable_logs=disable_logs, loud=loud)
            log_append_print(f'[ Frames Processor ] Frames to generate: {frames_to_generate}.', log_file=frames_processor_log_file, disable_logs=disable_logs, loud=loud)
            log_append_print(f'[ Frames Processor ] Frames to remove: {frames_to_remove}.', log_file=frames_processor_log_file, disable_logs=disable_logs, loud=loud)
            log_append_print(f'[ Frames Processor ] Source FPS Rounded: {source_fps_rounded}.', log_file=frames_processor_log_file, disable_logs=disable_logs, loud=loud)
            log_append_print(f'[ Frames Processor ] Dest FPS Rounded: {dest_fps_rounded}.', log_file=frames_processor_log_file, disable_logs=disable_logs, loud=loud)
            log_append_print(f'[ Frames Processor ] Real expected FPS: {real_expected_fps}.', log_file=frames_processor_log_file, disable_logs=disable_logs, loud=loud)
        
        if not closest:
            frames_in_dest_video = frames_in_source_video * frames_multiplier
            extra_frames = performFPSFloatingToIntegerMath(expected_fps, frames_in_dest_video)
            if loud or (not disable_logs):
                log_append_print(f'[ Frames Processor ] The strict FPS mode: calculated extra frames to be generated amount is {extra_frames}.', log_file=frames_processor_log_file, disable_logs=disable_logs, loud=loud)
            if extra_frames > 0:
                extra_frames_ins_pos = math.floor(frames_in_dest_video / (extra_frames + 1))
                if extra_frames_ins_pos < 1:
                    # frames_in_dest_video-1 = places - "holes" - to insert frames
                    extra_frames_ins_pos = 1
                    extra_frames_ins_frames = math.floor(extra_frames / (frames_in_dest_video-1))
                gen_additional_frames = True
                # We need now the index (each this-frame) at which in the source video more frames will be generated. How many - extra_frames_ins_frames.
                # this value will be stored in here: extra_frames_ins_pos_src
                extra_frames_ins_pos_src = math.floor(extra_frames / (frames_in_source_video-1))
                if loud or (not disable_logs):
                    temp_str1 = f'{extra_frames} frames'
                    if extra_frames == 1:
                        temp_str1 = f'{extra_frames} frame'
                    log_append_print(f'[ Frames Processor ] In order to round FPS to {dest_fps_rounded}, we need to generate extra {temp_str1} in the resulting video.', log_file=frames_processor_log_file, disable_logs=disable_logs, loud=loud)
                    temp_str1 = f'{extra_frames_ins_pos} frames'
                    if extra_frames_ins_pos == 1:
                        temp_str1 = f'{extra_frames_ins_pos} frame'
                    temp_str2 = f'{extra_frames_ins_frames} frames'
                    if extra_frames_ins_frames == 1:
                        temp_str2 = f'{extra_frames_ins_frames} frame'
                    log_append_print(f'[ Frames Processor ] This means that every {temp_str1} there will be additional {temp_str2}.', log_file=frames_processor_log_file, disable_logs=disable_logs, loud=loud)
            else:
                if loud or (not disable_logs):
                    log_append_print(f'[ Frames Processor ] No extra frames need to be generated for rounding FPS.', log_file=frames_processor_log_file, disable_logs=disable_logs, loud=loud)
        
        img_prev            = None
        img_prev2           = None
        img_prev_p          = None
        img_curr            = None
        img_curr_p          = None
        keep_model_loaded   = True
        if loud or (not disable_logs):
            log_append_print(f'[ Frames Processor ] Initializing FILM library.', log_file=frames_processor_log_file, disable_logs=disable_logs, loud=loud)
        videoProcessor      = filmIntLib(interpolation_model, use_gpu, use_half_precision)
        if loud or (not disable_logs):
            log_append_print(f'[ Frames Processor ] FILM library initialized.', log_file=frames_processor_log_file, disable_logs=disable_logs, loud=loud)
        
        if start_from != 0:
            if loud or (not disable_logs):
                log_append_print(f'[ Frames Processor ] Start from frame is non-zero: {start_from}.', log_file=frames_processor_log_file, disable_logs=disable_logs, loud=loud)
            global_counter2                 = start_from - 1
            generated_frames_count          = math.floor( (frames_to_generate + 1 ) * (start_from - 1) ) # without removal
            generated_frames_count_delayed  = generated_frames_count - (frames_to_generate + 1 ) # without removal
            
            frames_sent_to_writer           = math.floor( max(start_from - 1, 0) * expected_fps/source_fps_rounded) + 1  # after removal
            frames_sent_to_writer_delayed   = math.floor( max(start_from - 2, 0) * expected_fps/source_fps_rounded) + 1  # after removal
            
            if frames_to_remove > 0:
                frames_sent_to_writer = 1
                frames_sent_to_writer_delayed = 1
                
                generated_frames_length = 2 + frames_to_generate
                
                for i in range(global_counter2 * frames_multiplier):
                    if (i % (frames_to_remove + 1)) != 1:
                        frames_sent_to_writer_delayed += 1
                frames_sent_to_writer = frames_sent_to_writer_delayed
            
            
            if gen_additional_frames:
                additional_frames_c = (math.floor((start_from-1) / extra_frames_ins_pos) * extra_frames_ins_frames)
                if loud or (not disable_logs):
                    log_append_print(f'[ Frames Processor ] Additional frames has to be calculated as well. Calculated amount: {additional_frames_c}.', log_file=frames_processor_log_file, disable_logs=disable_logs, loud=loud)
                frames_sent_to_writer += additional_frames_c
                generated_frames_count += additional_frames_c
            if loud or (not disable_logs):
                log_append_print(f'[ Frames Processor ] Starting from a frame nr. {global_counter2} - it is set as a latest processed one (counted from zero). Already provided frames counter: {frames_sent_to_writer}. gl.indx.: {generated_frames_count};', log_file=frames_processor_log_file, disable_logs=disable_logs, loud=loud)
            if doTqbar:
                tqbar.n = global_counter2
                tqbar.last_print_n = global_counter2
                tqbar.refresh()
        if loud or (not disable_logs):
            log_append_print(f'[ Frames Processor ] Starting main cycle.', log_file=frames_processor_log_file, disable_logs=disable_logs, loud=loud)
        while not doStop:
            try:
                if not threads_comm.empty():
                    threads_signal_msg = threads_comm.get_nowait()
                    if threads_signal_msg == 'stop':
                        doStop = True
                        threads_comm.put('stop', timeout=th_tout)
                    if threads_signal_msg == 'faint':
                        time.sleep(1)
                        threads_comm.put('faint', timeout=th_tout)
                        raise Exception("[ Frames Processor ] Received info about another process dying")
                    if threads_signal_msg == 'no_frames':
                        out_of_frames = True
            except queue.Empty:
                pass
            if not doStop:
                try:
                    nextFrame, nsfi_c  = data_queue.get(timeout=th_tout)
                    if nextFrame is None:
                        raise RuntimeError('Frame with None value received.')
                    if loud or (not disable_logs):
                        log_append_print(f'[ Frames Processor ] Received frame number {global_counter2}, similarity is set to: {nsfi_c}.', log_file=frames_processor_log_file, disable_logs=disable_logs, loud=loud)
                    data_queue_is_empty = False
                except queue.Empty:
                    data_queue_is_empty = True
                if not data_queue_is_empty:
                    if first_frame:
                        img_prev2   = nextFrame
                        img_prev    = nextFrame
                        img_curr    = nextFrame
                    else:
                        img_prev2   = img_prev
                        img_prev    = img_curr
                        img_curr    = nextFrame
                        
                        # This block is related to generating additional frames for strictly keeping the FPS.
                        frames_to_generate_real = frames_to_generate
                        if gen_additional_frames and ((global_counter2 % extra_frames_ins_pos_src == 0) or (extra_frames_ins_pos_src  == 1)):
                            frames_to_generate_real = frames_to_generate + extra_frames_ins_frames
                            if loud or (not disable_logs):
                                temp_str1 = f'{frames_to_generate} frames'
                                if frames_to_generate == 1:
                                    temp_str1 = f'{frames_to_generate} frame'
                                temp_str2 = f'{extra_frames_ins_frames} frames'
                                if extra_frames_ins_frames == 1:
                                    temp_str2 = f'{extra_frames_ins_frames} frame'
                                log_append_print(f'[ Frames Processor ] Adding additional {temp_str2}. Now we will generate {frames_to_generate_real} instead of {temp_str1}.', log_file=frames_processor_log_file, disable_logs=disable_logs, loud=loud)
                        # End of the mentioned block of additional FPS.
                        
                        if (not nsfi_c) and nsfio_params['enabled'] and (global_counter2 > 1):
                            if nsfio_params['nsfio_method'] == 'duplicate':
                                if loud or (not disable_logs):
                                    log_append_print(f"[ Frames Processor ] Found the frames which similarity is below threshold. "
                                            f"Instead of performing inference, duplicating previous frame.", log_file=frames_processor_log_file, disable_logs=disable_logs, loud=loud)
                                generatedFramesList = dup_fr(permute_already_done, img_prev, img_curr, source_video_width, source_video_height, frames_to_generate_real, quiet, loud)
                            elif nsfio_params['nsfio_method'] == 'prolong':
                                if loud or (not disable_logs):
                                    log_append_print(f"[ Frames Processor ] Found the frames which similarity is below threshold. "
                                            f"Instead of performing inference, performing inference of the frame before that and the previous one. "
                                            f"Will generate {frames_to_generate*2 + 2 + (frames_to_generate_real-frames_to_generate)} frames.", log_file=frames_processor_log_file, disable_logs=disable_logs, loud=loud)
                                generatedFramesList = videoProcessor.inference2(
                                    interpolation_model,
                                    img_prev2,
                                    img_prev,
                                    use_gpu,
                                    frames_to_generate*2 + 2 + (frames_to_generate_real-frames_to_generate),
                                    use_half_precision,
                                    keep_model_loaded,
                                    False,
                                    permute_already_done
                                    )
                                fx_rem_frame = True
                        else:
                            if loud or (not disable_logs):
                                log_append_print(f"[ Frames Processor ] Performing regular inference.", log_file=frames_processor_log_file, disable_logs=disable_logs, loud=loud)
                            try:
                                generatedFramesList = videoProcessor.inference2(
                                    interpolation_model,
                                    img_prev,
                                    img_curr,
                                    use_gpu,
                                    frames_to_generate_real,
                                    use_half_precision,
                                    keep_model_loaded,
                                    False,
                                    permute_already_done
                                    )
                            except Exception as e:
                                s = str(e)
                                raise Exception("Inference unsuccessful! Details: {s}.")
                            if loud or (not disable_logs):
                                log_append_print(f"[ Frames Processor ] Regular inference done.", log_file=frames_processor_log_file, disable_logs=disable_logs, loud=loud)
                        generated_frames_length_delayed = generated_frames_length
                        generated_frames_length = len(generatedFramesList[0])-1 if generatedFramesList is not None else 0
                        # in generatedFramesList[0] we have frames
                        # in frames_sent_to_writer we have current counter of frames which were actually sent to writing. Sent, already in the queue.
                        if fx_rem_frame and (global_counter2 > 1):
                            if loud or (not disable_logs):
                                log_append_print(f'[ Frames Processor ] Performing change of both previous frames list and current one.', log_file=frames_processor_log_file, disable_logs=disable_logs, loud=loud)
                            generatedFramesListDelayed, generatedFramesList = do_fx_rem_frame(
                                generatedFramesListDelayed,
                                generatedFramesList,
                                frames_to_generate_real,
                                frames_to_generate,
                                frames_to_remove,
                                generated_frames_count_delayed,
                                generated_frames_length_delayed,
                                global_counter2,
                                frames_sent_to_writer_delayed,
                                disable_logs,
                                frames_processor_log_file,
                                quiet,
                                loud
                                )
                            fx_rem_frame = False
                            if loud or (not disable_logs):
                                log_append_print(f'[ Frames Processor ] After change - size (minus last frame) of Delayed frames list is: {len(generatedFramesListDelayed[0])-1} and current one is: {len(generatedFramesList[0])-1}.', log_file=frames_processor_log_file, disable_logs=disable_logs, loud=loud)
                        if loud or (not disable_logs):
                            log_append_print(f'[ Frames Processor ] Length of generated frames list: {len(generatedFramesList[0])-1}; gl.indx.: {generated_frames_count}; gen.len.:{generated_frames_length}.', log_file=frames_processor_log_file, disable_logs=disable_logs, loud=loud)
                        if frames_to_remove > 0:
                            generatedFramesListPre = []
                            for i in range(len(generatedFramesList[0])-1): #-1 as last frame will be removed anyways
                                if ((generated_frames_count + i) % (frames_to_remove+1)) != 1:
                                    generatedFramesListPre.append(generatedFramesList[0][i])
                            generatedFramesListPre.append(generatedFramesList[0][len(generatedFramesList[0])-1])
                            if loud or (not disable_logs):
                                log_append_print(f'[ Frames Processor ] Leaving {len(generatedFramesListPre)} frames.', log_file=frames_processor_log_file, disable_logs=disable_logs, loud=loud)
                            generatedFramesList[0]  = generatedFramesListPre
                            if loud or (not disable_logs):
                                log_append_print(f'[ Frames Processor ] Length of generated frames list (after frames removal): {len(generatedFramesList[0])-1}; gl.indx.: {generated_frames_count}; gen.len.:{generated_frames_length}.', log_file=frames_processor_log_file, disable_logs=disable_logs, loud=loud)
                        if loud or (not disable_logs):
                            log_append_print(f'[ Frames Processor ] Frame {global_counter2}: Resulting length of generated frames list: {len(generatedFramesList[0])-1}. Total expected already sent frames length: {frames_sent_to_writer}.', log_file=frames_processor_log_file, disable_logs=disable_logs, loud=loud)
                    global_counter2 += 1
                    if doTqbar:
                        tqbar.update(1)
                    if not first_frame:
                        try:
                            if generatedFramesListDelayed is not None:
                                frames_to_write_queue.put(generatedFramesListDelayed, timeout=th_tout)
                                frames_sent_to_writer_delayed    = frames_sent_to_writer
                                frames_sent_to_writer           += (len(generatedFramesListDelayed[0])-1) # after removal
                                generated_frames_count_delayed   = generated_frames_count   # without removal
                                generated_frames_count          += generated_frames_length
                            generatedFramesListDelayed = generatedFramesList
                        except queue.Full:
                            if loud or (not disable_logs):
                                log_append_print(f'[ Frames Processor ] This situation should not happen. The Frames to Write stack is full.', log_file=frames_processor_log_file, disable_logs=disable_logs, loud=loud)
                            raise Exception("The Frames to Write stack is full - this is not supposed to happen.")
                    else:
                        first_frame = False
                else:
                    if loud or (not disable_logs):
                        log_append_print(f'[ Frames Processor ] Pausing frames processor while no frames are in stack.', log_file=frames_processor_log_file, disable_logs=disable_logs, loud=loud)
                    while data_queue_is_empty and (not out_of_frames):
                        data_queue_is_empty = data_queue.empty()
                        stack_size          = data_queue.qsize()
                        try:
                            threads_signal_msg = threads_comm.get_nowait()
                            if threads_signal_msg == 'stop':
                                doStop = True
                                threads_comm.put('stop', timeout=th_tout)
                            if threads_signal_msg == 'faint':
                                time.sleep(1)
                                threads_comm.put('faint', timeout=th_tout)
                                raise Exception("Received info about another process dying")
                            if threads_signal_msg == 'no_frames':
                                out_of_frames = True
                        except queue.Empty:
                            pass
                        time.sleep(0.1)
                    if out_of_frames and data_queue_is_empty:
                        data_queue_is_empty = data_queue.empty()
                        stack_size          = data_queue.qsize()
                        if (stack_size > 0):
                            data_queue_is_empty = False
                        if data_queue_is_empty:
                            if loud or (not disable_logs):
                                log_append_print(f'[ Frames Processor ] Received the out of frames signal so stopping. Stack fize: {stack_size}', log_file=frames_processor_log_file, disable_logs=disable_logs, loud=loud)
                            # flushing delayed - generatedFramesListDelayed
                            if generatedFramesListDelayed is not None:
                                frames_to_write_queue.put(generatedFramesListDelayed, timeout=th_tout)
                                frames_sent_to_writer_delayed    = frames_sent_to_writer
                                frames_sent_to_writer           += (len(generatedFramesListDelayed[0])-1)  # after removal
                                generated_frames_count_delayed   = generated_frames_count
                                generated_frames_count          += generated_frames_length  # without removal
                            generatedFramesListDelayed = generatedFramesList
                            if generatedFramesListDelayed is not None:
                                frames_to_write_queue.put(generatedFramesListDelayed, timeout=th_tout)
                                frames_sent_to_writer_delayed    = frames_sent_to_writer
                                frames_sent_to_writer           += (len(generatedFramesListDelayed[0])-1)  # after removal
                                generated_frames_count_delayed   = generated_frames_count
                                generated_frames_count          += generated_frames_length   # without removal
                                generatedFramesListDelayed       = None
                                generatedFramesList              = None
                            doStop = True
                            if doTqbar:
                                tqbar.refresh()
        if doTqbar:
            tqbar.n = global_counter2
            tqbar.last_print_n = global_counter2
            tqbar.refresh()
            if tqbar.total != tqbar.n:
                raise Exception(f'[ Frames Processor ] Not all frames were processed: {tqbar.n} out of {tqbar.total}!')
        try:
            threads_comm.put('no_processing', timeout=th_tout)
        except:
            pass
        if loud or (not disable_logs):
            log_append_print(f'[ Frames Processor ] Finished processing frames. Waiting for the queue to be read', log_file=frames_processor_log_file, disable_logs=disable_logs, loud=loud)
        data_queue_is_empty = False
        while not data_queue_is_empty:
            data_queue_is_empty = data_queue.empty()
            try:
                threads_signal_msg = threads_comm.get_nowait()
                if threads_signal_msg == 'stop':
                    doStop = True
                    threads_comm.put('stop', timeout=th_tout)
                if threads_signal_msg == 'faint':
                    time.sleep(1)
                    threads_comm.put('faint', timeout=th_tout)
                    raise Exception("Received info about another process dying")
            except queue.Empty:
                pass
            time.sleep(0.1)
        if loud or (not disable_logs):
            log_append_print(f'[ Frames Processor ] Finished waiting. Exiting.', log_file=frames_processor_log_file, disable_logs=disable_logs, loud=loud)
    except Exception as e:
        s = str(e)
        if doTqbar:
            tqbar.clear()
        if loud or (not disable_logs):
            log_append_print(f'[ Frames Processor ] Exception happened. Cleaning queue and exiting gracefully. Details: {s}. Traceback:', log_file=frames_processor_log_file, disable_logs=disable_logs, loud=loud)
            log_append_print(traceback.format_exc(), log_file=frames_processor_log_file, disable_logs=disable_logs, loud=loud)
        else:
            print("Error during processing frames.")
            print(f'Technical details: {s}.')
        doStop = False
        while not doStop:
            try:
                msg = threads_comm.get(timeout=th_tout)
            except queue.Empty:
                doStop = True
        need_send_kill_signal = 0
        for c_proces in processes:
            try:
                if c_proces.is_alive():
                    need_send_kill_signal += need_send_kill_signal
            except AssertionError:
                need_send_kill_signal += 1
        if need_send_kill_signal > 1:
            if loud or (not disable_logs):
                log_append_print(f'[ Frames Processor ] Have to send an exit signal to another threads.', log_file=frames_processor_log_file, disable_logs=disable_logs, loud=loud)
            try:
                threads_comm.put('faint', timeout=th_tout)
            except:
                pass
        doStop = False
        while not doStop:
            try:
                msg = data_queue.get(timeout=th_tout)
            except queue.Empty:
                doStop = True
        if loud or (not disable_logs):
            log_append_print(f'[ Frames Processor ] Finished exception handling operations.', log_file=frames_processor_log_file, disable_logs=disable_logs, loud=loud)
            log_append_print(f'Data queue is empty: {data_queue.empty()}', log_file=frames_processor_log_file, disable_logs=disable_logs, loud=loud)
            log_append_print(f'Threads communication queue is empty: {threads_comm.empty()}', log_file=frames_processor_log_file, disable_logs=disable_logs, loud=loud)
        return False

def combineMultipleVideosToOne(
        dest_video_file: str,
        dest_video_temp: str,
        temp_dir_filenames_list_file: str='',
        quiet: cython.bint=False,
        loud: cython.bint=False
        ):
    s:                      str = ''
    ffmpeg_output:          str = ''
    video_filename_temp:    str = ''
    files:                  list = []
    if temp_dir_filenames_list_file == '':
        temp_dir_filenames_list_file   = os.path.join(dest_video_temp, 'list.txt')
        with open(temp_dir_filenames_list_file, "w") as f:
            f.write('')
            f.close()
        f = open(temp_dir_filenames_list_file, "a")
        files = sorted(glob.glob(f'{dest_video_temp}{os.path.sep}*'), reverse=False)
        for video_filename_temp in files:
            if not video_filename_temp.endswith('.txt'):
                f.writelines(['file \'%s\'\n' % video_filename_temp, ])
        f.close()
    try:
        video_filename_temp = "res.mp4"
        ffmpeg_params = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", os.path.split(temp_dir_filenames_list_file)[1],
            "-c", "copy",
            video_filename_temp,
        ]
        if loud:
            print(f'[ Videos Combining ] Calling ffmpeg in a directory "{dest_video_temp}", files from there will be concatenated without re-encoding to the file: "{dest_video_file}".')
        ffmpeg_output = subprocess.check_output(ffmpeg_params, cwd=dest_video_temp, text=True) # , stderr=subprocess.STDOUT, shell=True
        shutil.copyfile(os.path.join(dest_video_temp, video_filename_temp), dest_video_file)
        silentremove(os.path.join(dest_video_temp, video_filename_temp))
        if loud:
            print(f'[ Videos Combining ] Videos concatenation executed (hopefully) successfully.')
        return True
    except (OSError, ValueError, subprocess.CalledProcessError) as e:
        if loud:
            s = str(e)
            print(f'[ Videos Combining ] An error occured during ffmpeg video combining. Details: {s}.')
            print(f"[ Videos Combining ] Captured output: {ffmpeg_output}")
            print(f'[ Videos Combining ] Traceback:')
            print(traceback.format_exc())
        return False

def copyAudioFromVideo(
        source_video_with_audio: str,
        destination_video_without_audio: str,
        dest_video_temp:str,
        quiet: cython.bint=False,
        loud: cython.bint=False
        ):
    s: str = ''
    ffmpeg_output: str = ''
    destination_video_temp_path = os.path.join(dest_video_temp, 'audio_'+os.path.split(destination_video_without_audio)[1])
    try:
        ffmpeg_params = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-i", destination_video_without_audio,
            "-i", source_video_with_audio,
            "-c", "copy",
            "-map", "0:0",
            "-map", "1:1",
            "-c", "copy",
            "-shortest",
            destination_video_temp_path,
        ]
        if loud:
            print(f'[ Videos Combining ] Replacing audio in "{destination_video_without_audio}" from "{source_video_with_audio}" - using temporary file "{destination_video_temp_path}".')
        ffmpeg_output = subprocess.check_output(ffmpeg_params, cwd=dest_video_temp, text=True) # , stderr=subprocess.STDOUT, shell=True
        shutil.copyfile(destination_video_temp_path, destination_video_without_audio)
        silentremove(destination_video_temp_path)
        if loud:
            print(f'[ Videos Combining ] Audio replacement in video executed (hopefully) successfully.')
        return True
    except (OSError, ValueError, subprocess.CalledProcessError) as e:
        if loud:
            s = str(e)
            print(f'[ Videos Combining ] An error occured during ffmpeg video combining. Details: {s}.')
            print(f"[ Videos Combining ] Captured output: {ffmpeg_output}")
            print(f'[ Videos Combining ] Traceback:')
            print(traceback.format_exc())
        return False

def workerNsfioGenerator(
        #threads_comm:       queue.Queue,
        #data_queue:         queue.Queue,
        #processes:          list            = [],
        source_video_file:  str             = '',
        dest_video_temp:    str             = '',
        use_gpu:            cython.bint     = False,
        nsfio_params:       dict            = {'enabled':False},
        tqbar                               = None,
        doTqbar:            cython.bint     = False,
        disable_logs:       cython.bint     = False,
        quiet:              cython.bint     = False,
        loud:               cython.bint     = False
        ):
    frames_map:         list            = []
    from justai_video_processing_lib_frames_map import build_frames_map
    frames_map = build_frames_map(
        source_video_file,
        nsfio_params['frames_comparison_technique'],
        use_gpu,
        dest_video_temp,
        'file' if nsfio_params['frames_map_file_json'] != '' else 'no',
        '',
        False,
        nsfio_params['frames_map_file_json'],
        nsfio_params['clip_model_name'],
        nsfio_params['clip_pretrained'],
        doTqbar or loud,
        tqbar,
        disable_logs,
        quiet,
        loud
    )


def interpolate_film(
        source_video_file:                  str             = '',
        dest_video_file:                    str             = '',
        dest_video_temp:                    str             = '',
        dest_fps:                           cython.float    = 0,
        interpolation_model:                str             = '',
        use_gpu:                            cython.bint     = True,
        use_half_precision:                 cython.bint     = False,
        frames_stack_mem_limit:             cython.int      = 0,
        frames_stack_flush_limit:           cython.int      = 0,
        do_exit:                            cython.bint     = True,
        closest:                            cython.bint     = False,
        allowed_frames_float_threshold:     cython.float    = 0.05,
        unloadPermuteToCPU:                 cython.bint     = False,
        no_audio_copy:                      cython.bint     = False,
        no_files_concatenation:             cython.bint     = False,
        resume:                             cython.bint     = False,
        start_from_frame:                   cython.int      = 0,
        max_frames:                         cython.int      = 0,
        frames_supply_force_use_gpu:        cython.bint     = False,
        nsfio_params:                       dict            = {'enabled':False},
        save_all_frames:                    cython.bint     = False,
        disable_logs:                       cython.bint     = False,
        quiet:                              cython.bint     = False,
        loud:                               cython.bint     = False
        ):
    if quiet:
        loud = False
    frames_in_source_video:         cython.int      = 0
    source_video_width:             cython.int      = 0
    source_video_height:            cython.int      = 0
    source_video_fps:               cython.float    = 0
    doTqbar:                        cython.bint     = False
    destFPSisFloat:                 cython.bint     = False
    videos_combination_result:      cython.bint     = False
    audio_replacement_result:       cython.bint     = False
    keep_files:                     cython.bint     = False
    within_frames_float_threshold:  cython.bint     = False
    files_found_in_temp_dir:        cython.bint     = False
    thread_need_to_be_executed:     cython.bint     = True
    processes:                      list            = []
    frames_map:                     list            = []
    dest_video_pix_fmt:             str             = 'yuv444p'
    dest_video_options:             dict            = {'crf': '15'}
    source_fps_rounded:             cython.int      = 0
    dest_fps_rounded:               cython.int      = 0
    frames_to_generate:             cython.int      = 0
    frames_multiplier:              cython.int      = 0
    real_expected_fps:              cython.float    = 0
    frames_to_remove:               cython.int      = 0
    videos_list_filepath_temp:      str             = ''
    last_line:                      str             = ''
    chunks:                         cython.int      = 0
    last_chunk:                     cython.int      = 0
    start_from:                     cython.int      = 0
    tqbar_pos:                      cython.int      = 0
    
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=UserWarning
        )
    
    if math.ceil(dest_fps) != math.floor(dest_fps):
        destFPSisFloat = True
        # TODO: maybe to implement destination floating FPS later on. Currently this is the best answer.
        raise NotImplementedError('Floating FPS option is not supported.')
    if not disable_logs:
        init_logs()
    (frames_in_source_video, source_video_width, source_video_height, source_video_fps) = video_info(source_video_file, False, False)
    source_fps_rounded, dest_fps_rounded, frames_to_generate, real_expected_fps, frames_to_remove = performFpsMath(source_video_fps, dest_fps)    
    frames_multiplier = frames_to_generate + 1
    if not closest:
        # We actually care about resulting floating fps. Not source one.
        within_frames_float_threshold = checkReasonableFPS(real_expected_fps, frames_in_source_video * frames_multiplier, allowed_frames_float_threshold)
        if not within_frames_float_threshold:
            if not quiet:
                print(f'[ Interpolation ][ ERROR ] The strict interpolation cannot be done in such a way that the threshold of frames floating will be kept lower than {allowed_frames_float_threshold} seconds. '
                        f'In order to fix that You can: use non-strict option - so called "closest" interpolation, this will just multiply frames {frames_multiplier} times and not try to round them to the '
                        f'requested FPS ({dest_fps_rounded} will become {real_expected_fps}), or adjust threshold, or use different - higher - FPS option.')
            return False
        else:
            if loud:
                print(f'[ Interpolation ] Using strict requested FPS mode. Output FPS will be: {dest_fps}.')
    else:
        if loud:
            print(f'[ Interpolation ] Using closest to requested FPS mode. Output FPS will be: {real_expected_fps}.')
    if resume:
        if loud:
            print(f'Attempt to resume the failed interpolation.')
        videos_list_filepath_temp   = os.path.join(dest_video_temp, 'list.txt')
        try:
            assert os.path.isfile(videos_list_filepath_temp) and os.access(videos_list_filepath_temp, os.R_OK), \
                    f"File {videos_list_filepath_temp} doesn't exist or isn't readable"
            if loud:
                print(f'Found the list file "list.txt" in temp directory {dest_video_temp}.')
            with open(videos_list_filepath_temp, 'r') as f:
                for line in f:
                    if len(line) > 4:
                        last_line = line
        except AssertionError as e:
            if loud:
                print(f'Did not find the list file "list.txt" in temp directory {dest_video_temp}. Will try to rebuild it from files if there are any in that directory.')
            # resume apparently isn't possible. the easy way. Let's try to create a list file.
            f = open(videos_list_filepath_temp, "w")
            files = sorted(glob.glob(f'{dest_video_temp}{os.path.sep}*'), reverse=False)
            for video_filename_temp in files:
                if (not video_filename_temp.endswith('.txt')) and (not os.path.isdir(video_filename_temp)):
                    #files_found_in_temp_dir = True
                    f.writelines(['file \'%s\'\n' % os.path.split(video_filename_temp)[1], ])
                    last_line = 'file \'%s\'\n' % os.path.split(video_filename_temp)[1]
            f.close()
        if len(last_line) > 8:
            if loud:
                print(f'Getting the last filename from temporary files directory.')
            last_line = last_line[6:]
            last_line = last_line[:len(last_line)-2]
        else:
            if not quiet:
                print(f'[ Interpolation ] Could not resume. Proceeding as usual.')
            resume = False
        #TODO: it would be a good idea to check if all other files exist.
        if resume:
            # So, we CAN resume. We have a last filename which was generated stored in a variable "last_line"
            chunks = math.ceil(frames_in_source_video/frames_stack_flush_limit) if frames_stack_flush_limit > 0 else 1
            last_line = last_line.rsplit( ".", 1 )[ 0 ]
            last_chunk = int(last_line[3:])
            # the last processed frame can be calculated as last_chunk+1 multiplied by the batch size which is frames_stack_flush_limit
            # we have to check if we even have to run the workers - maybe we don't and it is a last stage.
            if last_chunk + 1 >= chunks:
                thread_need_to_be_executed = False
            else:
                start_from = (last_chunk + 1) * frames_stack_flush_limit
                keep_files = True
            if not quiet:
                print(f'[ Interpolation ] Resume attempt successful. Continuing from frame {start_from}.')
    
    start_from = max(start_from_frame, start_from)
    
    if (not quiet) and (not loud):
        if nsfio_params['enabled'] and nsfio_params['nsfio_before_interpolation'] and (not nsfio_params['frames_map_file_json_supplied']):
            tqbar0 = tqdm(desc='Prepared',      unit='frame(s)', total=frames_in_source_video, position=tqbar_pos,   leave=True)
            tqbar_pos = 1
        tqbar1 = tqdm(desc='Supplied',      unit='frame(s)', total=frames_in_source_video,     position=tqbar_pos,   leave=True)
        tqbar2 = tqdm(desc='Processed',     unit='frame(s)', total=frames_in_source_video,     position=tqbar_pos+1, leave=True)
        tqbar3 = tqdm(desc='Written',       unit='frame(s)', total=frames_in_source_video + 1, position=tqbar_pos+2, leave=True)
        tqbar4 = tqdm(desc='Finalization',  unit='step(s)',  total=2,                          position=tqbar_pos+3, leave=True)
        doTqbar = True
    else:
        tqbar0 = None
        tqbar1 = None
        tqbar2 = None
        tqbar3 = None
        tqbar4 = None
        doTqbar = False
    
    if thread_need_to_be_executed:
        mp.set_start_method('fork', force=True)
        ctx                     = mp.get_context()
        threads_comm            = ctx.Queue()
        data_queue              = ctx.Queue()
        frames_to_write_queue   = ctx.Queue()
        
        
        
        if nsfio_params['enabled'] and nsfio_params['nsfio_before_interpolation']:
            if (nsfio_params['frames_map_file_json'] is None) or (nsfio_params['frames_map_file_json'] == ''):
                nsfio_params['frames_map_file_json'] = TemporaryFile(prefix=this_app_codename).name
            if not nsfio_params['frames_map_file_json_supplied']:
                nsfioGenerator  = ctx.Process(target=workerNsfioGenerator,      args=(
                                                                                        #threads_comm,
                                                                                        #data_queue,
                                                                                        #processes,
                                                                                        source_video_file,
                                                                                        dest_video_temp,
                                                                                        use_gpu,
                                                                                        nsfio_params,
                                                                                        tqbar0,
                                                                                        doTqbar,
                                                                                        disable_logs,
                                                                                        quiet,
                                                                                        loud
                                                                                        )
                                            )
                processes.append(nsfioGenerator)
                nsfioGenerator.start()
                nsfioGenerator.join()
                if doTqbar:
                    tqbar1.reset(total=frames_in_source_video)
                    tqbar2.reset(total=frames_in_source_video)
                    tqbar3.reset(total=frames_in_source_video+1)
                    tqbar4.reset(total=2)
            with open(nsfio_params['frames_map_file_json']) as f:
                frames_map = json.load(f)
                f.close()
            if frames_map is None:
                raise RuntimeError('Building of frames map failed.')
            else:
                nsfio_params['frames_map_data'] = frames_map
        
        
        framesSupplier          = ctx.Process(target=workerFramesSupplier,      args=(
                                                                                        threads_comm,
                                                                                        data_queue,
                                                                                        processes,
                                                                                        source_video_file,
                                                                                        dest_video_temp,
                                                                                        use_gpu,
                                                                                        frames_supply_force_use_gpu,
                                                                                        nsfio_params,
                                                                                        tqbar1,
                                                                                        frames_stack_mem_limit,
                                                                                        unloadPermuteToCPU,
                                                                                        start_from,
                                                                                        max_frames,
                                                                                        disable_logs,
                                                                                        quiet,
                                                                                        loud
                                                                                        )
                                            )
        framesProcessor         = ctx.Process(target=workerFramesProcessor,     args=(
                                                                                        threads_comm,
                                                                                        data_queue,
                                                                                        processes,
                                                                                        frames_to_write_queue,
                                                                                        dest_video_temp,
                                                                                        source_video_fps,
                                                                                        dest_fps,
                                                                                        source_video_width,
                                                                                        source_video_height,
                                                                                        interpolation_model,
                                                                                        use_gpu,
                                                                                        use_half_precision,
                                                                                        frames_in_source_video,
                                                                                        nsfio_params,
                                                                                        tqbar2,
                                                                                        closest,
                                                                                        unloadPermuteToCPU,
                                                                                        start_from,
                                                                                        max_frames,
                                                                                        disable_logs,
                                                                                        quiet,
                                                                                        loud
                                                                                        )
                                            )
        framesToVideoWriter     = ctx.Process(target=workerFramesToVideoWriter, args=(
                                                                                        threads_comm,
                                                                                        data_queue,
                                                                                        processes,
                                                                                        frames_to_write_queue,
                                                                                        dest_video_file,
                                                                                        dest_video_temp,
                                                                                        source_video_fps,
                                                                                        dest_fps,
                                                                                        source_video_width,
                                                                                        source_video_height,
                                                                                        dest_video_pix_fmt,
                                                                                        dest_video_options,
                                                                                        frames_in_source_video,
                                                                                        tqbar3,
                                                                                        frames_stack_flush_limit,
                                                                                        keep_files,
                                                                                        closest,
                                                                                        start_from,
                                                                                        max_frames,
                                                                                        save_all_frames,
                                                                                        disable_logs,
                                                                                        quiet,
                                                                                        loud
                                                                                        )
                                            )
        processes.append(framesSupplier)
        processes.append(framesProcessor)
        framesSupplier.start()
        framesToVideoWriter.start()
        framesProcessor.start()
    try:
        if thread_need_to_be_executed:
            framesToVideoWriter.join()
            framesProcessor.join()
            framesSupplier.join()
        if doTqbar:
            tqbar4.reset(total=2)
        if not no_files_concatenation:
            videos_combination_result = combineMultipleVideosToOne(dest_video_file, dest_video_temp, os.path.join(dest_video_temp, 'list.txt'), quiet, loud)
            if not videos_combination_result:
                if loud:
                    print(f'[ Main process ][ ERROR ] Videos concatenation executed unsuccessfully.')
            else:
                if doTqbar:
                    tqbar4.update(1)
                if not no_audio_copy:
                    audio_replacement_result = copyAudioFromVideo(source_video_file, dest_video_file, dest_video_temp, quiet, loud)
                    if not audio_replacement_result:
                        if loud:
                            print(f'[ Main process ][ ERROR ] Audio replacement executed unsuccessfully.')
                    else:
                        if doTqbar:
                            tqbar4.update(1)
                else:
                    if doTqbar:
                        tqbar4.update(1)
        else:
            if doTqbar:
                tqbar4.update(2)
    except KeyboardInterrupt:
        print('Received keyboard interrupt. Stopping.')
        if thread_need_to_be_executed:
            framesSupplier.terminate()
            framesToVideoWriter.terminate()
            framesProcessor.terminate()
            framesToVideoWriter.join()
            framesProcessor.join()
            framesSupplier.join()
    if doTqbar:
        tqbar1.close()
        tqbar2.close()
        tqbar3.close()
    if do_exit:
        try:
            cv2.destroyAllWindows()
        except:
            pass
    return True

def interpolate(
        source_video_file:              str             = '',
        dest_video_file:                str             = '',
        dest_video_temp:                str             = '',
        dest_fps:                       cython.float    = 0,
        interpolator:                   str             = '',
        interpolation_model:            str             = '',
        use_gpu:                        cython.bint     = True,
        use_half_precision:             cython.bint     = False,
        frames_stack_mem_limit:         cython.int      = 0,
        frames_stack_flush_limit:       cython.int      = 0,
        do_exit:                        cython.bint     = True,
        closest:                        cython.bint     = False,
        allowed_frames_float_threshold: cython.float    = 0.05,
        unloadPermuteToCPU:             cython.bint     = False,
        no_audio_copy:                  cython.bint     = False,
        no_files_concatenation:         cython.bint     = False,
        resume:                         cython.bint     = False,
        start_from_frame:               cython.int      = 0,
        max_frames:                     cython.int      = 0,
        frames_supply_force_use_gpu:    cython.bint     = False,
        nsfio_params:                   dict            = {'enabled':False},
        save_all_frames:                cython.bint     = False,
        disable_logs:                   cython.bint     = False,
        quiet:                          cython.bint     = False,
        loud:                           cython.bint     = False
        ):
    if quiet:
        loud = False
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=UserWarning
        )
    if frames_stack_mem_limit == 0:
        frames_stack_mem_limit = default_frames_stack_mem_limit
    if interpolator == 'film':
        return interpolate_film(source_video_file,
            dest_video_file,
            dest_video_temp,
            dest_fps,
            interpolation_model,
            use_gpu,
            use_half_precision,
            frames_stack_mem_limit,
            frames_stack_flush_limit,
            do_exit,
            closest,
            allowed_frames_float_threshold,
            unloadPermuteToCPU,
            no_audio_copy,
            no_files_concatenation,
            resume,
            start_from_frame,
            max_frames,
            frames_supply_force_use_gpu,
            nsfio_params,
            save_all_frames,
            disable_logs,
            quiet,
            loud)
    else:
        return False
