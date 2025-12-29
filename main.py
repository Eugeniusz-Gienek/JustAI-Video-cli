from justai_video_processor import video_info, interpolate, available_interpolators, available_tasks, available_nsfio_methods, available_frame_comparison_techniques
import sys, math, os, time
from os import access, R_OK, W_OK
from os.path import isfile, isdir
from pathlib import Path

default_interpolation_model = 'film_models/film_net_fp32.pt'

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Video optimizations')
    
    parser.add_argument('--source_video', type=str, default='', help='Path to the source video')
    parser.add_argument('--dest_video', type=str, default='', help='Path to the destination video')
    parser.add_argument('--task', type=str, default='', help='What to execute - interpolate, upscale, anything else')
    parser.add_argument('--tasks_available', action='store_true', help='List available tasks')
    parser.add_argument('--interpolators_available', action='store_true', help='List available interpolators')
    parser.add_argument('--frame_comparison_methods_available', action='store_true', help='List available frame comparison techniques')
    parser.add_argument('--nsfio_methods_available', action='store_true', help='List available NSFIO (Non-Similar Frames Interpolation Optimization) methods.')
    parser.add_argument('--video_info', action='store_true', help='Provide some information about source video.')
    
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    parser.add_argument('--fp16', action='store_true', help='Use FP16 instead of FP32 (Uses twice as much RAM)')
    
    parser.add_argument('--interpolator', type=str, default='film', help='(Frames interpolation) Interpolator which will be used to produce additional frames - e.g. film, rife,...')
    parser.add_argument('--interpolation_model', type=str, default='', help='(Frames interpolation) Path to the TorchScript interpolation model')
    parser.add_argument('--interpolation_fps', type=int, default=0, help='Destination video should have this FPS. It is a whole number. For fractions, please check --interpolation_closest option.')
    parser.add_argument('--interpolation_closest', action='store_true', help='If the source fps is not a whol number, do not produce the output with a rounded FPS but find the closest multiplicator. e.g. 23.9 FPS will become 47.8 FPS if asked for 48 FPS instead of 48 FPS.')
    
    parser.add_argument('--disable_unload_permute_to_cpu', action='store_true', help='Do not use CPU for a tensor permute, if the CPU is not powerful enough for this optimization while preparing frames.')
    
    parser.add_argument('--allowed_frames_float_threshold', type=float, default=0.05, help='Allowed frames floating threshold (in seconds)')
    parser.add_argument('--frames_stack_mem_limit', type=int, default=100, help='Keep maximum this amount of frames in memory in advance.')
    parser.add_argument('--frames_stack_flush_limit', type=int, default=100, help='Write frames to a file when the amount of frames in a stack reaches this amount. Putting 1 is not recommended - this operation is a bit resource-hungry.')
    
    parser.add_argument('--no_copy_audio', action='store_true', help='Do not copy audio from source file')
    parser.add_argument('--no_files_concatenation', action='store_true', help='Do not concatenate the generated files. This also disables audio copying.')
    
    parser.add_argument('--start_from_frame', type=int, default=0, help='Start operations from this frame number.')
    parser.add_argument('--max_frames', type=int, default=0, help='Stop after this amount of frames')
    
    parser.add_argument('--clip_model_name', type=str, default='ViT-B-16-plus-240', help='CLIP model to use.')
    parser.add_argument('--clip_pretrained', type=str, default='laion400m_e32', help='CLIP pretrained to use.')
    
    parser.add_argument('--frames_map_file', type=str, default='', help='Path where to store frames map in human-readable text.')
    parser.add_argument('--frames_map_dont_show_progress', action='store_true', help='Do not show frames map building progress')
    parser.add_argument('--frames_map_file_no_frame_numbers', action='store_true', help='Do not store frame numbers in a human-readable text frames map file.')
    parser.add_argument('--frames_comparison_technique', type=str, default='ssim', help='Which approach to use for comparing frames.')
    parser.add_argument('--frames_map_file_json', type=str, default='', help='Path for/with frames map in JSON format.')
    parser.add_argument('--frames_map_file_json_supplied', action='store_true', help='Use the provided file and not re-generate it.')
    
    parser.add_argument('--disable_nsfio', action='store_true', help='Disable Non-Similar Frames Interpolation Optimization.')
    parser.add_argument('--nsfio_parralel_interpolation', action='store_true', help='Run Non-Similar Frames Interpolation Optimization in parralel to interpolation rather than prior to it.')
    parser.add_argument('--nsfio_threshold', type=float, default=0.90, help='Threshold for Non-Similar Frames Interpolation Optimization - float, from 0.00 to 1.00.')
    parser.add_argument('--nsfio_method', type=str, default='prolong', help='Method to use for Non-Similar Frames Interpolation Optimization.')
    
    parser.add_argument('--frames_supply_force_use_gpu', action='store_true', help='Force using GPU (if --gpu is provided) for frames supply as well when in threading mode. Warning - this might very likely slow down the overall process as the two processess will compete - supply and interpolation.')
    
    parser.add_argument('--save_all_frames', action='store_true', help='Save all frames and not remove them.')
    
    parser.add_argument('--loud', action='store_true', help='Basically enable a debug mode - report all the actions performed. Will not work if --quiet flag is set.')
    parser.add_argument('--quiet', action='store_true', help='Be completely silent. Overrides --loud flag.')
    
    parser.add_argument('--disable_logs', action='store_true', help='Disable logs generation.')
    
    
    parser.add_argument('--resume', action='store_true', help='Try to resume the failed process')
    
    args = parser.parse_args()
    
    if args.tasks_available:
        print('Available tasks:')
        tasks = available_tasks()
        for task in tasks:
            print(f'* {task}')
        sys.exit(0)
    
    if args.interpolators_available:
        print('Available interpolators:')
        interpolators = available_interpolators()
        for interpolator in interpolators:
            print(f'* {interpolator}')
        sys.exit(0)
    
    if args.nsfio_methods_available:
        print('Available NSFIO methods:')
        nsfio_methods = available_nsfio_methods()
        for nsfio_method in nsfio_methods:
            print(f'* {nsfio_method}')
        sys.exit(0)
    
    if args.frame_comparison_methods_available:
        print('Available frame comparison techniques')
        fcts = available_frame_comparison_techniques()
        for fct in fcts:
            print(f'* {fct}')
        sys.exit(0)
    
    if args.video_info and (args.source_video != ''):
        video_info(args.source_video, True, True)
        sys.exit(0)
    
    if args.task == 'interpolate':
        try:
            start_time = time.time()
            interpolators = available_interpolators()
            if args.interpolator.lower() not in interpolators:
                print(f'[ ERROR ] Interpolator "{args.interpolator}" is not supported.')
                sys.exit(0)
            if args.interpolation_fps <=0:
                (frames, width, height, fps) = video_info(args.source_video, False, False)
                fps2 = math.ceil(fps)
                print(f'[ ERROR ] Interpolation fps (--interpolation_fps NUM) should be set and should be a positive integer number which is higher than the original FPS: {fps2}.')
                sys.exit(0)
            if args.interpolation_model == '':
                args.interpolation_model = default_interpolation_model
            assert isfile(args.interpolation_model) and access(args.interpolation_model, R_OK), f"Model {args.interpolation_model} doesn't exist or isn't readable"
            if args.dest_video == '':
                print(f'[ ERROR ] Destination file (--dest_video VIDEO_FILE) was not specified while it is supposed to be.')
                sys.exit(0)
            dest_folder = os.path.split(args.dest_video)[0]
            dest_filename = os.path.split(args.dest_video)[1]
            dest_video_temp = dest_folder + os.path.sep +  f'{dest_filename}.temp'
            os.makedirs(dest_folder, exist_ok=True)
            os.makedirs(dest_video_temp, exist_ok=True)
            assert isdir(dest_folder) and access(dest_folder, W_OK), f"Destination directory {dest_folder} doesn't exist or isn't writable."
            Path(args.dest_video).touch()
            #Path(dest_video_file_temp).touch()
            frames_stack_mem_limit = args.frames_stack_mem_limit
            frames_stack_flush_limit = args.frames_stack_flush_limit
            
            nsfio_params = {'enabled': False}
            if not args.disable_nsfio:
                nsfio_params = {
                    'enabled':                          True,
                    'threshold':                        args.nsfio_threshold,
                    'frames_map_file_json':             args.frames_map_file_json,
                    'frames_map_file_json_supplied':    args.frames_map_file_json_supplied,
                    'frames_comparison_technique':      args.frames_comparison_technique,
                    'clip_model_name':                  args.clip_model_name,
                    'clip_pretrained':                  args.clip_pretrained,
                    'nsfio_before_interpolation':       not args.nsfio_parralel_interpolation,
                    'nsfio_method':                     args.nsfio_method,
                    'frames_map_data':                  None,
                }
            else:
                nsfio_params = {
                    'enabled':                      False,
                }
            
            res = interpolate(args.source_video,
                args.dest_video,
                dest_video_temp,
                args.interpolation_fps,
                args.interpolator.lower(),
                args.interpolation_model,
                args.gpu,
                args.fp16,
                frames_stack_mem_limit,
                frames_stack_flush_limit,
                True,
                args.interpolation_closest,
                args.allowed_frames_float_threshold,
                not args.disable_unload_permute_to_cpu,
                args.no_copy_audio,
                args.no_files_concatenation,
                args.resume,
                args.start_from_frame,
                args.max_frames,
                args.frames_supply_force_use_gpu,
                nsfio_params,
                args.save_all_frames,
                args.disable_logs,
                args.quiet,
                args.loud)
            time_sec = time.time() - start_time
            if time_sec >= 3600:
                res_time = '%d hours %d minutes %d seconds' % ( math.floor(time_sec/3600), math.floor(math.ceil(time_sec) % 3600)/60,  math.ceil(time_sec) % 60)
            elif time_sec >= 60:
                appndx = '' if math.floor(time_sec/60) == 1 else 's'
                res_time = '%d minute%s %.2f seconds' % (math.floor(time_sec/60), appndx,  math.ceil(time_sec) % 60 + (round(time_sec,2) - math.floor(time_sec)))
            else:
                res_time = '%.2f seconds' % (round(time_sec,2))
            if (type(res) is not dict) and (res == False):
                if not args.quiet:
                    print('[ ERROR ] Interpolation did not succeed.')
                    print("--- Interpolation execution total time: %s ---" % res_time)
                    sys.exit(0)
                else:
                    sys.exit(1)
            elif not args.quiet:
                print("--- Interpolation execution total time: %s ---" % res_time)
        except (AssertionError, OSError, IOError) as e:
            s = str(e)
            print(f'[ ERROR ] {s}')
    elif args.task == 'build_frames_map':
        try:
            start_time = time.time()
            from justai_video_processing_lib_frames_map import build_frames_map
            if args.source_video == '':
                print(f'[ ERROR ] Source video file (--source_video VIDEO_FILE) was not specified while it is supposed to be.')
                sys.exit(0)
            output_to = 'screen'
            if args.frames_map_file_json != '':
                dest_map_folder = os.path.split(args.frames_map_file_json)[0]
                os.makedirs(dest_map_folder, exist_ok=True)
                Path(args.frames_map_file_json).touch()
                output_to = 'file'
            if args.frames_map_file != '':
                dest_map_folder = os.path.split(args.frames_map_file)[0]
                os.makedirs(dest_map_folder, exist_ok=True)
                Path(args.frames_map_file).touch()
                output_to = 'file'
            if (output_to == 'file') and (args.frames_map_file == '') and (args.frames_map_file_json == ''):
                print(f'[ ERROR ] Output is set to file but neither json nor text-only output file was specified (--frames_map_file FILE, --frames_map_file_json FILE) while it is supposed to be.')
                sys.exit(0)
            
            dest_video_temp = ''
            if args.dest_video != '':
                dest_folder = os.path.split(args.dest_video)[0]
                dest_filename = os.path.split(args.dest_video)[1]
                dest_video_temp = dest_folder + os.path.sep +  f'{dest_filename}.temp'
            
            res = build_frames_map(args.source_video,
                args.frames_comparison_technique,
                args.gpu,
                dest_video_temp,
                output_to,
                args.frames_map_file,
                args.frames_map_file_no_frame_numbers,
                args.frames_map_file_json,
                args.clip_model_name,
                args.clip_pretrained,
                not args.frames_map_dont_show_progress,
                None,
                args.disable_logs,
                args.quiet,
                args.loud)
            time_sec = time.time() - start_time
            if time_sec >= 3600:
                res_time = '%d hours %d minutes %d seconds' % ( math.floor(time_sec/3600), math.floor(math.ceil(time_sec) % 3600)/60,  math.ceil(time_sec) % 60)
            elif time_sec >= 60:
                appndx = '' if math.floor(time_sec/60) == 1 else 's'
                res_time = '%d minute%s %.2f seconds' % (math.floor(time_sec/60), appndx,  math.ceil(time_sec) % 60 + (round(time_sec,2) - math.floor(time_sec)))
            else:
                res_time = '%.2f seconds' % (round(time_sec,2))
            if not args.quiet:
                print("--- Building frames map execution total time: %s ---" % res_time)
        except (AssertionError, OSError, IOError) as e:
            s = str(e)
            print(f'[ ERROR ] {s}')
    else:
        sys.exit(0)
