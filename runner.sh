# Example of video processing
./setup.sh && python -W ignore::UserWarning  main.py --gpu --fp16 --interpolation_fps=48 --task interpolate --source_video your_fancy_source_video.mp4 --dest_video your_fancy_dest_video.mp4 --frames_stack_flush_limit=10 --resume --max_frames=20 --loud 
