import os, cython
from pathlib import Path
from tempfile import TemporaryDirectory

this_app_codename = 'justai_video'

def init_logs(
        logs_temp_path: str = '',
        logs_file_code: str = ''
        ):
    pid1: cython.int = os.getppid()
    pid2: cython.int = os.getpid()
    if logs_temp_path == '':
        logs_temp_path = TemporaryDirectory(prefix=this_app_codename).name
    log_file:    str = os.path.join(logs_temp_path, f'log_{logs_file_code}_{pid1}_{pid2}.txt')
    os.makedirs(logs_temp_path, exist_ok=True)
    Path(log_file).touch()
    return log_file

def log_append(
        str_to_append: str = '',
        log_file: str = '',
        logs_temp_path: str = '',
        logs_file_code: str = '',
        disable_logs: cython.bint=False,
        ):
    if not disable_logs:
        if log_file == '':
            init_logs(logs_temp_path, logs_file_code)
        with open(log_file, "a") as f:
            f.write(f"{str_to_append}\n")
            f.close()

def log_append_print(
        str_to_append: str = '',
        log_file: str = '',
        logs_temp_path: str = '',
        logs_file_code: str = '',
        disable_logs: cython.bint=False,
        quiet: cython.bint=False,
        loud: cython.bint=False
        ):
    if loud:
        print(str_to_append)
    if not disable_logs:
        if log_file == '':
            init_logs(logs_temp_path, logs_file_code)
        with open(log_file, "a") as f:
            f.write(f"{str_to_append}\n")
            f.close()
