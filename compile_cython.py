from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from Cython.Build import cythonize

sourcefiles = [
    "justai_video_processing_lib.py",
    "justai_video_processor_logs.py",
    "justai_video_processor.py",
    "justai_video_processing_lib_fps_magic.py",
    "justai_video_processing_lib_frames_map.py"
]

ext_modules=[
    Extension("justai_video_processor_logs", ["justai_video_processor_logs.py"]),
    Extension("justai_video_processing_lib_frames_map", ["justai_video_processing_lib_frames_map.py"]),
    Extension("justai_video_processing_lib_fps_magic", ["justai_video_processing_lib_fps_magic.py"]),
    Extension("justai_video_processing_lib", ["justai_video_processing_lib.py"]),
    Extension("justai_video_processor", ["justai_video_processor.py"]),
]

setup(
    name='JustAI Video Processor',
    ext_modules = cythonize(ext_modules),
    #ext_modules=[Extension("justai_video_processor", sourcefiles)],
    #cmdclass = {'build_ext': build_ext}
)

