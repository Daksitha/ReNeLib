# This is the codebase I have commited during the writing period of my thesis

# master_thesis
code base for my master thesis

Include to conda: 

# change log
1. ERROR: ffprobe was not able to find libiconv.so.2 no such file or directory
   conda install -c conda-forge libiconv
````commandline
            try:
                ffmpeg.input(...) \
                    .output(...) \
                    .run(capture_stdout=True, capture_stderr=True)
            except ffmpeg.Error as e:
                print('stdout:', e.stdout.decode('utf8'))
                print('stderr:', e.stderr.decode('utf8'))
                raise e
````


2. got an error running video Flame: AttributeError: module 'distutils' has no attribute 'version'
    updated the setup tools
    -pip install setuptools==59.5.0
