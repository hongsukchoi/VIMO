from setuptools import setup, find_packages

print('Found packages:', find_packages())
setup(
    description='VIMO as a package',
    name='vimo',
    packages=find_packages(),
    install_requires=[
        'pulp',
        'supervision',
        'open3d',
        'opencv-python',
        'loguru',
        # 'git+https://github.com/mattloper/chumpy',
        'einops',
        'plyfile',
        'pyrender',
        # 'segment_anything',
        'scikit-image',
        'smplx',
        # 'timm==0.6.7',
        'evo',
        'pytorch-minimize',
        'imageio[ffmpeg]',
        # 'numpy==1.23',
    ],
    extras_require={
        'all': [
        ],
    },
)
