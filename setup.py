from setuptools import setup, find_packages

setup(
  name = 'scram-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.2.0',
  license='MIT',
  description = 'Scram Optimizer - Pytorch',
  author = 'Ross Morgan-Linial',
  author_email = 'ross.morgan.linial@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/RossM/scram-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'optimizers'
  ],
  install_requires=[
    'torch>=1.6'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)