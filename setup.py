from setuptools import setup, find_packages
import sys, os

VERSION = '0.2.0'

setup(name='mv_tracker',
      version=VERSION,
      description="movement tracker",
      long_description="""""",
      classifiers=[],
      keywords='movement tracker dbscan',
      author='o_poiron',
      author_email='o.poirion@gmail.com',
      url='',
      license='MIT',
      packages=find_packages(exclude=['examples', 'tests']),
      include_package_data=True,
      zip_safe=False,
      install_requires=[
          'sklearn',
          'cv2',
          'numpy',
      ],
      )
