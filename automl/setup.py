#!/usr/bin/env python

from setuptools import setup

setup(name="lowrank-automl",
      version="0.1",
      author="Yuji Akimoto, Chengrun Yang, Dae Won Kim",
      author_email="ya242@cornell.edu",
      packages=["automl"],
      package_dir={"automl": "automl"},
      url="https://github.com/yujiakimoto/lowrankautoml/",
      license="MIT",
      install_requires=["numpy >= 1.8",
                        "scipy >= 0.13",
                        "sklearn >= 0.18",
                        "pysmac >= 0.9",
                        "pathos >= 0.2.0"]
      )
