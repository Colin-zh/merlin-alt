import os
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install

__version__ = "0.1.0"

# with open("requirements.txt") as f:
#     require_packages = [line[:-1] if line[-1] == "\n" else line for line in f]

# class VerifyVersionCommand(install):
#     """Custom command to verify that the git tag matches our version"""
#     description = 'verify that the git tag matches our version'

#     def run(self):
#         tag = os.getenv('CIRCLE_TAG')

#         if tag != __version__:
#             info = "Git tag: {0} does not match the version of this app: {1}".format(
#                 tag, __version__
#             )
#             sys.exit(info)

setup(
    name='merlin',
    version=__version__,
    packages=find_packages(),
    author='Howl Zheng',
    author_email='colin_zh@outlook.com',
    # install_requires=require_packages,
    entry_points={
       'console_scripts': [
           "bert = merlin.charms.models.bert.__main__:train",
           "bert-vocab = merlin.charms.datasets.vocab:build"
       ]
   },
#    cmdclass={
#        'install': VerifyVersionCommand
#    }
)