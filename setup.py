from setuptools import find_packages,setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements = []
    with open(file_path,'r') as f:
        requirements = f.read().splitlines()

setup(
name='InernProject',
version='0.0.1',
author='Ayoub Ben Mahmoud',
author_email='ayoubb917@gmail.com',
packages=find_packages(),   
install_requires=get_requirements('requirements.txt')
)