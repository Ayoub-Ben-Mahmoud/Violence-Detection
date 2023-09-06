from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements = []
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:  # Use the variable HYPEN_E_DOT, not "HYPEN_E_DOT"
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name='InternProject',
    version='0.0.1',
    author='Ayoub Ben Mahmoud',
    author_email='ayoubb917@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
