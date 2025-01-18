from setuptools import setup, find_packages
from typing import List


def get_requirements() -> List[str]:
    """
    Reads the requirements.txt file and returns a list of requirements.
    Excludes '-e .' for editable installations.
    """
    requirement_list: List[str] = []
    try:
        with open('requirements.txt') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                # Ignore empty lines and '-e .' entry
                if line and line != '-e .':
                    requirement_list.append(line)
    except Exception as e:
        print(f"Error: {e}")
    return requirement_list


setup(
    name='WineQualityPrediction',
    version='0.1',
    author='Md Nasir Uddin',
    author_email='nasir.uddin.6314@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements(),  # Correctly processed list of requirements
)
