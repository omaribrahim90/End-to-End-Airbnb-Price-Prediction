from setuptools import setup, find_packages



HYPEN_DASHES = '-e .'


def get_requirements(file_path):
    with open(file_path) as f:
        requirements = f.read().splitlines()
    return [req for req in requirements if req != HYPEN_DASHES]


setup(
    name='End_to_End_ML_Project',
    version='0.0.1',
    author="Omar",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)