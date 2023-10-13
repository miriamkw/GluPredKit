from setuptools import setup, find_packages
from setuptools.command.install import install
import os

project_name = "GluPredKit"
version = "0.0.1"
author = "Miriam K. Wolff"
author_email = "miriamkwolff@outlook.com"
package_name = "glupredkit"  # The package name on pip install

with open('README.md', encoding='utf-8') as f:
    readme = f.read()


class CustomInstall(install):
    def run(self):
        # Create directories
        cwd = os.getcwd()

        folder_path= 'data'
        folder_names = ['raw', '.processed', '.trained_models', 'figures', 'reports']

        for folder_name in folder_names:
            path = os.path.join(cwd, folder_path, folder_name)
            os.makedirs(path, exist_ok=True)

        # Call the original installation command
        super().run()


setup(
    name=project_name,
    version=version,
    author=author,
    author_email=author_email,
    packages=find_packages(),
    package_dir={package_name: package_name},
    long_description=readme,
    python_requires='>=3.6',
    license='MIT',
    install_requires=[
        "matplotlib>=3.6.3",
        "pandas>=1.5.3",
        "tidepool_data_science_project @ git+https://github.com/miriamkw/data-science-tidepool-api-python.git@0.2",
        "python_nightscout @ git+https://github.com/ps2/python-nightscout.git@master",
        "scikit-learn",
        "xgboost",
        "tensorflow",
        "keras-tcn",
        "numpy",
        "aiohttp",
        "click",
        "dill"
    ],
    cmdclass={
        'install': CustomInstall,
    },
    entry_points={
        'console_scripts': [
            'glupredkit = glupredkit.cli:cli',
        ],
    }
)

