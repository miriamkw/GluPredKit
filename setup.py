from setuptools import setup, find_packages
from setuptools.command.install import install
import os

project_name = "GluPredKit"
version = "0.0.3"
author = "Miriam K. Wolff"
author_email = "miriamkwolff@outlook.com"
package_name = "glupredkit"  # The package name on pip install

with open('README.md', encoding='utf-8') as f:
    readme = f.read()


class CustomInstall(install):
    def run(self):
        # Get path of current working directory
        cwd = os.getcwd()

        print("Creating directories...")

        folder_path= 'data'
        folder_names = ['raw', '.processed', '.trained_models', 'figures', 'reports']

        for folder_name in folder_names:
            path = os.path.join(cwd, folder_path, folder_name)
            os.makedirs(path, exist_ok=True)

        print("Directories created for usage of GluPredKit.")

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
    long_description_content_type="text/markdown",
    python_requires='>=3.6',
    license='MIT',
    install_requires=[
        "matplotlib>=3.6.3",
        "pandas>=1.5.3",
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
    package_data={
        'glupredkit': ['config.json'],
    },
    entry_points={
        'console_scripts': [
            'glupredkit = glupredkit.cli:cli',
        ],
    }
)

