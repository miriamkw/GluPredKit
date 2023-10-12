from setuptools import setup, find_packages

project_name = "GluPredKit"
version = "0.0.1"
author = "Miriam K. Wolff"
author_email = "miriamkwolff@outlook.com"
package_name = "glupredkit"  # The package name on pip install

with open('README.md', encoding='utf-8') as f:
    readme = f.read()

setup(
    name=project_name,
    version=version,
    author=author,
    author_email=author_email,
    packages=find_packages(),
    package_dir={package_name: package_name},
    long_description=readme,
    python_requires='>=3.6',
    install_requires=[
        "matplotlib==3.6.3",
        "pandas==1.5.3",
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
    entry_points={
        'console_scripts': [
            'glupredkit = glupredkit.cli:cli',
        ],
    }
)
