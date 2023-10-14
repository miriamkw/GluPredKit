from setuptools import setup, find_packages

project_name = "GluPredKit"
version = "0.0.5"
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
    package_data={
        'glupredkit': ['config.json'],
    },
    entry_points={
        'console_scripts': [
            'glupredkit = glupredkit.cli:cli',
        ],
    }
)

