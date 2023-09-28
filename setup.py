from distutils.core import setup

project_name = "Loop Model Scoring"
version = "0.1.0"
author = "Miriam K. Wolff"
author_email = "miriamkwolff@outlook.com"
package_name = "loop_model_scoring"  # this is the thing you actually import

setup(
    name=project_name,
    version=version,
    author=author,
    author_email=author_email,
    packages=[package_name],  # add subpackages too
    package_dir={package_name: package_name},
    long_description=open('README.md').read(),
    python_requires='>=3.6',
)
