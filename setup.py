from setuptools import setup, find_packages

def readme():
  with open('README.md', 'r', encoding="utf-8") as f:
    return f.read()

setup(
  name='pypharm',
  version='1.6.1',
  author='Krash13',
  author_email='krasheninnikov.r.s@muctr.ru',
  description='Module for solving pharmacokinetic problems',
  long_description=readme(),
  long_description_content_type='text/markdown',
  url='https://github.com/Krash13/PyPharm',
  packages=find_packages(),
  install_requires=['numpy>=1.22.1', 'scipy<=1.13.0', 'numba>=0.58.1',
                    'matplotlib>=3.5.1', 'graycode>=1.0.5', 'numbalsoda>=0.3.4'],
  classifiers=[
    'Programming Language :: Python :: 3.9',
    'License :: OSI Approved :: BSD License',
    'Operating System :: OS Independent'
  ],
  keywords='pharmacokinetics compartment-model',
  project_urls={
  },
  python_requires='>=3.9'
)
