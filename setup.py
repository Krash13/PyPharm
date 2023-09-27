from setuptools import setup, find_packages

def readme():
  with open('README.md', 'r') as f:
    return f.read()

setup(
  name='pypharm',
  version='1.0.0',
  author='Krash13',
  author_email='krasheninnikov.r.s@muctr.ru',
  description='Module for solving pharmacokinetic problems',
  long_description=readme(),
  long_description_content_type='text/markdown',
  url='https://github.com/Krash13/PyPharm',
  packages=find_packages(),
  install_requires=['numpy>=1.22.1', 'scipy>=1.8.0'],
  classifiers=[
    'Programming Language :: Python :: 3.9',
    'License :: OSI Approved :: BSD License'
    'Operating System :: OS Independent'
  ],
  keywords='pharmacokinetics compartment-model',
  project_urls={
  },
  python_requires='>=3.9'
)
