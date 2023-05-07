
from setuptools import setup
from setuptools import find_packages

long_description = '''
NetworkAnalysis is a package that facilitates the analysis of omics data using networks.
and is distributed under the MIT license.
'''

setup(name='NetworkAnalysis',
      version='0.1.0',
      description='General omics data analysis tools using networks',
      long_description=long_description,
      author='Maarten Larmuseau',
      author_email='maarten.larmuseau@ugent.be',
      url='https://github.ugent.be/mlarmuse/NetworkAnalysis',
      license='MIT',
      install_requires=[
			'scikit-learn',
			'matplotlib',
			'lifelines',
			'networkx',
			'umap',
			'statsmodels',
			'seaborn',
			'joblib',
			'gseapy',
			'pytest',
			'coverage',
			'coverage_badge'
					],

      classifiers=[
         "Development Status :: 3 - Alpha",

      ],
packages=find_packages())
