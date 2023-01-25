"""
Build the wheel file for the library eks-ml-pipeline.
"""

from setuptools import find_packages, setup

setup(
    name='eks_ml_pipeline',
    version='0.0.1',
    description='pipeline for eks',
    url='',
    author= 'Hamza Khokhar, Praveen Mada',
    author_email='hamza.khokhar@dish.com, praveen.mada@dish.com',
    license='Dish Wireless',
    packages=find_packages(include=['eks_ml_pipeline',
                                    'eks_ml_pipeline.feature_engineering',
                                    'eks_ml_pipeline.models',
                                    'eks_ml_pipeline.inputs',
                                    'eks_ml_pipeline.utilities']),
    include_package_data=True,
    install_requires = [
        'pyspark',
        'pandas',
        'keras',
        'tensorflow',
        'matplotlib',
        'numpy',
        'dill',
        'scikit_learn',
        'statsmodels'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Dish Wireless',
        ],
    )
