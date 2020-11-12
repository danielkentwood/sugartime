from setuptools import setup

dependencies = [
    'numpy',
    'scipy',
    'scikit-learn',
    'pandas',
    'fireTS',
    'plotly',
    'streamlit'
]

setup(
    name='sugartime',
    version='0.0.1',
    description='A python package for modeling and forecasting blood glucose dynamics in diabetics',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url='https://github.com/danielkentwood/sugartime',
    author='Daniel Wood',
    author_email='danielkentwood@gmail.com',
    license='MIT',
    packages=['sugartime'],
    install_requires=dependencies,
    include_package_data=True,
    zip_safe=False)
