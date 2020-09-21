from distutils.core import setup


setup(
    name='sugarTS',
    version='0.1dev',
    description='sugarTS is an application that assists'
                'Type-1 diabetics in maintaining their'
                'blood glucose within range.',
    author='Daniel Wood',
    author_email='danielkentwood@gmail.com',
    packages=['sugarTS', 'sugarTS.test'],
    url='http://pypi.python.org/pypi/sugarTS/',
    license='LICENSE.txt',
    install_requires=[],
    classifiers=list,
    long_description=open('README.md').read(),
    include_package_data=bool,
    zip_safe=bool,
    keywords=str,
    entry_points=dict
)