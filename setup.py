from setuptools import setup

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='teg',
    version='0.1',
    description='Temporal Event Graphs',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/tbtraltaa/teg.git'
    packages=setuptools.find_packages('.'),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License::OSI Approved :: GNU General Public License v3 or later (GPLv3+)'
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        'Intended Audience :: Science/Researcg'
    ],
    keywords='temporal events',
    author='Altansuren Tumurbaatar',
    author_email='altaamgl@gmail.com',
    license='GNU General Public License v3 or later (GPLv3+)',
    packages=['teg'],
    python_requires='>=3.7',
    install_requires=[
        'networkx',
        'pandas',
        'numpy',
        'matplotlib',
        'scipy',
    ],
    include_package_data=True,
    zip_safe=False,
)
