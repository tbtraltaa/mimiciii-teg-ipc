import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='mimiciii_teg',
    version='1.0',
    description='MIMIC III - Temporal Event Graphs and Inverse Percolation Centrality',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/tbtraltaa/mimiciii-teg-ipc',
    packages=setuptools.find_packages('.'),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License::OSI Approved :: GNU '
            'General Public License v3 or later (GPLv3+)'
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        'Intended Audience :: Science/Research'
    ],
    keywords='temporal events',
    author='Altansuren Tumurbaatar',
    author_email='altaamgl@gmail.com',
    license='GNU General Public License v3 or later (GPLv3+)',
    python_requires='>=3.9',
    install_requires=[
        'networkx==2.8.4',
        'pandas==1.5.1',
        'numpy<1.21',
        'scipy>1.9.1',
        'pyvis==0.2.0',
        #'hdf5>=1.10.6',
        'psycopg2==2.8.6',
        #'pytables>=3.6.1',
        #'seaborn',
        'plotly==5.9.0',
        'psmpy==0.3.13'
        #'pygraphblas==5.1.8.0',
        #'suitesparse-graphblas==5.1.8.0'
    ],
    include_package_data=False,
    zip_safe=False,
)
