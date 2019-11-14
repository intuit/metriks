from setuptools import setup, find_packages

with open('README.rst', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    # Package information
    name='metriks',
    description='metriks is a Python package of commonly used metrics for evaluating information retrieval models.',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url='https://github.com/intuit/metriks',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License'
    ],
    pyton_requires='>=3.6',

    # Package data
    use_scm_version={
        "write_to": "metriks/__version.py",
        "write_to_template": '__version__ = "{version}"\n',
    },
    packages=find_packages(exclude=('tests*',)),

    # Insert dependencies list here
    install_requires=[
        'numpy',
    ],
    setup_requires=["setuptools-scm"],
    extras_require={
        'dev': [
            'pytest',
            'tox',
        ]
    }
)
