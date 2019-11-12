from setuptools import setup, find_packages

setup(
    # Package information
    name='metriks',

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
