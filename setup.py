from setuptools import setup, find_packages

setup(
    name='semantic-chunker',
    version='0.1.0',
    description='Semantic chunking and clustering for RAG systems',
    author='Your Name',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'networkx',
    ],
    entry_points={
        'console_scripts': [
            'semantic-chunker = cli:main',
        ],
    },
    python_requires='>=3.7',
)
