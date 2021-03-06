from setuptools import setup

setup(
    name='depressionAnalysis',
    version='0.2.2',    
    description='Detecting depression in social media posts',
    long_description="Read the wiki at https://github.com/Sam-Nielsen-Dot/depressionAnalysis/wiki",
    url='https://github.com/Sam-Nielsen-Dot/depressionAnalysis',
    author='Sam Nielsen',
    author_email='lenssimane@gmail.com',
    license='MIT',
    packages=['depressionAnalysis'],
    install_requires=['twint>=2.1.20',
                    'nltk>=3.5',
                    'requests>=2.24.0',
                    'pandas>=1.2.4',                  
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Healthcare Industry',
        'Environment :: Win32 (MS Windows)',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    include_package_data=True,
    package_data={'': ['depressionAnalysis/data/*.pickle']},
)
