#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Setup
"""

from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    readme_content = f.read()

with open('LICENSE', encoding='utf-8') as f:
    license_content = f.read()

setup(
    name='variables_collector', 
    version='1.0.0', 
    description='Variables collector', 
    long_description=readme_content,
    url='https://github.com/infodavide/data_collector',
    author='David R', 
    author_email='contact@infodavid.org', 
    license=license_content, 
    packages=find_packages(exclude=('tests', 'docs'), 
    zip_safe=False
)
