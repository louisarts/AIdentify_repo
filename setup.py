from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='aidentify',
      version="0.0.0",
      description="AIdentify model",
      license="MIT",
      author="Alba Gutierrez",
      author_email="albaguti@gmail.com",
      url="localhost:8000",
      install_requires=requirements,
      packages=find_packages(),
      test_suite="tests",
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
