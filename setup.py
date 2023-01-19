from setuptools import setup, find_packages


setup(
   name='miniature_voice',
   version='0.0.2',
   description='`Small` Speech to text model',
   author='Zurab Dzindzibadze',
   author_email='dzindzibadzezurabi@gmail.com',
   packages=find_packages(
      exclude=['scripts'], 
   ),
)