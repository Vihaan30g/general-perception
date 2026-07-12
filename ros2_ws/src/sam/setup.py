from setuptools import setup, find_packages

package_name = 'sam'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vihaan',
    maintainer_email='vihaan30g@gmail.com',
    description='SAM2 segmentation stage of the perception pipeline',
    license='Apache-2.0',
    tests_require=['pytest'],
    scripts=[
        'scripts/sam_node.py',
    ],
)