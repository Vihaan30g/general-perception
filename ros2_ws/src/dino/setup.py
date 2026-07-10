from setuptools import setup, find_packages

package_name = 'dino'

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
    description='Grounding DINO stage of the perception pipeline',
    license='Apache-2.0',
    tests_require=['pytest'],
    # These live in scripts/ (per project convention) rather than being
    # console_scripts entry_points inside the dino/ package. They still run
    # via `ros2 run dino <name>.py` - note the required .py suffix, since
    # setuptools' `scripts=` installs them under their original filename.
    scripts=[
        'scripts/grounding_dino_node.py',
        'scripts/prompt_manager_node.py',
        'scripts/set_prompt.py',
    ],
)