from setuptools import setup
import os
from glob import glob

package_name = 'dodo_controller'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'policy'), glob('dodo_controller/policy/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Dodo Team',
    maintainer_email='dodo@example.com',
    description='PPO policy controller for Dodo quadruped robot',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dodo_policy_node = dodo_controller.dodo_policy_node:main',
        ],
    },
)
