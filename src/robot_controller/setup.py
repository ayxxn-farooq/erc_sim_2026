from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'robot_controller'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),

    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')
        ),
    ],

    install_requires=['setuptools'],
    zip_safe=True,

    maintainer='ros',
    maintainer_email='ros@todo.todo',

    description='ERC 2026 Phase 1 autonomous book retrieval solution',
    license='TODO',

    tests_require=['pytest'],

    entry_points={
        'console_scripts': [
            'robot = robot_controller.main:main',
            'mission = robot_controller.mission:main',
        ],
    },
)
