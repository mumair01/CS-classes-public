import os
from glob import glob
from setuptools import setup

package_name = 'ldm_unit'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py')),
        # (os.path.join('share', package_name),
        #  glob('ldm_unit/decision_nodes/*.py')),
        # (os.path.join('share', package_name),
        #  glob('ldm_unit/percept_nodes/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ldm_node = ldm_unit.ldm_node:main',
            'audio_decision_node = ldm_unit.audio_decision_node:main',
            'video_decision_node = ldm_unit.video_decision_node:main',
            'camera_percept_node = ldm_unit.camera_percept_node:main',
            # 'mic_percept_node = ldm_unit.mic_percept_node:main',
        ],
    },
)
