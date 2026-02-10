from setuptools import find_packages, setup

package_name = 'cave_explorer'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/cave_explorer.launch.py', 'launch/mapping.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='oguz',
    maintainer_email='oguzhan.esen@tum.de',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'cave_explorer = cave_explorer.cave_explorer_node:main',
            'depth_to_pointcloud = cave_explorer.depth_to_pointcloud:main',
        ],
    },
)
