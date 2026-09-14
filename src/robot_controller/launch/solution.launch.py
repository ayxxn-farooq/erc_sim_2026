from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    shelf_column = LaunchConfiguration('shelf_column_number')
    book_colour = LaunchConfiguration('book_colour')

    return LaunchDescription([

        DeclareLaunchArgument(
            'shelf_column_number',
            default_value='1'
        ),

        DeclareLaunchArgument(
            'book_colour',
            default_value='red'
        ),

        Node(
            package='robot_controller',
            executable='mission',
            name='erc_mission',
            output='screen',

            parameters=[{
                'shelf_column_number': shelf_column,
                'book_colour': book_colour,
            }]
        )
    ])
