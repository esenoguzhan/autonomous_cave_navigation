project(trajectory_generator)
cmake_minimum_required(VERSION 3.5)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(trajectory_planner REQUIRED) # For ExecuteTrajectory.srv
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

add_executable(trajectory_generator_node src/trajectory_generator_node.cpp)
ament_target_dependencies(trajectory_generator_node
    rclcpp
    trajectory_msgs
    nav_msgs
    geometry_msgs
    std_msgs
    tf2
    tf2_geometry_msgs
    trajectory_planner
)

install(TARGETS
    trajectory_generator_node
    DESTINATION lib/${PROJECT_NAME}
)

ament_package()
