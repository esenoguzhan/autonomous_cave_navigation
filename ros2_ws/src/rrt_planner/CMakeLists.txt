project(rrt_planner)
cmake_minimum_required(VERSION 3.5)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(octomap_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(visualization_msgs REQUIRED)

# OMPL
find_library(OMPL_LIB ompl PATHS "/opt/ros/jazzy/lib/x86_64-linux-gnu" "/usr/lib/x86_64-linux-gnu" NO_DEFAULT_PATH)
if (NOT OMPL_LIB)
    find_library(OMPL_LIB ompl)
endif()

find_package(octomap REQUIRED)

add_executable(rrt_planner_node src/rrt_planner_node.cpp)
target_include_directories(rrt_planner_node PRIVATE "/opt/ros/jazzy/include/ompl-1.7")

ament_target_dependencies(rrt_planner_node
    rclcpp
    geometry_msgs
    nav_msgs
    octomap_msgs
    tf2
    tf2_ros
    tf2_geometry_msgs
    std_msgs
    std_srvs
    visualization_msgs
)
target_link_libraries(rrt_planner_node 
    ${OMPL_LIB}
    ${OCTOMAP_LIBRARIES}
)

install(TARGETS
    rrt_planner_node
    DESTINATION lib/${PROJECT_NAME})

ament_package()
