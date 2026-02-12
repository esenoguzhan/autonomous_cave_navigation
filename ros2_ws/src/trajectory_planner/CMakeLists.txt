cmake_minimum_required(VERSION 3.8)
project(trajectory_planner)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# C++14 standard (required for mav_trajectory_generation)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(mav_msgs REQUIRED)
find_package(mav_planning_msgs REQUIRED)
find_package(mav_trajectory_generation REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/ExecuteTrajectory.srv"
  DEPENDENCIES geometry_msgs
)

# Find glog and nlopt from mav_trajectory_generation install directory
# In a colcon workspace, packages are installed under install/<package_name>
get_filename_component(WORKSPACE_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}" DIRECTORY)
set(MAV_TRAJ_GEN_LIB_DIR "${WORKSPACE_INSTALL_DIR}/mav_trajectory_generation/lib")

if(EXISTS "${MAV_TRAJ_GEN_LIB_DIR}")
  link_directories("${MAV_TRAJ_GEN_LIB_DIR}")
  message(STATUS "Adding mav_trajectory_generation lib dir: ${MAV_TRAJ_GEN_LIB_DIR}")
else()
  message(WARNING "Could not find mav_trajectory_generation lib dir: ${MAV_TRAJ_GEN_LIB_DIR}")
endif()

# Trajectory planner node
add_executable(trajectory_planner_node src/trajectory_planner_node.cpp)

target_include_directories(trajectory_planner_node PRIVATE
  ${EIGEN3_INCLUDE_DIRS}
)

ament_target_dependencies(trajectory_planner_node
  rclcpp
  std_msgs
  std_srvs
  geometry_msgs
  nav_msgs
  trajectory_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  mav_msgs
  mav_planning_msgs
  mav_trajectory_generation
  Eigen3
)

rosidl_target_interfaces(trajectory_planner_node
  ${PROJECT_NAME} "rosidl_typesupport_cpp")

# Link glog and nlopt explicitly
target_link_libraries(trajectory_planner_node
  glog
  nlopt
)

# Install executables
install(TARGETS
  trajectory_planner_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install config files
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
