cmake_minimum_required(VERSION 3.8)
project(sampling_based_traj_gen)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(octomap REQUIRED)
find_package(octomap_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(trajectory_msgs REQUIRED) # <-- BURAYI EKLE
find_package(trajectory_planner REQUIRED) 

add_executable(sampling_planner_node src/sampling_planner_node.cpp)

target_include_directories(sampling_planner_node PRIVATE
  ${Eigen3_INCLUDE_DIRS}
  ${OCTOMAP_INCLUDE_DIRS}
)

ament_target_dependencies(sampling_planner_node
  rclcpp
  nav_msgs
  geometry_msgs
  std_msgs
  tf2
  tf2_geometry_msgs
  octomap_msgs
  trajectory_msgs
  trajectory_planner
)

target_link_libraries(sampling_planner_node
  ${OCTOMAP_LIBRARIES}
)

install(TARGETS sampling_planner_node DESTINATION lib/${PROJECT_NAME})

ament_package()
