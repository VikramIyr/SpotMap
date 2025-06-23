cmake_minimum_required(VERSION 3.8)
project(spot_image_sync)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(message_filters REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)

# === rgbd_sync_node ===
add_executable(rgbd_sync_node src/rgbd_sync.cpp)
target_compile_features(rgbd_sync_node PRIVATE cxx_std_17)
ament_target_dependencies(rgbd_sync_node
  rclcpp
  message_filters
  sensor_msgs
  geometry_msgs
  tf2
  tf2_ros
)

# === Install targets ===
install(TARGETS
  rgbd_sync_node
  DESTINATION lib/${PROJECT_NAME}
)

# === Linting (optional) ===
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_cpplint_FOUND TRUE)
  set(ament_cmake_copyright_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
