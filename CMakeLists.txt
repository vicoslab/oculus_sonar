cmake_minimum_required(VERSION 3.16)
project(oculus_sonar VERSION 1.2)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    std_msgs
    sensor_msgs
    geometry_msgs
    nav_msgs
    tf2
    tf2_ros
    message_generation
    dynamic_reconfigure
)
generate_dynamic_reconfigure_options(
    cfg/OculusSonar.cfg
    cfg/SonarReprojectorConfig.cfg
    cfg/FanAssembler.cfg
)

find_package(oculus_driver QUIET)
if(NOT TARGET oculus_driver)
    include(FetchContent)
    FetchContent_Declare(oculus_driver
        GIT_REPOSITORY https://github.com/ENSTABretagneRobotics/oculus_driver.git
        GIT_TAG        v1.2
    )
    FetchContent_MakeAvailable(oculus_driver)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs)

add_message_files(FILES
   OculusHeader.msg
   OculusVersionInfo.msg
   OculusStatus.msg
   OculusFireConfig.msg
   OculusPing.msg
   OculusStampedPing.msg
   Raw.msg
   Ping.msg
   SonarConfig.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(DEPENDENCIES
    std_msgs
    sensor_msgs
)

catkin_package(CATKIN_DEPENDS roscpp rospy std_msgs sensor_msgs geometry_msgs nav_msgs tf2 tf2_ros)

# replace with target include directories ?
include_directories(${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

add_executable(oculus_sonar_node
    src/OculusNode.cpp
    src/oculus_sonar_node.cpp
)
target_include_directories(oculus_sonar_node PRIVATE src)
target_link_libraries(oculus_sonar_node PRIVATE
    ${catkin_LIBRARIES}
    oculus_driver
    ${OpenCV_LIBS}
)
add_dependencies(oculus_sonar_node oculus_sonar_generate_messages_cpp ${PROJECT_NAME}_gencfg)

find_package(rtac_display QUIET)
if(TARGET rtac_display)
    add_executable(oculus_sonar_display
        src/FanRendererES.cpp
        src/PingRenderer.cpp
        src/oculus_sonar_display.cpp
    )
    target_include_directories(oculus_sonar_display PRIVATE src)
    target_link_libraries(oculus_sonar_display PRIVATE
        ${catkin_LIBRARIES}
        oculus_driver
        rtac_display
    )
    add_dependencies(oculus_sonar_display oculus_sonar_generate_messages_cpp ${PROJECT_NAME}_gencfg)
endif()

file(
    GLOB PYTHON_NODES "scripts/*.py"
)
set(PYTHON_MODULES ${CMAKE_CURRENT_SOURCE_DIR}/scripts/sonar_geometry.py)
list(REMOVE_ITEM PYTHON_NODES ${PYTHON_MODULES})

catkin_install_python(
    PROGRAMS ${PYTHON_NODES}
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

file(MAKE_DIRECTORY ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_BIN_DESTINATION})

foreach(module ${PYTHON_MODULES})
    get_filename_component(name ${module} NAME)
    file(REMOVE ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_BIN_DESTINATION}/${name})
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            ${module}
            ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_BIN_DESTINATION}/${name}
    )
endforeach()

install(
    FILES ${PYTHON_MODULES}
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)