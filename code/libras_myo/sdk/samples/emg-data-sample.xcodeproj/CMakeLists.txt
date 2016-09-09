cmake_minimum_required(VERSION 3.5)
project(emg_data_sample_xcodeproj)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
    project.pbxproj)

add_executable(emg_data_sample_xcodeproj ${SOURCE_FILES})