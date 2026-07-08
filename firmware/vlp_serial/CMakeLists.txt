cmake_minimum_required(VERSION 3.12)
project(vlp_pico C CXX ASM)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

add_executable(vlp_pico "")

target_include_directories(vlp_pico PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
)

set_target_properties(vlp_pico PROPERTIES
    COMPILE_FLAGS "-fno-rtti -fno-exceptions -fno-threadsafe-statics"
)

target_sources(vlp_pico PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/main.cpp
    ${CMAKE_CURRENT_LIST_DIR}/model_data.cc
)

target_link_libraries(vlp_pico
    pico-tflmicro
    pico_stdlib
)

pico_enable_stdio_usb(vlp_pico 1)
pico_enable_stdio_uart(vlp_pico 0)
pico_add_extra_outputs(vlp_pico)
