
add_executable(nn-planar-data src/main.cpp)
target_link_libraries(nn-planar-data tensor nn)
set_target_properties(nn-planar-data PROPERTIES LINKER_LANGUAGE CXX)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/resources DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/visualize_nn_planar_data.py DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/)
file(RENAME ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/visualize_nn_planar_data.py ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/visualize-nn-planar-data)
