# ---[ Test binaries.

file(GLOB_RECURSE CORETEN_ALL_TEST_FILES *.cpp)
if(BUILD_TEST)
    foreach(test_src ${CORETEN_ALL_TEST_FILES})
        get_filename_component(test_file_name ${test_src} NAME_WE)
        set(test_name "coreten_${test_file_name}")
        add_executable(${test_name} "${test_src}")
        target_link_libraries(${test_name} coreten gmock gtest gtest_main)
        add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)

        if(INSTALL_TEST)
            install(TARGETS ${test_name} DESTINATION testing)
        endif()
    endforeach()
endif()