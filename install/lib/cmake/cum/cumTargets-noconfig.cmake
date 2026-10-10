#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "cum::cum_MKL" for configuration ""
set_property(TARGET cum::cum_MKL APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(cum::cum_MKL PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CXX"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libcum_MKL.a"
  )

list(APPEND _cmake_import_check_targets cum::cum_MKL )
list(APPEND _cmake_import_check_files_for_cum::cum_MKL "${_IMPORT_PREFIX}/lib/libcum_MKL.a" )

# Import target "cum::Core" for configuration ""
set_property(TARGET cum::Core APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(cum::Core PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CXX"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libcum_Core.a"
  )

list(APPEND _cmake_import_check_targets cum::Core )
list(APPEND _cmake_import_check_files_for_cum::Core "${_IMPORT_PREFIX}/lib/libcum_Core.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
