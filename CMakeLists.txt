cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project (SOGP)

if (NOT CMAKE_BUILD_TYPE)
	message(STATUS "No build type selected, default to Release")
	set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-msse2 -msse3 -mssse3 -O3 -fPIC")
set(CMAKE_LINKER_FLAGS_RELEASE "-fPIC")

set(NEWMAT_SOURCE_DIR)

include_directories(${SOGP_SOURCE_DIR}/lib)
include_directories(${SOGP_SOURCE_DIR}/newmat)

# source files for newmat
file(GLOB NEWMAT_SOURCE_FILES
    "newmat/*.c" "newmat/*.cpp"
)

# source files for SOGP
set(SOGP_SOURCE_FILES "${SOGP_SOURCE_DIR}/lib/SOGP.cc" "${SOGP_SOURCE_DIR}/lib/SOGP_aux.cc")


set(ALL_SOURCE_FILES ${SOGP_SOURCE_FILES} ${NEWMAT_SOURCE_FILES})


## library itself
ADD_LIBRARY( SOGP STATIC ${ALL_SOURCE_FILES} )

INSTALL_TARGETS(/lib SOGP)
INSTALL(FILES ${CMAKE_SOURCE_DIR}/lib/SOGP.h ${CMAKE_SOURCE_DIR}/lib/SOGP_aux.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include )

#  only if cjbTest is to be built
#include_directories(/data/phd/bitbucket/sqb/include)

## add tests
add_executable( SOGPTest "${SOGP_SOURCE_DIR}/tests/SOGPtest.cc" )
#add_executable( cjbTest "${SOGP_SOURCE_DIR}/tests/cjbTest.cc" )

TARGET_LINK_LIBRARIES( SOGPTest SOGP )
#TARGET_LINK_LIBRARIES( cjbTest SOGP )