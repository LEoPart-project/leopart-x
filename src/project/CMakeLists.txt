
set(HEADERS_project
  ${CMAKE_CURRENT_SOURCE_DIR}/l2project.h
  ${CMAKE_CURRENT_SOURCE_DIR}/PDEproject.h
  PARENT_SCOPE)

target_sources(${PROJECT_NAME} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/l2project.cpp
)
