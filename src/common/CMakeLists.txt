set(HEADERS_common
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.h
  ${CMAKE_CURRENT_SOURCE_DIR}/dolfinxutils.h
  PARENT_SCOPE)

target_sources(${PROJECT_NAME} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/dolfinxutils.cpp
)