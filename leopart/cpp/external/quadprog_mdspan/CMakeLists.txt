set(HEADERS_external
  ${CMAKE_CURRENT_SOURCE_DIR}/QuadProg++.hh
  PARENT_SCOPE)

target_sources(${PROJECT_NAME} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/QuadProg++.cc
)