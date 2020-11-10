set(HEADERS_advect
  ${CMAKE_CURRENT_SOURCE_DIR}/advect.h
  PARENT_SCOPE)

target_sources(${PROJECT_NAME} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/advect.cpp
)
