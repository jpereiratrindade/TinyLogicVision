file(REMOVE_RECURSE "${TEST_ROOT}")
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    QT_QPA_PLATFORM=offscreen QSG_RHI_BACKEND=software
    "${GUI}" --workspace "${TEST_ROOT}" --exercise-server
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "GUI workspace smoke failed (${result})\n${output}\n${error}")
endif()
foreach(entry IN ITEMS workspace.json uploads datasets models runs manifests)
  if(NOT EXISTS "${TEST_ROOT}/${entry}")
    message(FATAL_ERROR "GUI did not initialize workspace entry: ${entry}")
  endif()
endforeach()
file(READ "${TEST_ROOT}/workspace.json" manifest)
string(FIND "${manifest}" "tinylogicvision.workspace/v1" schema_position)
if(schema_position EQUAL -1)
  message(FATAL_ERROR "GUI workspace manifest schema missing")
endif()
