cmake_minimum_required(VERSION 3.20)
project(python_wizard_flite C)

set(FLITE_SOURCE "" CACHE PATH "Existing Flite source checkout")
if(NOT FLITE_SOURCE)
    include(FetchContent)
    FetchContent_Declare(flite
        GIT_REPOSITORY https://github.com/festvox/flite.git
        GIT_TAG 6c9f20dc915b17f5619340069889db0aa007fcdc)
    FetchContent_MakeAvailable(flite)
    set(FLITE_SOURCE "${flite_SOURCE_DIR}")
endif()

set(sources)
foreach(component utils regex hrg stats speech lexicon synth wavesynth cg)
    file(GLOB component_sources "${FLITE_SOURCE}/src/${component}/*.c")
    list(APPEND sources ${component_sources})
endforeach()
# These platform variants and the waveform vocoder are deliberately excluded.
list(FILTER sources EXCLUDE REGEX "cst_(file_(palmos|wince)|mmap_(posix|win32)|mlsa)\\.c$")
file(GLOB english "${FLITE_SOURCE}/lang/usenglish/*.c")
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../flite_voices.json" voice_registry)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/../flite_voices.json")
string(JSON voice_count LENGTH "${voice_registry}")
math(EXPR last_voice "${voice_count} - 1")
set(voice_declarations "")
set(voice_entries "")
foreach(index RANGE ${last_voice})
    string(JSON voice_id GET "${voice_registry}" ${index} id)
    string(JSON model GET "${voice_registry}" ${index} model)
    file(GLOB voice_sources "${FLITE_SOURCE}/lang/${model}/*.c")
    if(NOT voice_sources)
        message(FATAL_ERROR "No Flite source files for voice ${voice_id}: ${model}")
    endif()
    list(APPEND sources ${voice_sources})
    string(APPEND voice_declarations
        "extern cst_voice *register_${model}(const char *);\n"
        "extern void unregister_${model}(cst_voice *);\n")
    string(APPEND voice_entries
        "    {\"${voice_id}\", register_${model}, unregister_${model}},\n")
endforeach()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/flite_voices.h"
    "${voice_declarations}\nstatic const VoiceEntry voices[] = {\n${voice_entries}};\n")
foreach(name cmu_lts_rules cmu_lts_model cmu_lex cmu_lex_entries cmu_lex_data cmu_postlex)
    list(APPEND sources "${FLITE_SOURCE}/lang/cmulex/${name}.c")
endforeach()
list(APPEND sources ${english}
    "${FLITE_SOURCE}/src/audio/audio.c"
    "${FLITE_SOURCE}/src/audio/au_none.c"
    "${FLITE_SOURCE}/src/audio/au_streaming.c")

add_executable(python-wizard-flite flite_parameters.c ${sources})
target_include_directories(python-wizard-flite PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}"
    "${FLITE_SOURCE}/include" "${FLITE_SOURCE}/src/regex"
    "${FLITE_SOURCE}/src/cg" "${FLITE_SOURCE}/src/audio"
    "${FLITE_SOURCE}/lang/usenglish" "${FLITE_SOURCE}/lang/cmulex")
target_compile_definitions(python-wizard-flite PRIVATE CST_NO_SOCKETS)
if(MSVC)
    target_compile_definitions(python-wizard-flite PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(python-wizard-flite PRIVATE -O2)
    target_link_libraries(python-wizard-flite PRIVATE m)
    if(WIN32)
        target_link_options(python-wizard-flite PRIVATE -static)
    endif()
endif()
install(TARGETS python-wizard-flite RUNTIME DESTINATION .)
install(FILES "${FLITE_SOURCE}/COPYING" DESTINATION . RENAME FLITE-COPYING)
