if(AMReX_SPACEDIM GREATER_EQUAL 3)
  quokka_add_problem(JOB_NAME CosmoSphereDM)
endif()

# Possibility of include explicitly a test for Quokka

# # Define the name of the executable
# set(JOB_NAME CosmologicalExpansion)

# # Build the executable by merging the.cpp with the Quokka base sources
# # Note: ${QuokkaObjSources} contains the main and the basic idro/cosmo modules
# add_executable(${JOB_NAME} testCosmologicalExpansion.cpp ${QuokkaObjSources})

# # Configuration of GPU
# if(AMReX_GPU_BACKEND MATCHES "CUDA")
#   setup_target_for_cuda_compilation(${JOB_NAME})
# endif()

# # Make th problem available in the list 'ctest -R'
# add_test(
#   NAME ${JOB_NAME}
#   COMMAND ${JOB_NAME} inputs.in
#   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
