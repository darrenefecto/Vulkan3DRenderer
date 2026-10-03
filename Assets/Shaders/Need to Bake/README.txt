glslc belongs to VulkanSDK, for instance my path: C:\VulkanSDK\<version>\Bin\glslc.exe

glslc Assets/Shaders/mesh.vert -o Assets/Shaders/mesh.vert.spv
glslc Assets/Shaders/mesh.frag -o Assets/Shaders/mesh.frag.spv
glslc Assets/Shaders/skybox.vert -o Assets/Shaders/skybox.vert.spv
glslc Assets/Shaders/skybox.frag -o Assets/Shaders/skybox.frag.spv

Or:

C:\VulkanSDK\<version>\Bin\glslc.exe Assets/Shaders/mesh.vert -o Assets/Shaders/mesh.vert.spv
C:\VulkanSDK\<version>\Bin\glslc.exe Assets/Shaders/mesh.frag -o Assets/Shaders/mesh.frag.spv
C:\VulkanSDK\<version>\Bin\glslc.exe Assets/Shaders/skybox.vert -o Assets/Shaders/skybox.vert.spv
C:\VulkanSDK\<version>\Bin\glslc.exe Assets/Shaders/skybox.frag -o Assets/Shaders/skybox.frag.spv