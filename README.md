# Vulkan 3D Renderer

A modern 3D rendering application written in **C++20** using **Vulkan 1.4**. The renderer provides a flexible foundation for real-time 3D graphics, featuring hybrid rendering, Forward+ lighting, custom rendering pipelines, shadow mapping, physically based rendering (PBR), and glTF/GLB material texturing.

## Features

- **Vulkan-based 3D rendering**
- **C++20**
- **Visual Studio 2026**
- **Hybrid rendering architecture**
- **Forward+ rendering and clustered lighting**
- **Custom rendering pipelines and shaders**
- **Shadow mapping and local shadows**
- **Physically Based Rendering (PBR)**
- **glTF and GLB model and material loading**
- **PBR texture support**, including base color, metallic-roughness, normal, ambient occlusion, and emissive maps
- Custom mesh loading
- Real-time camera movement
- Texture loading and mapping
- Skybox rendering
- GLM for mathematics
- GLFW3 for window and input management

## Screenshots

### Frame 1

![Frame 1](screenshots/frame1.jpg)

### Frame 2

![Frame 2](screenshots/frame2.png)

## Technology Stack

| Technology | Purpose |
|---|---|
| **C++20** | Core programming language |
| **Vulkan 1.4** | Low-level graphics and rendering API |
| **GLFW3** | Window creation, input, and Vulkan surface handling |
| **GLM** | Mathematics, vectors, matrices, and transformations |
| **Assimp** | 3D model importing, including glTF and GLB |
| **stb_image** | Image loading and texture processing |
| **Visual Studio 2026** | Development environment |

The project targets modern versions of the graphics libraries and development tools.

## Rendering Features

### Hybrid Rendering

The renderer combines multiple rendering techniques in a flexible architecture, allowing different rendering passes and lighting approaches to work together within the same frame.

### Forward+ Rendering

Forward+ rendering uses light culling to identify relevant lights for scene regions, reducing unnecessary lighting calculations and supporting scenes with multiple light sources.

### Custom Rendering

Custom rendering pipelines and shaders provide control over rendering passes, material appearance, lighting calculations, and other graphics effects.

### Physically Based Rendering (PBR)

The renderer supports a metallic-roughness PBR workflow designed to produce consistent, material-dependent lighting under different illumination conditions.

Supported material properties include:

- Base color and base-color textures
- Metallic and roughness factors
- Metallic-roughness textures
- Tangent-space normal maps
- Ambient occlusion
- Emissive color and textures

### glTF and GLB Material Loading

The renderer imports 3D models and their material properties from glTF and GLB assets. PBR textures and material factors can be used to reproduce the appearance intended by the original asset.

### Shadow Mapping

Shadow mapping provides depth-based shadow information for objects illuminated by supported light sources, adding depth and spatial definition to the scene.

### Custom Meshes

The renderer loads custom 3D meshes, allowing scenes to be built from external geometry rather than being limited to predefined primitives.

### Shaders

Custom Vulkan shaders provide flexibility for implementing rendering techniques, material effects, and lighting models.

### Camera Movement

The camera can be moved through the 3D environment in real time, allowing the scene to be inspected from different perspectives.

### Texturing

Textures provide surface detail and material information. Depending on the material, they can control color, roughness, metallic response, surface normals, ambient occlusion, and emission.

### Skybox

A cubemap-based skybox surrounds the scene with an environment texture, providing a background for the rendered world.

## Vendors

The project includes or uses the following dependencies in its `vendors` directory:

- **Assimp**
- **GLFW3**
- **GLM**
- **stb_image**

## Requirements

To build and run the project, you need:

- **Visual Studio 2026**
- A compiler with **C++20** support
- **Vulkan SDK 1.4**
- A Vulkan-compatible GPU and up-to-date graphics drivers

Make sure the Vulkan SDK and required dependencies are correctly installed and configured in the build environment.

## Build

Open the project in **Visual Studio 2026** and build it using the appropriate configuration.

Before running the application, make sure that:

1. Your GPU supports the required Vulkan features.
2. Your Vulkan drivers are up to date.
3. The Vulkan SDK is installed and configured.
4. All required dependencies are available.
5. Runtime assets, including meshes, textures, shaders, and skybox textures, are available at the expected paths.

## Project Overview

The renderer follows a Vulkan-based rendering architecture that combines GPU resources, shaders, meshes, materials, lighting, shadows, and camera transformations to produce the final image.

A typical frame consists of:

1. Processing user input and updating the camera.
2. Preparing scene data and object transformations.
3. Updating lighting and shadow information.
4. Preparing material resources and PBR textures.
5. Performing light culling for the Forward+ rendering path.
6. Recording Vulkan rendering commands for the relevant rendering passes.
7. Rendering scene geometry, applying lighting and materials, and drawing the skybox.
8. Presenting the final image to the screen.