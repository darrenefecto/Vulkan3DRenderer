#pragma once
#include "VulkanContext.h"
#include "Light.h"
#include <glm/glm.hpp>

// ---------------------------------------------------------------------------
// Clustered lighting (Forward+)
//
// The view frustum is split into a CLUSTER_GRID x/y/z grid of clusters.
// A compute pass (cluster_build.comp) assigns every point/spot light to the
// clusters it overlaps; shading shaders then only iterate the lights of the
// cluster they fall into.
//
// The light list lives in an SSBO that is independent of the legacy
// GlobalUBO limits, so the clustered path supports far more lights than the
// forward/deferred UBO path.
// ---------------------------------------------------------------------------
namespace ClusterLimits {
    constexpr uint32_t GRID_X = 16;
    constexpr uint32_t GRID_Y = 9;
    constexpr uint32_t GRID_Z = 24;
    constexpr uint32_t CLUSTER_COUNT = GRID_X * GRID_Y * GRID_Z;
    constexpr uint32_t MAX_PER_CLUSTER = 64;    // must match cluster_build.comp
    constexpr uint32_t MAX_LIGHTS = 256;        // points + spots combined
}

// Must match `struct ClusterLight` in cluster_build.comp / forwardplus.frag.
struct ClusterLight {
    glm::vec4 posRange;   // xyz = world pos, w = range
    glm::vec4 color;      // rgb = color*intensity, w = shadow index (-1 = none)
    glm::vec4 dirInner;   // xyz = direction (spots), w = innerCos
    glm::vec4 outerType;  // x = outerCos, y = type (0 = point, 1 = spot)
};

// Push constants shared by cluster_build.comp and forwardplus.frag
// (112 bytes; forwardplus.frag reads them at offset 64, after the model push).
struct ClusterParams {
    glm::mat4 view;
    glm::vec4 screen;   // w, h, tileW, tileH (pixels)
    glm::vec4 zparams;  // near, far, tanHalfFovX, tanHalfFovY
    glm::ivec4 dims;    // gridX, gridY, gridZ, lightCount
};
static_assert(sizeof(ClusterParams) == 112, "ClusterParams must fit the 128-byte push limit");

// Builds the ClusterLight array from the light system. Shadow indices match
// the slot assignment used by LightSystem::fillGlobalUBO (deterministic,
// vector order per type), so clustered shading reuses the same shadow maps.
inline uint32_t buildClusterLights(const LightSystem& lights, ClusterLight* out, uint32_t maxLights) {
    uint32_t n = 0;
    int pointShadow = 0;
    uint32_t pi = 0;
    for (auto& l : lights.pointLights) {
        if (n >= maxLights) break;
        ClusterLight c{};
        c.posRange = glm::vec4(l.position, l.range);
        // shadow slots must match LightSystem::fillGlobalUBO, which only
        // assigns them to the first MAX_POINT_LIGHTS entries
        float shadowIdx = -1.0f;
        if (l.castShadow && pi < (uint32_t)LightLimits::MAX_POINT_LIGHTS
            && pointShadow < LightLimits::MAX_SHADOW_POINT)
            shadowIdx = (float)pointShadow++;
        c.color = glm::vec4(l.color * l.intensity, shadowIdx);
        c.dirInner = glm::vec4(0.0f);
        c.outerType = glm::vec4(0.0f); // type 0 = point
        out[n++] = c;
        pi++;
    }
    int spotShadow = 0;
    uint32_t si = 0;
    for (auto& l : lights.spotLights) {
        if (n >= maxLights) break;
        ClusterLight c{};
        c.posRange = glm::vec4(l.position, l.range);
        float shadowIdx = -1.0f;
        if (l.castShadow && si < (uint32_t)LightLimits::MAX_SPOT_LIGHTS
            && spotShadow < LightLimits::MAX_SHADOW_SPOT)
            shadowIdx = (float)spotShadow++;
        c.color = glm::vec4(l.color * l.intensity, shadowIdx);
        c.dirInner = glm::vec4(glm::normalize(l.direction),
            std::cos(glm::radians(l.innerAngleDeg)));
        c.outerType = glm::vec4(std::cos(glm::radians(l.outerAngleDeg)), 1.0f, 0.0f, 0.0f);
        out[n++] = c;
        si++;
    }
    return n;
}
