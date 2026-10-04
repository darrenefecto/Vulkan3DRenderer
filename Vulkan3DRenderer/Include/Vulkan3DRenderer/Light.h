#pragma once
#include "VulkanContext.h"
#include "Camera.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>

// ---------------------------------------------------------------------------
// Lighting limits (must match the #defines in mesh.frag / shadow shaders)
// ---------------------------------------------------------------------------
namespace LightLimits {
    constexpr int MAX_POINT_LIGHTS = 8;   // total point lights
    constexpr int MAX_SPOT_LIGHTS  = 8;   // total spot lights
    constexpr int MAX_AREA_LIGHTS  = 4;   // total rect area lights
    constexpr int MAX_SHADOW_POINT = 4;   // point lights that cast shadows
    constexpr int MAX_SHADOW_SPOT  = 4;   // spot  lights that cast shadows
    constexpr int MAX_SHADOW_AREA  = 4;   // area  lights that cast shadows
    constexpr int NUM_CASCADES     = 4;   // CSM cascade count for the sun

    constexpr uint32_t CSM_RESOLUTION   = 2048;
    constexpr uint32_t SPOT_RESOLUTION  = 1024;
    constexpr uint32_t AREA_RESOLUTION  = 1024;
    constexpr uint32_t POINT_RESOLUTION = 1024; // per cube face
}

// ---------------------------------------------------------------------------
// CPU-side light descriptions (the "light system")
// ---------------------------------------------------------------------------
struct SunLight {
    bool      enabled = true;
    glm::vec3 direction = glm::normalize(glm::vec3(-0.6f, -1.0f, -0.35f));
    glm::vec3 color{ 1.0f, 0.98f, 0.92f };
    float     intensity = 3.0f;
    bool      castShadow = true;
    float     shadowDistance = 80.0f;   // world units covered by the cascade chain
    float     splitLambda = 0.65f;      // 0 = uniform splits, 1 = logarithmic splits
    float     depthBias = 0.0006f;
    float     slopeBias = 0.0025f;
};

struct PointLight {
    glm::vec3 position{ 0.0f, 2.0f, 0.0f };
    glm::vec3 color{ 1.0f };
    float     intensity = 20.0f;
    float     range = 15.0f;
    bool      castShadow = true;
};

struct SpotLight {
    glm::vec3 position{ 0.0f, 4.0f, 0.0f };
    glm::vec3 direction{ 0.0f, -1.0f, 0.0f };
    glm::vec3 color{ 1.0f };
    float     intensity = 40.0f;
    float     range = 25.0f;
    float     innerAngleDeg = 18.0f;    // full-intensity cone
    float     outerAngleDeg = 26.0f;    // falloff cone
    bool      castShadow = true;
};

// Rectangular area light: center position + orthonormal right/up axes.
// right/up are scaled by width/height inside the shader UBO (half extents).
struct AreaLight {
    glm::vec3 position{ 0.0f, 3.0f, 0.0f };
    glm::vec3 right{ 1.0f, 0.0f, 0.0f };  // will be normalized
    glm::vec3 up{ 0.0f, 0.0f, 1.0f };
    glm::vec3 color{ 1.0f };
    float     width = 2.0f;
    float     height = 1.0f;
    float     intensity = 8.0f;
    float     range = 20.0f;
    bool      castShadow = true;
    bool      twoSided = true;
};

// ---------------------------------------------------------------------------
// GPU global UBO (std140). MUST match the `Global` block in the GLSL shaders.
// ---------------------------------------------------------------------------
struct GlobalUBO {
    glm::mat4 view;
    glm::mat4 proj;
    glm::vec4 camPos;        // xyz = position
    glm::vec4 ambient;       // rgb = flat ambient color, w = IBL intensity
    glm::vec4 skyParams;     // x = env cubemap max LOD

    // --- sun / CSM ---
    glm::vec4 sunDir;        // xyz = direction (towards light), w = enabled
    glm::vec4 sunColor;      // rgb = color*intensity, w = castShadow
    glm::mat4 sunViewProj[LightLimits::NUM_CASCADES];
    glm::vec4 cascadeSplits; // view-space depth at the end of each cascade

    glm::ivec4 counts;       // x = points, y = spots, z = areas, w = active cascades

    // --- point lights ---
    glm::vec4 pointPos[LightLimits::MAX_POINT_LIGHTS];   // xyz = pos, w = range
    glm::vec4 pointColor[LightLimits::MAX_POINT_LIGHTS]; // rgb = color*intensity, w = shadow index (-1 = none)

    // --- spot lights ---
    glm::vec4 spotPos[LightLimits::MAX_SPOT_LIGHTS];     // xyz = pos, w = range
    glm::vec4 spotDir[LightLimits::MAX_SPOT_LIGHTS];     // xyz = dir, w = shadow index (-1 = none)
    glm::vec4 spotColor[LightLimits::MAX_SPOT_LIGHTS];   // rgb = color*intensity, w = innerCos
    glm::vec4 spotParams[LightLimits::MAX_SPOT_LIGHTS];  // x = outerCos

    // --- area lights ---
    glm::vec4 areaPos[LightLimits::MAX_AREA_LIGHTS];     // xyz = center, w = shadow index (-1 = none)
    glm::vec4 areaRight[LightLimits::MAX_AREA_LIGHTS];   // xyz = right * halfWidth
    glm::vec4 areaUp[LightLimits::MAX_AREA_LIGHTS];      // xyz = up * halfHeight
    glm::vec4 areaColor[LightLimits::MAX_AREA_LIGHTS];   // rgb = color*intensity, w = range

    // --- shadow-caster matrices ---
    glm::mat4 spotViewProj[LightLimits::MAX_SHADOW_SPOT];
    glm::mat4 areaViewProj[LightLimits::MAX_SHADOW_AREA];
    glm::mat4 pointViewProj[LightLimits::MAX_SHADOW_POINT * 6]; // 6 cube faces per light
    glm::vec4 pointShadowPosFar[LightLimits::MAX_SHADOW_POINT]; // xyz = pos, w = far plane
};

// ---------------------------------------------------------------------------
// Light system: owns all lights, assigns shadow-map slots and fills the UBO.
// ---------------------------------------------------------------------------
class LightSystem {
public:
    SunLight sun;
    std::vector<PointLight> pointLights;
    std::vector<SpotLight>  spotLights;
    std::vector<AreaLight>  areaLights;

    glm::vec3 ambientColor{ 0.03f, 0.04f, 0.06f }; // fallback flat ambient
    float     iblIntensity = 1.0f;                 // skybox-based ambient

    // Vulkan-depth [0,1] ortho (unlike glm::ortho which targets GL [-1,1]).
    static glm::mat4 orthoVK(float l, float r, float b, float t, float n, float f) {
        glm::mat4 m(1.0f);
        m[0][0] =  2.0f / (r - l);
        m[1][1] =  2.0f / (t - b);
        m[2][2] = -1.0f / (f - n);
        m[3][0] = -(r + l) / (r - l);
        m[3][1] = -(t + b) / (t - b);
        m[3][2] = -n / (f - n);
        return m;
    }

    static glm::vec3 safeUp(const glm::vec3& dir) {
        return (std::abs(dir.y) > 0.99f) ? glm::vec3(1, 0, 0) : glm::vec3(0, 1, 0);
    }

    // Practical-split CSM matrices. Returns cascade count; splits written to outSplits.
    int computeCascades(const Camera& cam, float aspect,
                        std::array<glm::mat4, LightLimits::NUM_CASCADES>& outVP,
                        glm::vec4& outSplits) const {
        using namespace LightLimits;
        const float nearClip = cam.nearClip;
        const float farClip = std::min(cam.farClip, sun.shadowDistance);
        const glm::vec3 dir = glm::normalize(sun.direction);
        const glm::vec3 up = safeUp(dir);

        float splits[NUM_CASCADES + 1];
        splits[0] = nearClip;
        for (int i = 1; i <= NUM_CASCADES; i++) {
            float p = (float)i / NUM_CASCADES;
            float logS = nearClip * std::pow(farClip / nearClip, p);
            float uniS = nearClip + (farClip - nearClip) * p;
            splits[i] = sun.splitLambda * logS + (1.0f - sun.splitLambda) * uniS;
        }

        glm::mat4 camView = cam.view();

        for (int i = 0; i < NUM_CASCADES; i++) {
            // frustum slice corners in world space
            glm::mat4 sliceProj = glm::perspective(glm::radians(cam.fov), aspect,
                                                   splits[i], splits[i + 1]);
            sliceProj[1][1] *= -1.0f; // match Camera::proj (Vulkan clip space)
            glm::mat4 invVP = glm::inverse(sliceProj * camView);

            glm::vec3 corners[8];
            glm::vec3 center(0.0f);
            int c = 0;
            for (int x = 0; x < 2; x++)
                for (int y = 0; y < 2; y++)
                    for (int z = 0; z < 2; z++) {
                        glm::vec4 ndc(x ? 1.f : -1.f, y ? 1.f : -1.f, z ? 1.f : 0.f, 1.f);
                        glm::vec4 w = invVP * ndc;
                        corners[c] = glm::vec3(w) / w.w;
                        center += corners[c++];
                    }
            center /= 8.0f;

            // tight sphere fit around the slice
            float radius = 0.0f;
            for (auto& k : corners) radius = std::max(radius, glm::length(k - center));
            const float zMargin = radius; // catch shadow casters behind the slice

            // texel snapping for stable edges while the camera moves
            float texel = (2.0f * radius) / (float)CSM_RESOLUTION;
            glm::mat4 view = glm::lookAt(center - dir * (radius + zMargin), center, up);
            glm::vec3 lsCenter = glm::vec3(view * glm::vec4(center, 1.0f));
            lsCenter.x = std::floor(lsCenter.x / texel) * texel;
            lsCenter.y = std::floor(lsCenter.y / texel) * texel;
            center = glm::vec3(glm::inverse(view) * glm::vec4(lsCenter, 1.0f));

            view = glm::lookAt(center - dir * (radius + zMargin), center, up);
            outVP[i] = orthoVK(-radius, radius, -radius, radius, 0.0f, 2.0f * (radius + zMargin)) * view;
            outSplits[i] = splits[i + 1];
        }
        return NUM_CASCADES;
    }

    static glm::mat4 spotMatrix(const SpotLight& s) {
        float fov = glm::radians(s.outerAngleDeg) * 2.0f * 1.05f; // small margin
        fov = std::min(fov, glm::radians(170.0f));
        glm::vec3 d = glm::normalize(s.direction);
        glm::mat4 proj = glm::perspective(fov, 1.0f, 0.05f, s.range);
        return proj * glm::lookAt(s.position, s.position + d, safeUp(d));
    }

    // Shadow view from the rect center, looking along the rect normal.
    static glm::mat4 areaMatrix(const AreaLight& a) {
        glm::vec3 right = glm::normalize(a.right);
        glm::vec3 up = glm::normalize(a.up);
        glm::vec3 n = glm::normalize(glm::cross(right, up));
        float diag = 0.5f * std::sqrt(a.width * a.width + a.height * a.height);
        float fov = 2.0f * std::atan(std::max(diag, 0.5f) / 0.5f); // covers the rect
        fov = glm::clamp(fov, glm::radians(60.0f), glm::radians(150.0f));
        glm::mat4 proj = glm::perspective(fov, 1.0f, 0.05f, a.range);
        return proj * glm::lookAt(a.position, a.position + n, up);
    }

    static glm::mat4 pointFaceMatrix(const glm::vec3& pos, int face, float range) {
        static const glm::vec3 dirs[6] = {
            { 1, 0, 0}, {-1, 0, 0}, { 0, 1, 0}, { 0,-1, 0}, { 0, 0, 1}, { 0, 0,-1} };
        static const glm::vec3 ups[6] = {
            { 0,-1, 0}, { 0,-1, 0}, { 0, 0, 1}, { 0, 0,-1}, { 0,-1, 0}, { 0,-1, 0} };
        glm::mat4 proj = glm::perspective(glm::radians(90.0f), 1.0f, 0.05f, range);
        return proj * glm::lookAt(pos, pos + dirs[face], ups[face]);
    }

    // Fills light/shadow fields of the UBO. view/proj/camPos filled by the caller.
    // Returns nothing; shadow-slot assignment is deterministic: first N lights with
    // castShadow=true of each type get shadow indices 0..N-1 (in vector order).
    void fillGlobalUBO(GlobalUBO& ubo, const Camera& cam, float aspect, float envMaxLod) const {
        using namespace LightLimits;
        ubo.ambient = glm::vec4(ambientColor, iblIntensity);
        ubo.skyParams = glm::vec4(envMaxLod, 0.0f, 0.0f, 0.0f);

        // --- sun ---
        std::array<glm::mat4, NUM_CASCADES> csm{};
        glm::vec4 splits(0.0f);
        int numCascades = 0;
        if (sun.enabled && sun.castShadow)
            numCascades = computeCascades(cam, aspect, csm, splits);
        ubo.sunDir = glm::vec4(sun.enabled ? -glm::normalize(sun.direction) : glm::vec3(0.0f),
                               sun.enabled ? 1.0f : 0.0f); // direction towards the light
        ubo.sunColor = glm::vec4(sun.color * sun.intensity,
                                 (sun.enabled && sun.castShadow) ? 1.0f : 0.0f);
        for (int i = 0; i < NUM_CASCADES; i++) ubo.sunViewProj[i] = csm[i];
        ubo.cascadeSplits = splits;

        // --- point lights ---
        int pc = std::min((int)pointLights.size(), MAX_POINT_LIGHTS);
        int ps = 0;
        for (int i = 0; i < pc; i++) {
            const PointLight& l = pointLights[i];
            ubo.pointPos[i] = glm::vec4(l.position, l.range);
            int shadowIdx = -1;
            if (l.castShadow && ps < MAX_SHADOW_POINT) {
                shadowIdx = ps++;
                ubo.pointShadowPosFar[shadowIdx] = glm::vec4(l.position, l.range);
                for (int f = 0; f < 6; f++)
                    ubo.pointViewProj[shadowIdx * 6 + f] = pointFaceMatrix(l.position, f, l.range);
            }
            ubo.pointColor[i] = glm::vec4(l.color * l.intensity, (float)shadowIdx);
        }
        for (int i = pc; i < MAX_POINT_LIGHTS; i++) {
            ubo.pointPos[i] = glm::vec4(0.0f);
            ubo.pointColor[i] = glm::vec4(0.0f, 0.0f, 0.0f, -1.0f);
        }

        // --- spot lights ---
        int sc = std::min((int)spotLights.size(), MAX_SPOT_LIGHTS);
        int ss = 0;
        for (int i = 0; i < sc; i++) {
            const SpotLight& l = spotLights[i];
            int shadowIdx = -1;
            if (l.castShadow && ss < MAX_SHADOW_SPOT) {
                shadowIdx = ss++;
                ubo.spotViewProj[shadowIdx] = spotMatrix(l);
            }
            ubo.spotPos[i] = glm::vec4(l.position, l.range);
            ubo.spotDir[i] = glm::vec4(glm::normalize(l.direction), (float)shadowIdx);
            ubo.spotColor[i] = glm::vec4(l.color * l.intensity,
                                         std::cos(glm::radians(l.innerAngleDeg)));
            ubo.spotParams[i] = glm::vec4(std::cos(glm::radians(l.outerAngleDeg)), 0, 0, 0);
        }
        for (int i = sc; i < MAX_SPOT_LIGHTS; i++) {
            ubo.spotPos[i] = ubo.spotDir[i] = ubo.spotColor[i] = ubo.spotParams[i] = glm::vec4(0.0f);
        }

        // --- area lights ---
        int ac = std::min((int)areaLights.size(), MAX_AREA_LIGHTS);
        int as = 0;
        for (int i = 0; i < ac; i++) {
            const AreaLight& l = areaLights[i];
            int shadowIdx = -1;
            if (l.castShadow && as < MAX_SHADOW_AREA) {
                shadowIdx = as++;
                ubo.areaViewProj[shadowIdx] = areaMatrix(l);
            }
            ubo.areaPos[i] = glm::vec4(l.position, (float)shadowIdx);
            ubo.areaRight[i] = glm::vec4(glm::normalize(l.right) * (l.width * 0.5f), 0.0f);
            ubo.areaUp[i] = glm::vec4(glm::normalize(l.up) * (l.height * 0.5f),
                                      l.twoSided ? 1.0f : 0.0f);
            ubo.areaColor[i] = glm::vec4(l.color * l.intensity, l.range);
        }
        for (int i = ac; i < MAX_AREA_LIGHTS; i++) {
            ubo.areaPos[i] = glm::vec4(0.0f, 0.0f, 0.0f, -1.0f);
            ubo.areaRight[i] = ubo.areaUp[i] = ubo.areaColor[i] = glm::vec4(0.0f);
        }

        ubo.counts = glm::ivec4(pc, sc, ac, numCascades);
    }
};
