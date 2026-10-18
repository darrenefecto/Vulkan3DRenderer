#version 450
// SSGI foundation: stochastic cosine-weighted diffuse ray, one sample per pixel.
// Output is indirect incident radiance estimate (RGB) and hit confidence (A).
// Not yet connected to the RenderGraph. Sample pre-SSGI scene HDR only.
layout(std140,set=0,binding=0) uniform GlobalView { mat4 view; mat4 proj; } g;
layout(set=1,binding=0) uniform sampler2D sourceHDR;
layout(set=1,binding=1) uniform sampler2D gNormalRough;
layout(set=1,binding=3) uniform sampler2D gDepth;
layout(push_constant) uniform SSGISettings {
    mat4 invProj;
    vec4 trace; // x max distance, y thickness, z stride, w frame index
} pc;
layout(location=0) in vec2 uv;
layout(location=0) out vec4 indirectRadiance;
bool finite1(float x) { return !isnan(x) && !isinf(x); }
bool finite3(vec3 v) { return !any(isnan(v)) && !any(isinf(v)); }
vec3 reconstructView(vec2 q,float d) {
    vec4 h=pc.invProj*vec4(q*2.0-1.0,d,1.0);
    return h.xyz/h.w;
}
vec2 projectUV(vec3 p) {
    vec4 h=g.proj*vec4(p,1.0);
    if (!finite1(h.w) || h.w<=1e-5) return vec2(-1.0);
    return h.xy/h.w*0.5+0.5;
}
float hash(vec3 p) { return fract(sin(dot(p,vec3(127.1,311.7,74.7)))*43758.5453); }
void main() {
    indirectRadiance=vec4(0.0);
    float d=texture(gDepth,uv).r;
    if (!finite1(d) || d<=0.0 || d>=0.999999) return;
    vec3 normalWS=texture(gNormalRough,uv).xyz;
    if (!finite3(normalWS) || dot(normalWS,normalWS)<1e-8) return;
    vec3 p=reconstructView(uv,d);
    if (!finite3(p)) return;
    vec3 n=normalize(mat3(g.view)*normalize(normalWS));
    if (dot(n,normalize(-p))<=0.02) return;
    vec3 tangent=normalize(cross(abs(n.z)<0.9?vec3(0,0,1):vec3(0,1,0),n));
    vec3 bitangent=cross(n,tangent);
    float a=6.2831853*hash(vec3(gl_FragCoord.xy,pc.trace.w));
    float r=sqrt(clamp(hash(vec3(gl_FragCoord.yx+19.0,pc.trace.w+31.0)),0.001,0.999));
    vec3 ray=normalize(tangent*cos(a)*r+bitangent*sin(a)*r+n*sqrt(1.0-r*r));
    float thickness=max(pc.trace.y,0.01);
    float stride=max(pc.trace.z,0.02);
    float maxDistance=max(pc.trace.x,stride);
    vec3 origin=p+n*max(thickness,0.025);
    float previousDelta=-1.0;
    for(int i=1;i<=96;++i) {
        float t=float(i)*stride;
        if(t>maxDistance) break;
        vec3 probe=origin+ray*t;
        vec2 q=projectUV(probe);
        if(any(lessThanEqual(q,vec2(0.001))) || any(greaterThanEqual(q,vec2(0.999)))) break;
        float sd=texture(gDepth,q).r;
        if(!finite1(sd) || sd<=0.0 || sd>=0.999999) { previousDelta=-1.0; continue; }
        vec3 surface=reconstructView(q,sd);
        if(!finite3(surface)) break;
        float delta=surface.z-probe.z;
        if(previousDelta<0.0 && delta>=0.0 && delta<thickness) {
            vec3 hitNormalWS=texture(gNormalRough,q).xyz;
            if(!finite3(hitNormalWS) || dot(hitNormalWS,hitNormalWS)<1e-8) return;
            vec3 hitNormal=normalize(mat3(g.view)*normalize(hitNormalWS));
            float facing=max(dot(hitNormal,-ray),0.0);
            if(facing<=0.02) return;
            vec3 incoming=texture(sourceHDR,q).rgb;
            if(!finite3(incoming)) return;
            float edge=clamp(min(min(q.x,q.y),min(1.0-q.x,1.0-q.y))*12.0,0.0,1.0);
            // Smoothly suppress unstable hits near the ray-range boundary and
            // depth-thickness threshold; grazing hit normals are less reliable.
            float fade=1.0-smoothstep(0.65,1.0,t/maxDistance);
            float thicknessConfidence=1.0-smoothstep(0.35,1.0,delta/thickness);
            float facingConfidence=smoothstep(0.02,0.35,facing);
            float hitConfidence=clamp(edge*fade*thicknessConfidence*facingConfidence,0.0,1.0);
            // Diffuse BRDF (albedo / PI) must be applied in a later lighting pass.
            indirectRadiance=vec4(max(incoming,vec3(0.0))*facing,hitConfidence);
            return;
        }
        previousDelta=delta;
    }
}
