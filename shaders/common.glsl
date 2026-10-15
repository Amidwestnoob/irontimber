#version 450
#extension GL_EXT_control_flow_attributes : enable
// Shared declarations: one activation/state arena, one small-params buffer, and the
// two halves of a repacked ternary weight (2-bit code words + per-128 f32 scales).
layout(std430, set = 0, binding = 0) buffer Arena { float a[]; };
layout(std430, set = 0, binding = 0) buffer ArenaV { vec4 a4[]; };
layout(std430, set = 0, binding = 1) readonly buffer Params { float prm[]; };
layout(std430, set = 0, binding = 2) readonly buffer Words { uvec4 wq4[]; };
layout(std430, set = 0, binding = 3) readonly buffer Scales { float wsc[]; };
layout(push_constant) uniform PC { uint u[32]; } pc;
#define NONE 0xffffffffu
float pf(uint i) { return uintBitsToFloat(pc.u[i]); }
float silu(float x) { return x / (1.0 + exp(-x)); }
float sigm(float x) { return 1.0 / (1.0 + exp(-x)); }
