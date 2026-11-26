struct FrameUniform {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    proj: mat4x4<f32>,
    light_view_proj: mat4x4<f32>,
    camera_position_time: vec4<f32>,
    sun_direction_intensity: vec4<f32>,
    sun_color: vec4<f32>,
    screen_params: vec4<f32>,
    point_light_positions: array<vec4<f32>, 8>,
    point_light_colors: array<vec4<f32>, 8>,
};

@group(0) @binding(0) var<uniform> frame: FrameUniform;

// 0xffffff00: the largest u32-scaled depth exactly representable as f32.
const SHADOW_DEPTH_SCALE: f32 = 4294967040.0;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(4) model0: vec4<f32>,
    @location(5) model1: vec4<f32>,
    @location(6) model2: vec4<f32>,
    @location(7) model3: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) depth: f32,
};

@vertex
fn vs_shadow(input: VertexInput) -> VertexOutput {
    let model = mat4x4<f32>(input.model0, input.model1, input.model2, input.model3);
    let clip = frame.light_view_proj * model * vec4<f32>(input.position, 1.0);
    var output: VertexOutput;
    output.clip_position = clip;
    output.depth = clip.z / max(clip.w, 0.0001);
    return output;
}

@fragment
fn fs_shadow(input: VertexOutput) -> @location(0) vec4<u32> {
    let slope = max(abs(dpdx(input.depth)), abs(dpdy(input.depth)));
    let biased_depth = clamp(input.depth + slope * 1.75, 0.0, 1.0);
    let encoded_depth = u32(biased_depth * SHADOW_DEPTH_SCALE);
    return vec4<u32>(encoded_depth, 0u, 0u, 1u);
}
