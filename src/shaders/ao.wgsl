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
@group(0) @binding(1) var view_depth_texture: texture_2d<f32>;
@group(0) @binding(2) var normal_texture: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(3.0, 1.0),
    );
    let position = positions[vertex_index];
    var output: VertexOutput;
    output.position = vec4<f32>(position, 0.0, 1.0);
    output.uv = position * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
    return output;
}

fn hash12(seed: vec2<f32>) -> f32 {
    let value = sin(dot(seed, vec2<f32>(12.9898, 78.233))) * 43758.5453;
    return fract(value);
}

fn view_position_from_depth(uv: vec2<f32>, view_depth: f32) -> vec3<f32> {
    let clip = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 1.0, 1.0);
    let far_view = frame.inv_proj * clip;
    let ray = far_view.xyz / max(far_view.w, 0.000001);
    return ray * (view_depth / max(-ray.z, 0.000001));
}

@fragment
fn fs_ao(input: VertexOutput) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(input.position.xy);
    let depth = textureLoad(view_depth_texture, pixel, 0).r;
    let encoded_normal = textureLoad(normal_texture, pixel, 0).xyz;
    let normal = normalize(encoded_normal * 2.0 - vec3<f32>(1.0));
    if length(encoded_normal) < 0.05 {
        return vec4<f32>(1.0, 1.0, 1.0, 1.0);
    }

    let position = view_position_from_depth(input.uv, depth);
    let radius = 0.42;
    let bias = 0.025;
    let noise = hash12(input.position.xy + vec2<f32>(frame.camera_position_time.w, -frame.camera_position_time.w));

    var tangent = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), normal));
    if length(tangent) < 0.05 {
        tangent = vec3<f32>(1.0, 0.0, 0.0);
    }
    let bitangent = normalize(cross(normal, tangent));
    let tbn = mat3x3<f32>(tangent, bitangent, normal);

    var occlusion = 0.0;
    var weight = 0.0;
    for (var index = 0; index < 16; index = index + 1) {
        let fi = f32(index);
        let angle = (fi / 16.0 + noise) * 6.2831853;
        let spiral_radius = mix(0.18, 1.0, fract(fi * 0.6180339 + noise));
        let height = mix(0.12, 0.95, hash12(input.position.xy + vec2<f32>(fi, fi * 1.7)));
        let kernel = vec3<f32>(cos(angle) * spiral_radius, sin(angle) * spiral_radius, height);
        let sample_position = position + tbn * kernel * radius;

        let projected = frame.proj * vec4<f32>(sample_position, 1.0);
        let clip = projected.xyz / max(projected.w, 0.000001);
        let sample_uv = vec2<f32>(clip.x * 0.5 + 0.5, 0.5 - clip.y * 0.5);
        if sample_uv.x <= 0.0 || sample_uv.x >= 1.0 || sample_uv.y <= 0.0 || sample_uv.y >= 1.0 {
            continue;
        }
        let sample_pixel = vec2<i32>(sample_uv * frame.screen_params.xy);
        let sample_depth = textureLoad(view_depth_texture, sample_pixel, 0).r;
        let scene_position = view_position_from_depth(sample_uv, sample_depth);
        let range_weight = 1.0 - smoothstep(radius * 0.5, radius, abs(scene_position.z - position.z));
        if scene_position.z > sample_position.z + bias {
            occlusion += range_weight;
        }
        weight += range_weight;
    }

    let occlusion_factor = clamp(1.0 - occlusion / max(weight, 1.0) * 0.9, 0.0, 1.0);
    return vec4<f32>(occlusion_factor, occlusion_factor, occlusion_factor, 1.0);
}
