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

struct PostUniform {
    direction_threshold: vec4<f32>,
    texel_size: vec4<f32>,
};

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var input_sampler: sampler;
@group(0) @binding(2) var<uniform> post: PostUniform;

@group(0) @binding(0) var hdr_texture: texture_2d<f32>;
@group(0) @binding(1) var ao_texture: texture_2d<f32>;
@group(0) @binding(2) var bloom_texture: texture_2d<f32>;
@group(0) @binding(3) var composite_sampler: sampler;
@group(0) @binding(4) var<uniform> frame: FrameUniform;
@group(0) @binding(5) var ambient_texture: texture_2d<f32>;

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

fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

@fragment
fn fs_bright(input: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(input_texture, input_sampler, input.uv).rgb;
    let threshold = post.direction_threshold.z;
    let knee = post.direction_threshold.w;
    let brightness = luminance(color);
    let soft = clamp(brightness - threshold + knee, 0.0, 2.0 * knee);
    let soft_weight = soft * soft / max(4.0 * knee, 0.0001);
    let weight = max(soft_weight, brightness - threshold) / max(brightness, 0.0001);
    return vec4<f32>(color * max(weight, 0.0), 1.0);
}

@fragment
fn fs_blur(input: VertexOutput) -> @location(0) vec4<f32> {
    let direction = post.direction_threshold.xy;
    let texel = post.texel_size.xy;
    var color = textureSample(input_texture, input_sampler, input.uv).rgb * 0.227027;
    let offsets = array<f32, 4>(1.3846154, 3.2307692, 5.0769231, 7.0);
    let weights = array<f32, 4>(0.3162162, 0.0702703, 0.0285714, 0.0075);
    for (var index = 0; index < 4; index = index + 1) {
        let offset = direction * texel * offsets[index];
        color += textureSample(input_texture, input_sampler, input.uv + offset).rgb * weights[index];
        color += textureSample(input_texture, input_sampler, input.uv - offset).rgb * weights[index];
    }
    return vec4<f32>(color, 1.0);
}

fn hash12(seed: vec2<f32>) -> f32 {
    let value = sin(dot(seed, vec2<f32>(12.9898, 78.233))) * 43758.5453;
    return fract(value);
}

fn sky_color(direction: vec3<f32>, frame_data: FrameUniform) -> vec3<f32> {
    let height = clamp(direction.y * 0.5 + 0.5, 0.0, 1.0);
    let horizon = vec3<f32>(0.34, 0.24, 0.18);
    let zenith = vec3<f32>(0.045, 0.16, 0.38);
    var sky = mix(horizon, zenith, pow(height, 0.62));
    let sun_amount = max(dot(direction, -frame_data.sun_direction_intensity.xyz), 0.0);
    sky += frame_data.sun_color.rgb * pow(sun_amount, 600.0) * 28.0;
    sky += frame_data.sun_color.rgb * pow(sun_amount, 8.0) * 0.25;
    return sky;
}

fn aces_filmic(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((color * (a * color + vec3<f32>(b))) / (color * (c * color + vec3<f32>(d)) + vec3<f32>(e)), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_composite(input: VertexOutput) -> @location(0) vec4<f32> {
    let hdr = textureSample(hdr_texture, composite_sampler, input.uv);
    let ao = textureSample(ao_texture, composite_sampler, input.uv).r;
    let bloom = textureSample(bloom_texture, composite_sampler, input.uv).rgb;

    let clip = vec4<f32>(input.uv.x * 2.0 - 1.0, 1.0 - input.uv.y * 2.0, 1.0, 1.0);
    let world = frame.inv_view_proj * clip;
    let ray = normalize(world.xyz / max(world.w, 0.0001) - frame.camera_position_time.xyz);
    let sky = sky_color(ray, frame);

    let ambient = textureSample(ambient_texture, composite_sampler, input.uv).rgb;
    let direct_lighting = max(hdr.rgb - ambient, vec3<f32>(0.0));
    let shaded = direct_lighting + ambient * ao;
    var color = mix(sky, shaded, hdr.a);
    color += bloom * 0.48;
    color *= 1.05;

    let centered = input.uv - vec2<f32>(0.5);
    let vignette = 1.0 - dot(centered, centered) * 0.28;
    color *= vignette;
    color += vec3<f32>((hash12(input.position.xy + frame.camera_position_time.w) - 0.5) * 0.006);

    return vec4<f32>(aces_filmic(color), 1.0);
}
