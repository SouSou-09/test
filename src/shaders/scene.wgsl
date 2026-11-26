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
@group(0) @binding(1) var shadow_map: texture_2d<u32>;

@group(1) @binding(0) var base_color_texture: texture_2d<f32>;
@group(1) @binding(1) var normal_texture: texture_2d<f32>;
@group(1) @binding(2) var metallic_roughness_texture: texture_2d<f32>;
@group(1) @binding(3) var emissive_texture: texture_2d<f32>;
@group(1) @binding(4) var occlusion_texture: texture_2d<f32>;
@group(1) @binding(5) var material_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec4<f32>,
    @location(15) uv2: vec2<f32>,
    @location(4) model0: vec4<f32>,
    @location(5) model1: vec4<f32>,
    @location(6) model2: vec4<f32>,
    @location(7) model3: vec4<f32>,
    @location(8) normal0: vec4<f32>,
    @location(9) normal1: vec4<f32>,
    @location(10) normal2: vec4<f32>,
    @location(11) normal3: vec4<f32>,
    @location(12) base_color_metallic: vec4<f32>,
    @location(13) roughness_emissive: vec4<f32>,
    @location(14) material_params: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) view_normal: vec3<f32>,
    @location(3) view_depth: f32,
    @location(4) uv: vec2<f32>,
    @location(5) base_color: vec3<f32>,
    @location(6) metallic_roughness: vec2<f32>,
    @location(7) emissive: vec3<f32>,
    @location(8) material_params: vec4<f32>,
    @location(9) world_tangent: vec4<f32>,
    @location(10) uv2: vec2<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    let model = mat4x4<f32>(input.model0, input.model1, input.model2, input.model3);
    let normal_model = mat4x4<f32>(input.normal0, input.normal1, input.normal2, input.normal3);
    let world = model * vec4<f32>(input.position, 1.0);
    let world_normal = normalize((normal_model * vec4<f32>(input.normal, 0.0)).xyz);
    let view_normal = normalize((frame.view * vec4<f32>(world_normal, 0.0)).xyz);

    var output: VertexOutput;
    output.clip_position = frame.view_proj * world;
    output.world_position = world.xyz;
    output.world_normal = world_normal;
    output.view_normal = view_normal;
    output.view_depth = -(frame.view * world).z;
    output.uv = input.uv;
    output.uv2 = input.uv2;
    output.base_color = input.base_color_metallic.rgb;
    output.metallic_roughness = vec2<f32>(input.base_color_metallic.w, input.roughness_emissive.x);
    output.emissive = input.roughness_emissive.yzw;
    output.material_params = input.material_params;
    output.world_tangent = vec4<f32>(
        normalize((model * vec4<f32>(input.tangent.xyz, 0.0)).xyz),
        input.tangent.w,
    );
    return output;
}

fn saturate(value: f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn distribution_ggx(normal_dot_half: f32, roughness: f32) -> f32 {
    let alpha = max(roughness * roughness, 0.0004);
    let alpha2 = alpha * alpha;
    let denominator = normal_dot_half * normal_dot_half * (alpha2 - 1.0) + 1.0;
    return alpha2 / max(3.14159265 * denominator * denominator, 0.000001);
}

fn geometry_schlick_ggx(normal_dot_x: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = r * r / 8.0;
    return normal_dot_x / max(normal_dot_x * (1.0 - k) + k, 0.000001);
}

fn geometry_smith(normal_dot_view: f32, normal_dot_light: f32, roughness: f32) -> f32 {
    return geometry_schlick_ggx(normal_dot_view, roughness) * geometry_schlick_ggx(normal_dot_light, roughness);
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn cook_torrance(
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    light_dir: vec3<f32>,
    radiance: vec3<f32>,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
) -> vec3<f32> {
    let half_dir = normalize(view_dir + light_dir);
    let normal_dot_view = max(dot(normal, view_dir), 0.0001);
    let normal_dot_light = max(dot(normal, light_dir), 0.0);
    let normal_dot_half = max(dot(normal, half_dir), 0.0);
    let view_dot_half = max(dot(view_dir, half_dir), 0.0);

    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    let ndf = distribution_ggx(normal_dot_half, roughness);
    let geometry = geometry_smith(normal_dot_view, normal_dot_light, roughness);
    let fresnel = fresnel_schlick(view_dot_half, f0);
    let numerator = ndf * geometry * fresnel;
    let denominator = max(4.0 * normal_dot_view * normal_dot_light, 0.0001);
    let specular = numerator / denominator;

    let diffuse_fraction = (vec3<f32>(1.0) - fresnel) * (1.0 - metallic);
    let diffuse = diffuse_fraction * albedo / 3.14159265;
    return (diffuse + specular) * radiance * normal_dot_light;
}

const SHADOW_MAP_SIZE: f32 = 2048.0;
const SHADOW_ORTHO_WIDTH: f32 = 76.0;
const SHADOW_NEAR: f32 = 0.1;
const SHADOW_FAR: f32 = 110.0;
const PCSS_POISSON_COUNT: u32 = 16u;
const PCSS_BLOCKER_COUNT: u32 = 8u;
// 0xffffff00: the largest u32-scaled depth exactly representable as f32.
const SHADOW_DEPTH_SCALE: f32 = 4294967040.0;

var<private> PCSS_POISSON: array<vec2<f32>, 16> = array<vec2<f32>, 16>(
    vec2<f32>(-0.94201624, -0.39906216),
    vec2<f32>(0.94558609, -0.76890725),
    vec2<f32>(-0.09418410, -0.92938870),
    vec2<f32>(0.34495938, 0.29387760),
    vec2<f32>(-0.91588581, 0.45771432),
    vec2<f32>(-0.81544232, -0.87912464),
    vec2<f32>(-0.38277543, 0.27676845),
    vec2<f32>(0.97484398, 0.75648379),
    vec2<f32>(0.44323325, -0.97511554),
    vec2<f32>(0.53742981, -0.47373420),
    vec2<f32>(-0.26496911, -0.41893023),
    vec2<f32>(0.79197514, 0.19090188),
    vec2<f32>(-0.24188840, 0.99706507),
    vec2<f32>(-0.81409955, 0.91437590),
    vec2<f32>(0.19984126, 0.78641367),
    vec2<f32>(0.14383161, -0.14100790),
);

fn shadow_linear_depth(depth: f32) -> f32 {
    return SHADOW_NEAR + depth * (SHADOW_FAR - SHADOW_NEAR);
}

fn shadow_map_depth(uv: vec2<f32>) -> f32 {
    let texel = clamp(
        vec2<i32>(uv * SHADOW_MAP_SIZE),
        vec2<i32>(0, 0),
        vec2<i32>(i32(SHADOW_MAP_SIZE) - 1, i32(SHADOW_MAP_SIZE) - 1),
    );
    return f32(textureLoad(shadow_map, texel, 0).r) / SHADOW_DEPTH_SCALE;
}

fn average_blocker_depth(uv: vec2<f32>, receiver_depth: f32, bias: f32) -> f32 {
    var blocker_depth = 0.0;
    var blocker_count = 0.0;
    let search_radius = 13.0 / SHADOW_MAP_SIZE;
    for (var index = 0u; index < PCSS_BLOCKER_COUNT; index = index + 1u) {
        let depth = shadow_map_depth(uv + PCSS_POISSON[index] * search_radius);
        if depth + bias < receiver_depth {
            blocker_depth += depth;
            blocker_count += 1.0;
        }
    }
    if blocker_count < 0.5 {
        return -1.0;
    }
    return blocker_depth / blocker_count;
}

fn shadow_visibility(world_position: vec3<f32>, normal: vec3<f32>) -> f32 {
    let light_space = frame.light_view_proj * vec4<f32>(world_position, 1.0);
    let projected = light_space.xyz / max(light_space.w, 0.0001);
    let uv = vec2<f32>(projected.x * 0.5 + 0.5, projected.y * -0.5 + 0.5);
    if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || projected.z > 1.0 {
        return 1.0;
    }

    let normal_dot_light = max(dot(normal, -frame.sun_direction_intensity.xyz), 0.0);
    let bias = max(0.0024 * (1.0 - normal_dot_light), 0.00035);
    let receiver_world_depth = shadow_linear_depth(projected.z);
    let blocker_depth = average_blocker_depth(uv, projected.z, bias);
    if blocker_depth < 0.0 {
        return 1.0;
    }
    let blocker_world_depth = shadow_linear_depth(blocker_depth);
    let penumbra = max(receiver_world_depth - blocker_world_depth, 0.0)
        * max(frame.screen_params.w, 0.001);
    let filter_radius = clamp(
        penumbra / SHADOW_ORTHO_WIDTH * SHADOW_MAP_SIZE,
        1.25,
        13.0,
    );

    var visibility = 0.0;
    let texel_radius = filter_radius / SHADOW_MAP_SIZE;
    for (var index = 0u; index < PCSS_POISSON_COUNT; index = index + 1u) {
        let sample_uv = uv + PCSS_POISSON[index] * texel_radius;
        let map_depth = shadow_map_depth(sample_uv);
        visibility += select(0.0, 1.0, map_depth + bias >= projected.z);
    }
    return visibility / f32(PCSS_POISSON_COUNT);
}

fn sky_ambient(direction: vec3<f32>) -> vec3<f32> {
    let height = clamp(direction.y * 0.5 + 0.5, 0.0, 1.0);
    let horizon = vec3<f32>(0.34, 0.25, 0.19);
    let zenith = vec3<f32>(0.08, 0.20, 0.42);
    let sky = mix(horizon, zenith, pow(height, 0.7));
    let ground = vec3<f32>(0.12, 0.095, 0.07);
    return mix(ground, sky, smoothstep(-0.18, 0.16, direction.y));
}

fn environment_radiance(direction: vec3<f32>, roughness: f32, sun_visibility: f32) -> vec3<f32> {
    var radiance = sky_ambient(direction);
    let sun_amount = max(dot(direction, -frame.sun_direction_intensity.xyz), 0.0);
    let sharpness = pow(1.0 - roughness, 2.0);
    let forward_scatter = pow(sun_amount, mix(3.0, 18.0, sharpness)) * mix(0.035, 0.2, sharpness);
    radiance += frame.sun_color.rgb * forward_scatter * sun_visibility;
    return radiance;
}

fn hash21(seed: vec2<f32>) -> f32 {
    return fract(sin(dot(seed, vec2<f32>(127.1, 311.7))) * 43758.5453123);
}

fn value_noise(position: vec2<f32>) -> f32 {
    let cell = floor(position);
    let fraction = fract(position);
    let blend = fraction * fraction * (vec2<f32>(3.0) - fraction * 2.0);
    let a = hash21(cell);
    let b = hash21(cell + vec2<f32>(1.0, 0.0));
    let c = hash21(cell + vec2<f32>(0.0, 1.0));
    let d = hash21(cell + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, blend.x), mix(c, d, blend.x), blend.y);
}

fn boolean_mask(value: bool) -> f32 {
    return select(0.0, 1.0, value);
}

struct FragmentOutput {
    @location(0) hdr_color: vec4<f32>,
    @location(1) normal_material: vec4<f32>,
    @location(2) view_depth: vec4<f32>,
    @location(3) ambient_color: vec4<f32>,
};

@fragment
fn fs_main(input: VertexOutput) -> FragmentOutput {
    var normal = normalize(input.world_normal);
    let view_dir = normalize(frame.camera_position_time.xyz - input.world_position);
    var albedo = input.base_color;
    var metallic = clamp(input.metallic_roughness.x, 0.0, 1.0);
    var roughness = clamp(input.metallic_roughness.y, 0.035, 1.0);
    var emissive = input.emissive;
    let texture_flags = u32(input.material_params.z + 0.5);

    let base_uv = select(input.uv, input.uv2, (texture_flags & 32u) != 0u);
    let normal_uv = select(input.uv, input.uv2, (texture_flags & 64u) != 0u);
    let metallic_roughness_uv = select(input.uv, input.uv2, (texture_flags & 128u) != 0u);
    let emissive_uv = select(input.uv, input.uv2, (texture_flags & 256u) != 0u);
    let occlusion_uv = select(input.uv, input.uv2, (texture_flags & 512u) != 0u);
    if (texture_flags & 1u) != 0u {
        albedo *= textureSample(base_color_texture, material_sampler, base_uv).rgb;
    }
    if (texture_flags & 4u) != 0u {
        let metallic_roughness = textureSample(metallic_roughness_texture, material_sampler, metallic_roughness_uv);
        metallic = clamp(metallic * metallic_roughness.b, 0.0, 1.0);
        roughness = clamp(roughness * metallic_roughness.g, 0.035, 1.0);
    }
    if (texture_flags & 8u) != 0u {
        emissive *= textureSample(emissive_texture, material_sampler, emissive_uv).rgb;
    }
    var occlusion = 1.0;
    if (texture_flags & 16u) != 0u {
        occlusion = textureSample(occlusion_texture, material_sampler, occlusion_uv).r;
    }
    if (texture_flags & 2u) != 0u {
        let tangent_normal = textureSample(normal_texture, material_sampler, normal_uv).xyz * 2.0 - vec3<f32>(1.0);
        let tangent = normalize(input.world_tangent.xyz);
        let bitangent = cross(normal, tangent) * input.world_tangent.w;
        normal = normalize(
            tangent * tangent_normal.x + bitangent * tangent_normal.y + normal * tangent_normal.z,
        );
    }

    let detail_kind = input.material_params.x;
    if detail_kind > 0.5 && detail_kind < 1.5 {
        let concrete_noise = value_noise(input.world_position.xz * 1.7 + input.uv * 3.0);
        albedo *= 0.82 + concrete_noise * 0.28;
        roughness = clamp(roughness + concrete_noise * 0.08, 0.05, 1.0);

        if abs(normal.y) < 0.55 {
            let rows = max(2.0, floor(input.material_params.y * 2.2));
            let columns = 5.0;
            let cell_uv = vec2<f32>(fract(input.uv.x * columns), fract(input.uv.y * rows));
            let cell_id = floor(vec2<f32>(input.uv.x * columns, input.uv.y * rows));
            let in_window_x = boolean_mask(cell_uv.x > 0.2 && cell_uv.x < 0.78);
            let in_window_y = boolean_mask(cell_uv.y > 0.22 && cell_uv.y < 0.66);
            let window_mask = in_window_x * in_window_y;
            if window_mask > 0.0 {
                let lit = step(0.68, hash21(cell_id + floor(input.world_position.xz * 0.37)));
                let glass = vec3<f32>(0.025, 0.055, 0.075);
                albedo = mix(albedo, glass, window_mask * 0.86);
                let warm_cool = hash21(cell_id.yx + vec2<f32>(0.31, 0.73));
                let window_color = mix(vec3<f32>(1.0, 0.52, 0.2), vec3<f32>(0.38, 0.68, 1.0), warm_cool);
                emissive += window_color * lit * window_mask * 1.8;
            }
        }
    } else if detail_kind > 1.5 {
        let mineral = value_noise(input.world_position.xz * input.material_params.y * 0.08);
        let fine = value_noise(input.world_position.xz * input.material_params.y * 0.45);
        albedo *= 0.78 + mineral * 0.26 + fine * 0.08;
        roughness = clamp(roughness + fine * 0.1, 0.05, 1.0);
    }

    var color = vec3<f32>(0.0);
    let sun_visibility = shadow_visibility(input.world_position, normal);
    let sun_radiance = frame.sun_color.rgb * frame.sun_direction_intensity.w * sun_visibility;
    color += cook_torrance(
        normal,
        view_dir,
        -frame.sun_direction_intensity.xyz,
        sun_radiance,
        albedo,
        metallic,
        roughness,
    );

    let light_count = i32(frame.screen_params.z);
    for (var index = 0; index < 8; index = index + 1) {
        if index >= light_count {
            break;
        }
        let light_position = frame.point_light_positions[index].xyz;
        let radius = frame.point_light_positions[index].w;
        let to_light = light_position - input.world_position;
        let distance = max(length(to_light), 0.001);
        let light_dir = to_light / distance;
        let distance_falloff = 1.0 / max(distance * distance, 0.04);
        let radius_falloff = pow(clamp(1.0 - distance / max(radius, 0.01), 0.0, 1.0), 2.0);
        let radiance = frame.point_light_colors[index].rgb * frame.point_light_colors[index].w * distance_falloff * radius_falloff;
        color += cook_torrance(normal, view_dir, light_dir, radiance, albedo, metallic, roughness);
    }

    let reflection = reflect(-view_dir, normal);
    let rough_reflection = normalize(mix(reflection, normal, roughness * roughness));
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    let normal_dot_view = max(dot(normal, view_dir), 0.0);
    let env_fresnel = f0
        + (max(vec3<f32>(1.0 - roughness), f0) - f0) * pow(1.0 - normal_dot_view, 5.0);
    let diffuse_ambient = sky_ambient(normal) * albedo * (1.0 - metallic) * 0.42;
    let specular_ambient = environment_radiance(rough_reflection, roughness, sun_visibility)
        * env_fresnel * (1.0 - roughness * 0.62) * 0.86;
    let ambient = (diffuse_ambient + specular_ambient) * occlusion;
    color += ambient + emissive;

    var output: FragmentOutput;
    output.hdr_color = vec4<f32>(color, 1.0);
    output.ambient_color = vec4<f32>(ambient, 1.0);
    let shading_view_normal = normalize((frame.view * vec4<f32>(normal, 0.0)).xyz);
    output.normal_material = vec4<f32>(shading_view_normal * 0.5 + vec3<f32>(0.5), roughness);
    output.view_depth = vec4<f32>(input.view_depth, 0.0, 0.0, 1.0);
    return output;
}
