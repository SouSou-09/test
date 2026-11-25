use glam::Mat4;

use crate::{camera::Camera, scene::Scene};

pub const MAX_POINT_LIGHTS: usize = 8;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FrameUniform {
    pub view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub inv_proj: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub light_view_proj: [[f32; 4]; 4],
    pub camera_position_time: [f32; 4],
    pub sun_direction_intensity: [f32; 4],
    pub sun_color: [f32; 4],
    pub screen_params: [f32; 4],
    pub point_light_positions: [[f32; 4]; MAX_POINT_LIGHTS],
    pub point_light_colors: [[f32; 4]; MAX_POINT_LIGHTS],
}

impl FrameUniform {
    pub fn new(
        camera: &Camera,
        scene: &Scene,
        width: u32,
        height: u32,
        light_view_proj: Mat4,
    ) -> Self {
        let view = camera.view_matrix();
        let proj = camera.projection_matrix();
        let view_proj = proj * view;
        let mut point_light_positions = [[0.0; 4]; MAX_POINT_LIGHTS];
        let mut point_light_colors = [[0.0; 4]; MAX_POINT_LIGHTS];
        for (index, light) in scene.point_lights.iter().take(MAX_POINT_LIGHTS).enumerate() {
            point_light_positions[index] = [
                light.position.x,
                light.position.y,
                light.position.z,
                light.radius,
            ];
            point_light_colors[index] =
                [light.color.x, light.color.y, light.color.z, light.intensity];
        }
        Self {
            view_proj: view_proj.to_cols_array_2d(),
            inv_view_proj: view_proj.inverse().to_cols_array_2d(),
            view: view.to_cols_array_2d(),
            inv_proj: proj.inverse().to_cols_array_2d(),
            proj: proj.to_cols_array_2d(),
            light_view_proj: light_view_proj.to_cols_array_2d(),
            camera_position_time: [
                camera.position.x,
                camera.position.y,
                camera.position.z,
                scene.time,
            ],
            sun_direction_intensity: [
                scene.sun_direction.x,
                scene.sun_direction.y,
                scene.sun_direction.z,
                scene.sun_intensity,
            ],
            sun_color: [scene.sun_color.x, scene.sun_color.y, scene.sun_color.z, 1.0],
            screen_params: [
                width as f32,
                height as f32,
                scene.point_lights.len().min(MAX_POINT_LIGHTS) as f32,
                0.0,
            ],
            point_light_positions,
            point_light_colors,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceUniform {
    pub model: [[f32; 4]; 4],
    pub normal_model: [[f32; 4]; 4],
    pub base_color_metallic: [f32; 4],
    pub roughness_emissive: [f32; 4],
    pub material_params: [f32; 4],
}

impl InstanceUniform {
    pub fn new(model: Mat4, material: crate::scene::Material) -> Self {
        let normal_model = model.inverse().transpose();
        Self {
            model: model.to_cols_array_2d(),
            normal_model: normal_model.to_cols_array_2d(),
            base_color_metallic: [
                material.base_color.x,
                material.base_color.y,
                material.base_color.z,
                material.metallic,
            ],
            roughness_emissive: [
                material.roughness,
                material.emissive.x,
                material.emissive.y,
                material.emissive.z,
            ],
            material_params: [
                material.detail_kind,
                material.detail_scale,
                material.texture_flags,
                0.0,
            ],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PostUniform {
    pub direction_threshold: [f32; 4],
    pub texel_size: [f32; 4],
}
