use glam::{EulerRot, Mat4, Quat, Vec3};
use rayon::prelude::*;

use crate::{
    assets::{self, AssetMesh, AssetTexture},
    camera::Frustum,
    mesh::{MeshData, MeshKind},
    physics::{PhysicsShape, PhysicsWorld, RigidBodyKind, RigidBodySpec},
};

pub const MAX_POINT_LIGHTS: usize = 8;
pub const MATERIAL_TEXTURE_BASE_COLOR: f32 = 1.0;
pub const MATERIAL_TEXTURE_NORMAL: f32 = 2.0;
pub const MATERIAL_TEXTURE_METALLIC_ROUGHNESS: f32 = 4.0;
pub const MATERIAL_TEXTURE_EMISSIVE: f32 = 8.0;
pub const MATERIAL_TEXTURE_OCCLUSION: f32 = 16.0;
pub const MATERIAL_UV2_BASE_COLOR: f32 = 32.0;
pub const MATERIAL_UV2_NORMAL: f32 = 64.0;
pub const MATERIAL_UV2_METALLIC_ROUGHNESS: f32 = 128.0;
pub const MATERIAL_UV2_EMISSIVE: f32 = 256.0;
pub const MATERIAL_UV2_OCCLUSION: f32 = 512.0;

#[derive(Clone, Copy, Debug)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Transform {
    pub fn new(translation: Vec3, rotation: Vec3, scale: Vec3) -> Self {
        Self {
            translation,
            rotation: Quat::from_euler(EulerRot::YXZ, rotation.y, rotation.x, rotation.z),
            scale,
        }
    }

    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Material {
    pub base_color: Vec3,
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: Vec3,
    /// 0: smooth, 1: procedural architectural concrete/windows, 2: mineral ground detail.
    pub detail_kind: f32,
    pub detail_scale: f32,
    /// Bitmask: 1 base color, 2 normal, 4 metallic-roughness textures.
    pub texture_flags: f32,
}

impl Material {
    pub fn new(base_color: Vec3, metallic: f32, roughness: f32) -> Self {
        Self {
            base_color,
            metallic,
            roughness,
            emissive: Vec3::ZERO,
            detail_kind: 0.0,
            detail_scale: 1.0,
            texture_flags: 0.0,
        }
    }

    pub fn emissive(base_color: Vec3, emissive: Vec3) -> Self {
        Self {
            base_color,
            metallic: 0.0,
            roughness: 0.45,
            emissive,
            detail_kind: 0.0,
            detail_scale: 1.0,
            texture_flags: 0.0,
        }
    }
}

fn material_from_asset(material: assets::AssetMaterial) -> Material {
    let mut result = Material::new(material.base_color, material.metallic, material.roughness);
    result.emissive = material.emissive;
    result.texture_flags = texture_flags(
        material.base_color_texture,
        material.base_color_tex_coord,
        MATERIAL_TEXTURE_BASE_COLOR,
        MATERIAL_UV2_BASE_COLOR,
    ) + texture_flags(
        material.normal_texture,
        material.normal_tex_coord,
        MATERIAL_TEXTURE_NORMAL,
        MATERIAL_UV2_NORMAL,
    ) + texture_flags(
        material.metallic_roughness_texture,
        material.metallic_roughness_tex_coord,
        MATERIAL_TEXTURE_METALLIC_ROUGHNESS,
        MATERIAL_UV2_METALLIC_ROUGHNESS,
    ) + texture_flags(
        material.emissive_texture,
        material.emissive_tex_coord,
        MATERIAL_TEXTURE_EMISSIVE,
        MATERIAL_UV2_EMISSIVE,
    ) + texture_flags(
        material.occlusion_texture,
        material.occlusion_tex_coord,
        MATERIAL_TEXTURE_OCCLUSION,
        MATERIAL_UV2_OCCLUSION,
    );
    result
}

fn texture_flags(texture: Option<usize>, tex_coord: u32, texture_bit: f32, uv2_bit: f32) -> f32 {
    if texture.is_none() {
        return 0.0;
    }
    texture_bit + if tex_coord == 1 { uv2_bit } else { 0.0 }
}

#[derive(Clone, Copy, Debug)]
pub struct Spin {
    pub axis: Vec3,
    pub radians_per_second: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct Orbit {
    pub center: Vec3,
    pub radius: f32,
    pub height: f32,
    pub radians_per_second: f32,
    pub phase: f32,
}

#[derive(Clone, Debug)]
pub struct Object {
    pub mesh: MeshKind,
    pub bounding_radius: Option<f32>,
    pub base: Transform,
    pub transform: Transform,
    pub material: Material,
    pub spin: Option<Spin>,
    pub orbit: Option<Orbit>,
    pub physics: Option<RigidBodySpec>,
}

impl Object {
    pub fn new(mesh: MeshKind, base: Transform, material: Material) -> Self {
        Self {
            mesh,
            bounding_radius: None,
            base,
            transform: base,
            material,
            spin: None,
            orbit: None,
            physics: None,
        }
    }

    pub fn bounding_radius(&self) -> f32 {
        let mesh_radius = self.bounding_radius.unwrap_or(match self.mesh {
            MeshKind::Cube => 1.732,
            MeshKind::Sphere => 1.0,
            MeshKind::Plane => std::f32::consts::FRAC_1_SQRT_2,
            MeshKind::Torus => 0.97,
            MeshKind::Gltf(_) => 1.0,
        });
        mesh_radius * self.transform.scale.max_element()
    }

    pub fn has_dynamic_physics(&self) -> bool {
        self.physics
            .as_ref()
            .is_some_and(|physics| physics.kind == RigidBodyKind::Dynamic)
    }

    fn update(&mut self, time: f32) {
        self.transform = self.base;
        if let Some(orbit) = self.orbit {
            let angle = orbit.phase + time * orbit.radians_per_second;
            self.transform.translation = orbit.center
                + Vec3::new(
                    angle.cos() * orbit.radius,
                    orbit.height,
                    angle.sin() * orbit.radius,
                );
        }
        if let Some(spin) = self.spin
            && spin.axis.length_squared() > 0.000001
        {
            let spin_rotation =
                Quat::from_axis_angle(spin.axis.normalize(), time * spin.radians_per_second);
            self.transform.rotation = spin_rotation * self.base.rotation;
        }
    }
}

fn hash01(index: usize, salt: u32) -> f32 {
    let mut value = (index as u32)
        .wrapping_mul(747_796_405)
        .wrapping_add(salt.wrapping_mul(2_891_233));
    value ^= value >> 16;
    value = value.wrapping_mul(2_246_822_519);
    value ^= value >> 13;
    (value % 1000) as f32 / 1000.0
}

fn fixed_trimesh(
    mesh_data: &MeshData,
    scale: Vec3,
    friction: f32,
    restitution: f32,
) -> RigidBodySpec {
    let vertices = mesh_data
        .vertices
        .iter()
        .map(|vertex| Vec3::from_array(vertex.position) * scale)
        .collect();
    let indices = mesh_data
        .indices
        .as_chunks::<3>()
        .0
        .iter()
        .map(|triangle| [triangle[0], triangle[1], triangle[2]])
        .collect();
    let mut body = RigidBodySpec::fixed(PhysicsShape::TriMesh { vertices, indices });
    body.friction = friction;
    body.restitution = restitution;
    body
}

#[derive(Clone, Copy, Debug)]
pub struct PointLight {
    pub position: Vec3,
    pub radius: f32,
    pub color: Vec3,
    pub intensity: f32,
}

#[derive(Debug)]
pub struct Scene {
    pub objects: Vec<Object>,
    pub asset_meshes: Vec<AssetMesh>,
    pub asset_textures: Vec<AssetTexture>,
    pub point_lights: Vec<PointLight>,
    pub sun_direction: Vec3,
    pub sun_color: Vec3,
    pub sun_intensity: f32,
    /// Tangent of the sun's angular radius, used for contact-hardening shadows.
    pub sun_angular_size: f32,
    pub time: f32,
    pub physics: PhysicsWorld,
    pub last_physics_microseconds: u128,
}

fn demo_asset_path() -> std::path::PathBuf {
    let runtime = std::path::PathBuf::from("assets/models/WaterBottle.glb");
    if runtime.exists() {
        runtime
    } else {
        std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/models/WaterBottle.glb"
        ))
    }
}

impl Scene {
    pub fn demo() -> Self {
        let mut objects = Vec::with_capacity(2100);

        let mut ground_material = Material::new(Vec3::new(0.18, 0.20, 0.19), 0.0, 0.82);
        ground_material.detail_kind = 2.0;
        ground_material.detail_scale = 32.0;
        let mut ground = Object::new(
            MeshKind::Plane,
            Transform::new(Vec3::ZERO, Vec3::ZERO, Vec3::splat(72.0)),
            ground_material,
        );
        let mut ground_body = RigidBodySpec::fixed(PhysicsShape::Cuboid {
            half_extents: Vec3::new(36.0, 0.5, 36.0),
        });
        ground_body.collider_offset = Vec3::new(0.0, -0.5, 0.0);
        ground_body.friction = 0.94;
        ground_body.restitution = 0.04;
        ground.physics = Some(ground_body);
        objects.push(ground);

        for row in 0..7 {
            for column in 0..11 {
                let x = (column as f32 - 5.0) * 1.45;
                let z = (row as f32 - 3.0) * 1.55 + 1.0;
                let roughness = column as f32 / 10.0;
                let metallic = row as f32 / 6.0;
                let material = Material::new(
                    Vec3::new(
                        0.78,
                        0.36 + row as f32 * 0.045,
                        0.12 + column as f32 * 0.025,
                    ),
                    metallic,
                    0.06 + roughness * 0.86,
                );
                let mut sphere = Object::new(
                    MeshKind::Sphere,
                    Transform::new(Vec3::new(x, 0.48, z), Vec3::ZERO, Vec3::splat(0.48)),
                    material,
                );
                sphere.spin = Some(Spin {
                    axis: Vec3::Y,
                    radians_per_second: 0.12,
                });
                let mut sphere_body = RigidBodySpec::fixed(PhysicsShape::Ball { radius: 0.48 });
                sphere_body.friction = 0.86;
                sphere_body.restitution = 0.12;
                sphere.physics = Some(sphere_body);
                objects.push(sphere);
            }
        }

        for index in 0usize..1250 {
            let ring = index / 50;
            let lane = index % 50;
            let angle = lane as f32 / 50.0 * std::f32::consts::TAU + ring as f32 * 0.041;
            let radius = 18.0 + ring as f32 * 1.22 + (lane % 4) as f32 * 0.18;
            let hash =
                ((index.wrapping_mul(747_796_405).wrapping_add(2_891_233)) % 1000) as f32 / 1000.0;
            let height = 0.8 + hash * hash * 5.2 + (ring % 5) as f32 * 0.18;
            let width = 0.58 + ((index * 31) % 100) as f32 / 260.0;
            let position = Vec3::new(angle.cos() * radius, height * 0.5, angle.sin() * radius);
            let color = 0.24 + hash * 0.25;
            let material = if index % 37 == 0 {
                Material::emissive(
                    Vec3::new(0.12, 0.16, 0.19),
                    Vec3::new(0.35, 0.62, 1.0) * 1.2,
                )
            } else {
                Material::new(
                    Vec3::new(color * 0.8, color * 0.9, color),
                    0.05 + hash * 0.22,
                    0.42 + hash * 0.35,
                )
            };
            let mut material = material;
            material.detail_kind = 1.0;
            material.detail_scale = height;
            let half_extents = Vec3::new(width, height * 0.5, width);
            let mut building = Object::new(
                MeshKind::Cube,
                Transform::new(position, Vec3::new(0.0, -angle, 0.0), half_extents),
                material,
            );
            let mut building_body = RigidBodySpec::fixed(PhysicsShape::Cuboid { half_extents });
            building_body.friction = 0.82;
            building_body.restitution = 0.04;
            building.physics = Some(building_body);
            objects.push(building);
        }

        let torus_mesh_data = MeshKind::Torus.generate();
        for index in 0..12 {
            let angle = index as f32 / 12.0 * std::f32::consts::TAU;
            let mut torus = Object::new(
                MeshKind::Torus,
                Transform::new(
                    Vec3::new(angle.cos() * 9.0, 1.2, angle.sin() * 9.0),
                    Vec3::new(0.0, angle, 0.0),
                    Vec3::splat(1.15),
                ),
                Material::new(Vec3::new(0.72, 0.68, 0.58), 0.88, 0.2),
            );
            torus.physics = Some(fixed_trimesh(
                &torus_mesh_data,
                Vec3::splat(1.15),
                0.48,
                0.24,
            ));
            objects.push(torus);
        }

        for index in 0..44 {
            let angle = index as f32 / 44.0 * std::f32::consts::TAU + hash01(index, 1) * 0.23;
            let spawn_radius = 2.4 + hash01(index, 2) * 5.2;
            let position = Vec3::new(
                angle.cos() * spawn_radius,
                2.8 + (index % 8) as f32 * 0.42 + hash01(index, 3) * 0.7,
                angle.sin() * spawn_radius * 0.78 + 0.8,
            );
            let inward = Vec3::new(-angle.cos(), 0.0, -angle.sin());
            let tangent = Vec3::new(-angle.sin(), 0.0, angle.cos());
            let linear_velocity = inward * (0.45 + hash01(index, 4) * 0.85)
                + tangent * ((hash01(index, 5) - 0.5) * 1.3)
                + Vec3::Y * (-0.1 + hash01(index, 6) * 0.25);
            let angular_velocity = Vec3::new(
                hash01(index, 7) - 0.5,
                hash01(index, 8) - 0.5,
                hash01(index, 9) - 0.5,
            ) * 3.0;
            let rotation = Vec3::new(
                hash01(index, 10) * std::f32::consts::TAU,
                hash01(index, 11) * std::f32::consts::TAU,
                hash01(index, 12) * std::f32::consts::TAU,
            );
            let use_cube = index % 3 == 0;
            let (mesh, scale, mut body) = if use_cube {
                let half_extents = Vec3::new(
                    0.24 + hash01(index, 13) * 0.12,
                    0.22 + hash01(index, 14) * 0.12,
                    0.24 + hash01(index, 15) * 0.12,
                );
                (
                    MeshKind::Cube,
                    half_extents,
                    RigidBodySpec::dynamic_cuboid(
                        half_extents,
                        linear_velocity,
                        angular_velocity,
                        0.72 + hash01(index, 16) * 0.22,
                        0.02 + hash01(index, 17) * 0.08,
                    ),
                )
            } else {
                let radius = 0.25 + hash01(index, 18) * 0.16;
                (
                    MeshKind::Sphere,
                    Vec3::splat(radius),
                    RigidBodySpec::dynamic_ball(
                        radius,
                        linear_velocity,
                        angular_velocity,
                        0.55 + hash01(index, 19) * 0.34,
                        0.14 + hash01(index, 20) * 0.22,
                    ),
                )
            };
            body.density = 0.8 + hash01(index, 21) * 1.5;
            let material = if use_cube {
                Material::new(
                    Vec3::new(
                        0.48 + hash01(index, 23) * 0.28,
                        0.25 + hash01(index, 24) * 0.18,
                        0.10 + hash01(index, 25) * 0.08,
                    ),
                    0.02,
                    0.62 + hash01(index, 26) * 0.25,
                )
            } else {
                Material::new(
                    Vec3::new(
                        0.12 + hash01(index, 27) * 0.28,
                        0.34 + hash01(index, 28) * 0.38,
                        0.48 + hash01(index, 29) * 0.46,
                    ),
                    0.08 + hash01(index, 30) * 0.38,
                    0.24 + hash01(index, 31) * 0.42,
                )
            };
            let mut object = Object::new(mesh, Transform::new(position, rotation, scale), material);
            object.physics = Some(body);
            objects.push(object);
        }

        let mut beacon = Object::new(
            MeshKind::Sphere,
            Transform::new(Vec3::new(0.0, 7.0, 0.0), Vec3::ZERO, Vec3::splat(0.7)),
            Material::emissive(
                Vec3::new(1.0, 0.74, 0.35),
                Vec3::new(1.0, 0.45, 0.12) * 12.0,
            ),
        );
        beacon.orbit = Some(Orbit {
            center: Vec3::ZERO,
            radius: 5.4,
            height: 6.2,
            radians_per_second: 0.42,
            phase: 0.0,
        });
        objects.push(beacon);

        let point_lights = (0..MAX_POINT_LIGHTS)
            .map(|i| {
                let angle = i as f32 / MAX_POINT_LIGHTS as f32 * std::f32::consts::TAU;
                PointLight {
                    position: Vec3::new(angle.cos() * 10.5, 3.2, angle.sin() * 10.5),
                    radius: 12.0,
                    color: match i % 3 {
                        0 => Vec3::new(1.0, 0.42, 0.16),
                        1 => Vec3::new(0.2, 0.55, 1.0),
                        _ => Vec3::new(0.72, 1.0, 0.46),
                    },
                    intensity: 42.0,
                }
            })
            .collect();

        let mut scene = Self {
            physics: PhysicsWorld::new(&[]),
            objects,
            asset_meshes: Vec::new(),
            asset_textures: Vec::new(),
            point_lights,
            sun_direction: Vec3::new(-0.42, -0.78, -0.46).normalize(),
            sun_color: Vec3::new(1.0, 0.86, 0.68),
            sun_intensity: 3.2,
            sun_angular_size: 0.045,
            time: 0.0,
            last_physics_microseconds: 0,
        };

        let model_path = demo_asset_path();
        let asset_transform = Transform::new(
            Vec3::new(3.0, 1.08, 5.0),
            Vec3::new(0.0, 0.55, 0.0),
            Vec3::splat(8.0),
        );
        if let Err(error) = scene.add_gltf_object(&model_path, asset_transform) {
            log::warn!(
                "failed to load bundled glTF asset {}: {error:#}",
                model_path.display()
            );
        }
        scene.physics = PhysicsWorld::new(&scene.objects);
        scene
    }

    pub fn add_gltf_object(
        &mut self,
        path: impl AsRef<std::path::Path>,
        transform: Transform,
    ) -> anyhow::Result<usize> {
        let asset = assets::load_gltf(path)?;
        let texture_offset = self.asset_textures.len();
        let mut created = 0;
        for mut asset_mesh in asset.meshes {
            if let Some(index) = asset_mesh.material.base_color_texture.as_mut() {
                *index += texture_offset;
            }
            if let Some(index) = asset_mesh.material.normal_texture.as_mut() {
                *index += texture_offset;
            }
            if let Some(index) = asset_mesh.material.metallic_roughness_texture.as_mut() {
                *index += texture_offset;
            }
            if let Some(index) = asset_mesh.material.emissive_texture.as_mut() {
                *index += texture_offset;
            }
            if let Some(index) = asset_mesh.material.occlusion_texture.as_mut() {
                *index += texture_offset;
            }
            let material = material_from_asset(asset_mesh.material);
            let mesh_index = self.asset_meshes.len() as u32;
            let bounding_radius = asset_mesh.mesh.bounding_radius;
            self.asset_meshes.push(asset_mesh);
            let mut object = Object::new(MeshKind::Gltf(mesh_index), transform, material);
            object.bounding_radius = Some(bounding_radius);
            object.spin = Some(Spin {
                axis: Vec3::Y,
                radians_per_second: 0.18,
            });
            self.objects.push(object);
            created += 1;
        }
        self.asset_textures.extend(asset.textures);
        Ok(created)
    }

    pub fn update(&mut self, dt: f32) {
        self.time += dt;
        let time = self.time;
        self.objects.par_iter_mut().for_each(|object| {
            if !object.has_dynamic_physics() {
                object.update(time);
            }
        });
        let physics_start = std::time::Instant::now();
        self.physics.step(dt, &mut self.objects);
        self.last_physics_microseconds = physics_start.elapsed().as_micros();
        for (index, light) in self.point_lights.iter_mut().enumerate() {
            let angle =
                self.time * 0.24 + index as f32 / MAX_POINT_LIGHTS as f32 * std::f32::consts::TAU;
            light.position = Vec3::new(
                angle.cos() * 10.5,
                2.8 + (self.time * 0.8 + index as f32).sin() * 0.7,
                angle.sin() * 10.5,
            );
        }
    }

    pub fn visible_indices(&self, frustum: &Frustum) -> Vec<usize> {
        let mut visible: Vec<usize> = self
            .objects
            .par_iter()
            .enumerate()
            .filter_map(|(index, object)| {
                frustum
                    .intersects_sphere(object.transform.translation, object.bounding_radius())
                    .then_some(index)
            })
            .collect();
        visible.sort_unstable_by_key(|&index| self.objects[index].mesh);
        visible
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demo_scene_is_large_and_bounded() {
        let scene = Scene::demo();
        assert!(scene.objects.len() > 1_300);
        assert!(scene.point_lights.len() == MAX_POINT_LIGHTS);
        assert_eq!(scene.physics.dynamic_body_count(), 44);
        assert!(
            scene
                .objects
                .iter()
                .all(|object| object.bounding_radius().is_finite())
        );
    }

    #[test]
    fn demo_physics_objects_settle_without_tunneling() {
        let mut scene = Scene::demo();
        for _ in 0..240 {
            scene.physics.step(1.0 / 60.0, &mut scene.objects);
        }
        let minimum_center_height = scene
            .objects
            .iter()
            .filter(|object| object.has_dynamic_physics())
            .map(|object| object.transform.translation.y)
            .fold(f32::INFINITY, f32::min);
        assert!(
            minimum_center_height > 0.05,
            "minimum_center_height={minimum_center_height}"
        );
    }

    #[test]
    fn spin_preserves_its_rotation_axis() {
        let axis = Vec3::new(0.3, 1.0, 0.1).normalize();
        let mut object = Object::new(
            MeshKind::Torus,
            Transform::new(Vec3::ZERO, Vec3::ZERO, Vec3::ONE),
            Material::new(Vec3::ONE, 0.8, 0.2),
        );
        object.spin = Some(Spin {
            axis,
            radians_per_second: 0.35,
        });
        object.update(1.73);

        let rotated_axis = object
            .transform
            .matrix()
            .transform_vector3(axis)
            .normalize();
        assert!((rotated_axis - axis).length() < 0.0001);
    }
}
