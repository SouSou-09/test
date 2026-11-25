use glam::{EulerRot, Mat4, Quat, Vec3};
use rayon::prelude::*;

use crate::{camera::Frustum, mesh::MeshKind};

pub const MAX_POINT_LIGHTS: usize = 8;

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
        }
    }
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
    pub base: Transform,
    pub transform: Transform,
    pub material: Material,
    pub spin: Option<Spin>,
    pub orbit: Option<Orbit>,
}

impl Object {
    pub fn new(mesh: MeshKind, base: Transform, material: Material) -> Self {
        Self {
            mesh,
            base,
            transform: base,
            material,
            spin: None,
            orbit: None,
        }
    }

    pub fn bounding_radius(&self) -> f32 {
        let mesh_radius = match self.mesh {
            MeshKind::Cube => 1.732,
            MeshKind::Sphere => 1.0,
            MeshKind::Plane => std::f32::consts::FRAC_1_SQRT_2,
            MeshKind::Torus => 0.97,
        };
        mesh_radius * self.transform.scale.max_element()
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
        if let Some(spin) = self.spin {
            if spin.axis.length_squared() > 0.000001 {
                let spin_rotation =
                    Quat::from_axis_angle(spin.axis.normalize(), time * spin.radians_per_second);
                self.transform.rotation = spin_rotation * self.base.rotation;
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PointLight {
    pub position: Vec3,
    pub radius: f32,
    pub color: Vec3,
    pub intensity: f32,
}

#[derive(Clone, Debug)]
pub struct Scene {
    pub objects: Vec<Object>,
    pub point_lights: Vec<PointLight>,
    pub sun_direction: Vec3,
    pub sun_color: Vec3,
    pub sun_intensity: f32,
    pub time: f32,
}

impl Scene {
    pub fn demo() -> Self {
        let mut objects = Vec::with_capacity(2100);

        let mut ground_material = Material::new(Vec3::new(0.18, 0.20, 0.19), 0.0, 0.82);
        ground_material.detail_kind = 2.0;
        ground_material.detail_scale = 32.0;
        objects.push(Object::new(
            MeshKind::Plane,
            Transform::new(Vec3::ZERO, Vec3::ZERO, Vec3::splat(72.0)),
            ground_material,
        ));

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
                    Transform::new(Vec3::new(x, 1.0, z), Vec3::ZERO, Vec3::splat(0.48)),
                    material,
                );
                sphere.spin = Some(Spin {
                    axis: Vec3::Y,
                    radians_per_second: 0.12,
                });
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
            objects.push(Object::new(
                MeshKind::Cube,
                Transform::new(
                    position,
                    Vec3::new(0.0, -angle, 0.0),
                    Vec3::new(width, height * 0.5, width),
                ),
                material,
            ));
        }

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
            torus.spin = Some(Spin {
                axis: Vec3::new(0.3, 1.0, 0.1).normalize(),
                radians_per_second: 0.35,
            });
            objects.push(torus);
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

        Self {
            objects,
            point_lights,
            sun_direction: Vec3::new(-0.42, -0.78, -0.46).normalize(),
            sun_color: Vec3::new(1.0, 0.86, 0.68),
            sun_intensity: 3.2,
            time: 0.0,
        }
    }

    pub fn update(&mut self, dt: f32) {
        self.time += dt;
        let time = self.time;
        self.objects
            .par_iter_mut()
            .for_each(|object| object.update(time));
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
        assert!(
            scene
                .objects
                .iter()
                .all(|object| object.bounding_radius().is_finite())
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
