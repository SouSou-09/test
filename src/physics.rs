use glam::{Quat, Vec3};
use rapier3d::prelude::{
    BroadPhaseBvh, CCDSolver, ColliderBuilder, ColliderSet, ImpulseJointSet, IntegrationParameters,
    IslandManager, MultibodyJointSet, NarrowPhase, PhysicsPipeline, Pose, RigidBodyBuilder,
    RigidBodyHandle, RigidBodySet, Rotation, SoftBodySet, Vector,
};

use crate::scene::Object;

const FIXED_TIMESTEP: f32 = 1.0 / 60.0;
const MAX_SUBSTEPS: usize = 5;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RigidBodyKind {
    Dynamic,
    Fixed,
}

#[derive(Clone, Debug)]
pub enum PhysicsShape {
    Ball {
        radius: f32,
    },
    Cuboid {
        half_extents: Vec3,
    },
    TriMesh {
        vertices: Vec<Vec3>,
        indices: Vec<[u32; 3]>,
    },
}

impl PhysicsShape {
    pub fn bounding_radius(&self) -> f32 {
        match self {
            Self::Ball { radius } => *radius,
            Self::Cuboid { half_extents } => half_extents.length(),
            Self::TriMesh { vertices, .. } => vertices
                .iter()
                .map(|vertex| vertex.length())
                .fold(0.0, f32::max),
        }
    }
}

#[derive(Clone, Debug)]
pub struct RigidBodySpec {
    pub kind: RigidBodyKind,
    pub shape: PhysicsShape,
    pub density: f32,
    pub friction: f32,
    pub restitution: f32,
    pub linear_velocity: Vec3,
    pub angular_velocity: Vec3,
    pub linear_damping: f32,
    pub angular_damping: f32,
    pub ccd: bool,
    pub collider_offset: Vec3,
    /// Seconds before the sandbox respawns this body at its initial state; zero disables respawning.
    pub respawn_interval: f32,
}

impl RigidBodySpec {
    fn new(kind: RigidBodyKind, shape: PhysicsShape) -> Self {
        Self {
            kind,
            shape,
            density: 1.0,
            friction: 0.82,
            restitution: 0.08,
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            linear_damping: 0.015,
            angular_damping: 0.04,
            ccd: kind == RigidBodyKind::Dynamic,
            collider_offset: Vec3::ZERO,
            respawn_interval: 0.0,
        }
    }

    pub fn fixed(shape: PhysicsShape) -> Self {
        Self::new(RigidBodyKind::Fixed, shape)
    }

    pub fn dynamic(shape: PhysicsShape) -> Self {
        Self::new(RigidBodyKind::Dynamic, shape)
    }

    pub fn dynamic_ball(
        radius: f32,
        linear_velocity: Vec3,
        angular_velocity: Vec3,
        friction: f32,
        restitution: f32,
    ) -> Self {
        let mut body = Self::dynamic(PhysicsShape::Ball { radius });
        body.linear_velocity = linear_velocity;
        body.angular_velocity = angular_velocity;
        body.friction = friction;
        body.restitution = restitution;
        body
    }

    pub fn dynamic_cuboid(
        half_extents: Vec3,
        linear_velocity: Vec3,
        angular_velocity: Vec3,
        friction: f32,
        restitution: f32,
    ) -> Self {
        let mut body = Self::dynamic(PhysicsShape::Cuboid { half_extents });
        body.linear_velocity = linear_velocity;
        body.angular_velocity = angular_velocity;
        body.friction = friction;
        body.restitution = restitution;
        body
    }
}

struct DynamicBinding {
    object_index: usize,
    body_handle: RigidBodyHandle,
    spawn_translation: Vector,
    spawn_rotation: Rotation,
    spawn_linear_velocity: Vector,
    spawn_angular_velocity: Vector,
    respawn_interval: f32,
    age: f32,
}

pub struct PhysicsWorld {
    pipeline: PhysicsPipeline,
    integration_parameters: IntegrationParameters,
    gravity: Vector,
    islands: IslandManager,
    broad_phase: BroadPhaseBvh,
    narrow_phase: NarrowPhase,
    bodies: RigidBodySet,
    colliders: ColliderSet,
    impulse_joints: ImpulseJointSet,
    multibody_joints: MultibodyJointSet,
    soft_bodies: SoftBodySet,
    ccd_solver: CCDSolver,
    accumulator: f32,
    dynamic_bodies: Vec<DynamicBinding>,
}

impl std::fmt::Debug for PhysicsWorld {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PhysicsWorld")
            .field("bodies", &self.bodies.len())
            .field("colliders", &self.colliders.len())
            .field("dynamic_bodies", &self.dynamic_bodies.len())
            .finish()
    }
}

fn rapier_vec(vector: Vec3) -> Vector {
    Vector::new(vector.x, vector.y, vector.z)
}

fn glam_vec(vector: Vector) -> Vec3 {
    Vec3::new(vector.x, vector.y, vector.z)
}

fn rapier_quat(rotation: Quat) -> Rotation {
    Rotation::from_xyzw(rotation.x, rotation.y, rotation.z, rotation.w)
}

fn glam_quat(rotation: Rotation) -> Quat {
    Quat::from_xyzw(rotation.x, rotation.y, rotation.z, rotation.w)
}

fn rapier_pose(translation: Vec3, rotation: Quat) -> Pose {
    Pose::from_parts(rapier_vec(translation), rapier_quat(rotation))
}

impl PhysicsWorld {
    pub fn new(objects: &[Object]) -> Self {
        let mut world = Self {
            pipeline: PhysicsPipeline::new(),
            integration_parameters: IntegrationParameters::default(),
            gravity: Vector::new(0.0, -9.81, 0.0),
            islands: IslandManager::new(),
            broad_phase: BroadPhaseBvh::new(),
            narrow_phase: NarrowPhase::new(),
            bodies: RigidBodySet::with_capacity(objects.len()),
            colliders: ColliderSet::with_capacity(objects.len()),
            impulse_joints: ImpulseJointSet::new(),
            multibody_joints: MultibodyJointSet::new(),
            soft_bodies: SoftBodySet::new(),
            ccd_solver: CCDSolver::new(),
            accumulator: 0.0,
            dynamic_bodies: Vec::new(),
        };
        world.integration_parameters.dt = FIXED_TIMESTEP;
        world.integration_parameters.num_solver_iterations = 6;

        for (object_index, object) in objects.iter().enumerate() {
            let Some(spec) = object.physics.as_ref() else {
                continue;
            };
            let pose = rapier_pose(object.transform.translation, object.transform.rotation);
            let body = match spec.kind {
                RigidBodyKind::Dynamic => RigidBodyBuilder::dynamic()
                    .pose(pose)
                    .linvel(rapier_vec(spec.linear_velocity))
                    .angvel(rapier_vec(spec.angular_velocity))
                    .linear_damping(spec.linear_damping)
                    .angular_damping(spec.angular_damping)
                    .ccd_enabled(spec.ccd)
                    .build(),
                RigidBodyKind::Fixed => RigidBodyBuilder::fixed().pose(pose).build(),
            };
            let body_handle = world.bodies.insert(body);
            let shape_builder = match &spec.shape {
                PhysicsShape::Ball { radius } => ColliderBuilder::ball(*radius),
                PhysicsShape::Cuboid { half_extents } => {
                    ColliderBuilder::cuboid(half_extents.x, half_extents.y, half_extents.z)
                }
                PhysicsShape::TriMesh { vertices, indices } => {
                    let vertices: Vec<Vector> = vertices.iter().copied().map(rapier_vec).collect();
                    ColliderBuilder::trimesh(vertices, indices.clone())
                        .expect("physics trimesh collider must be valid")
                }
            };
            let collider = shape_builder
                .translation(rapier_vec(spec.collider_offset))
                .density(spec.density)
                .friction(spec.friction)
                .restitution(spec.restitution)
                .build();
            world
                .colliders
                .insert_with_parent(collider, body_handle, &mut world.bodies);
            if spec.kind == RigidBodyKind::Dynamic {
                world.dynamic_bodies.push(DynamicBinding {
                    object_index,
                    body_handle,
                    spawn_translation: rapier_vec(object.transform.translation),
                    spawn_rotation: rapier_quat(object.transform.rotation),
                    spawn_linear_velocity: rapier_vec(spec.linear_velocity),
                    spawn_angular_velocity: rapier_vec(spec.angular_velocity),
                    respawn_interval: spec.respawn_interval,
                    age: 0.0,
                });
            }
        }
        world
    }

    pub fn step(&mut self, frame_dt: f32, objects: &mut [Object]) {
        self.accumulator = (self.accumulator + frame_dt.max(0.0)).min(0.12);
        let mut substeps = 0;
        while self.accumulator >= FIXED_TIMESTEP && substeps < MAX_SUBSTEPS {
            self.pipeline.step(
                self.gravity,
                &self.integration_parameters,
                &mut self.islands,
                &mut self.broad_phase,
                &mut self.narrow_phase,
                &mut self.bodies,
                &mut self.colliders,
                &mut self.impulse_joints,
                &mut self.multibody_joints,
                &mut self.soft_bodies,
                &mut self.ccd_solver,
                &(),
                &(),
            );
            self.accumulator -= FIXED_TIMESTEP;
            substeps += 1;
            self.update_respawns(FIXED_TIMESTEP);
        }
        self.sync_objects(objects);
    }

    fn update_respawns(&mut self, dt: f32) {
        for binding in &mut self.dynamic_bodies {
            if binding.respawn_interval <= 0.0 {
                continue;
            }
            binding.age += dt;
            let Some(body) = self.bodies.get_mut(binding.body_handle) else {
                continue;
            };
            let fell_out = body.translation().y < -5.0;
            if binding.age < binding.respawn_interval && !fell_out {
                continue;
            }
            body.set_translation(binding.spawn_translation, true);
            body.set_rotation(binding.spawn_rotation, true);
            body.set_linvel(binding.spawn_linear_velocity, true);
            body.set_angvel(binding.spawn_angular_velocity, true);
            binding.age = 0.0;
        }
    }

    fn sync_objects(&self, objects: &mut [Object]) {
        for binding in &self.dynamic_bodies {
            let (Some(object), Some(body)) = (
                objects.get_mut(binding.object_index),
                self.bodies.get(binding.body_handle),
            ) else {
                continue;
            };
            object.transform.translation = glam_vec(body.translation());
            object.transform.rotation = glam_quat(*body.rotation());
        }
    }

    pub fn dynamic_body_count(&self) -> usize {
        self.dynamic_bodies.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        mesh::MeshKind,
        scene::{Material, Object, Transform},
    };

    fn physics_objects() -> Vec<Object> {
        let mut ground = Object::new(
            MeshKind::Plane,
            Transform::new(Vec3::ZERO, Vec3::ZERO, Vec3::splat(20.0)),
            Material::new(Vec3::ONE, 0.0, 0.8),
        );
        let mut ground_body = RigidBodySpec::fixed(PhysicsShape::Cuboid {
            half_extents: Vec3::new(20.0, 0.1, 20.0),
        });
        ground_body.collider_offset = Vec3::new(0.0, -0.1, 0.0);
        ground_body.friction = 0.9;
        ground.physics = Some(ground_body);

        let mut sphere = Object::new(
            MeshKind::Sphere,
            Transform::new(Vec3::new(0.0, 4.0, 0.0), Vec3::ZERO, Vec3::splat(0.5)),
            Material::new(Vec3::ONE, 0.0, 0.8),
        );
        sphere.physics = Some(RigidBodySpec::dynamic_ball(
            0.5,
            Vec3::ZERO,
            Vec3::ZERO,
            0.8,
            0.1,
        ));
        vec![ground, sphere]
    }

    #[test]
    fn falling_sphere_settles_on_ground() {
        let mut objects = physics_objects();
        let mut world = PhysicsWorld::new(&objects);
        for _ in 0..180 {
            world.step(1.0 / 60.0, &mut objects);
        }
        let height = objects[1].transform.translation.y;
        assert!((height - 0.5).abs() < 0.035, "height={height}");
    }

    #[test]
    fn restitution_returns_energy_after_ground_collision() {
        let mut objects = physics_objects();
        objects[1].transform.translation.y = 2.5;
        if let Some(physics) = objects[1].physics.as_mut() {
            physics.restitution = 0.9;
        }
        let mut world = PhysicsWorld::new(&objects);
        let mut minimum_height = f32::INFINITY;
        for _ in 0..45 {
            world.step(1.0 / 60.0, &mut objects);
            minimum_height = minimum_height.min(objects[1].transform.translation.y);
        }
        assert!(minimum_height < 0.55, "minimum_height={minimum_height}");
        let mut rebound_height: f32 = 0.0;
        for _ in 0..35 {
            world.step(1.0 / 60.0, &mut objects);
            rebound_height = rebound_height.max(objects[1].transform.translation.y);
        }
        assert!(
            rebound_height > minimum_height + 0.35,
            "rebound_height={rebound_height} minimum_height={minimum_height}"
        );
    }

    #[test]
    fn low_fps_frames_preserve_fractional_physics_time() {
        let mut low_fps_objects = physics_objects();
        let mut fixed_objects = physics_objects();
        let mut low_fps = PhysicsWorld::new(&low_fps_objects);
        let mut fixed = PhysicsWorld::new(&fixed_objects);
        for _ in 0..5 {
            low_fps.step(0.08, &mut low_fps_objects);
        }
        for _ in 0..24 {
            fixed.step(1.0 / 60.0, &mut fixed_objects);
        }
        let difference =
            low_fps_objects[1].transform.translation - fixed_objects[1].transform.translation;
        assert!(difference.length() < 0.0001, "difference={difference:?}");
    }

    #[test]
    fn respawn_timing_is_independent_of_render_frame_grouping() {
        fn respawning_objects() -> Vec<Object> {
            let mut objects = physics_objects();
            if let Some(physics) = objects[1].physics.as_mut() {
                physics.linear_velocity = Vec3::X;
                physics.respawn_interval = 29.0 / 60.0;
            }
            objects
        }

        let mut coarse_objects = respawning_objects();
        let mut fine_objects = respawning_objects();
        let mut coarse = PhysicsWorld::new(&coarse_objects);
        let mut fine = PhysicsWorld::new(&fine_objects);
        for _ in 0..30 {
            coarse.step(1.0 / 30.0, &mut coarse_objects);
        }
        for _ in 0..60 {
            fine.step(1.0 / 60.0, &mut fine_objects);
        }
        let difference =
            coarse_objects[1].transform.translation - fine_objects[1].transform.translation;
        assert!(difference.length() < 0.0001, "difference={difference:?}");
    }

    #[test]
    fn fast_falling_sphere_does_not_tunnel_through_thin_ground() {
        let mut objects = physics_objects();
        objects[1].transform.translation.y = 50.0;
        objects[1].transform.scale = Vec3::splat(0.1);
        if let Some(physics) = objects[1].physics.as_mut() {
            physics.shape = PhysicsShape::Ball { radius: 0.1 };
        }
        let mut world = PhysicsWorld::new(&objects);
        for _ in 0..300 {
            world.step(1.0 / 60.0, &mut objects);
        }
        let height = objects[1].transform.translation.y;
        assert!(height > 0.05, "height={height}");
    }

    #[test]
    fn physics_time_is_deterministic_for_fixed_steps() {
        let mut first_objects = physics_objects();
        let mut second_objects = physics_objects();
        let mut first = PhysicsWorld::new(&first_objects);
        let mut second = PhysicsWorld::new(&second_objects);
        for _ in 0..90 {
            first.step(1.0 / 60.0, &mut first_objects);
            second.step(1.0 / 60.0, &mut second_objects);
        }
        let a = first_objects[1].transform.translation;
        let b = second_objects[1].transform.translation;
        assert!((a - b).length() < 0.0001);
    }
}
