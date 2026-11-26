//! Photon3D: a data-oriented realtime physically based rendering engine.
//!
//! The crate is split into small reusable subsystems: camera math and frustum
//! culling, procedural mesh generation, scene representation, and the wgpu
//! renderer. The binary in `main.rs` is a reference sandbox using these pieces.

pub mod assets;
pub mod camera;
pub mod mesh;
pub mod physics;
pub mod renderer;
pub mod scene;
