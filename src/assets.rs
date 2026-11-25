use std::path::Path;

use anyhow::{Context, Result, bail};
use glam::{Mat4, Vec2, Vec3};

use crate::mesh::{MeshData, Vertex};

#[derive(Clone, Debug)]
pub struct AssetTexture {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AssetMaterial {
    pub base_color: Vec3,
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: Vec3,
    pub base_color_texture: Option<usize>,
    pub normal_texture: Option<usize>,
    pub metallic_roughness_texture: Option<usize>,
    pub emissive_texture: Option<usize>,
    pub occlusion_texture: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct AssetMesh {
    pub mesh: MeshData,
    pub material: AssetMaterial,
}

#[derive(Clone, Debug, Default)]
pub struct GltfAsset {
    pub meshes: Vec<AssetMesh>,
    pub textures: Vec<AssetTexture>,
}

pub fn load_gltf(path: impl AsRef<Path>) -> Result<GltfAsset> {
    let path = path.as_ref();
    let (document, buffers, images) =
        gltf::import(path).with_context(|| format!("load glTF asset {}", path.display()))?;
    let textures = images
        .iter()
        .map(convert_texture)
        .collect::<Result<Vec<_>>>()?;
    let mut meshes = Vec::new();
    let scene = document
        .default_scene()
        .or_else(|| document.scenes().next())
        .context("glTF file contains no scenes")?;
    for node in scene.nodes() {
        visit_node(&node, Mat4::IDENTITY, &buffers, &mut meshes);
    }
    if meshes.is_empty() {
        bail!("glTF scene contains no triangle mesh primitives");
    }
    Ok(GltfAsset { meshes, textures })
}

fn visit_node(
    node: &gltf::Node<'_>,
    parent_transform: Mat4,
    buffers: &[gltf::buffer::Data],
    meshes: &mut Vec<AssetMesh>,
) {
    let local_transform = Mat4::from_cols_array_2d(&node.transform().matrix());
    let world_transform = parent_transform * local_transform;
    if let Some(mesh) = node.mesh() {
        for primitive in mesh.primitives() {
            if primitive.mode() == gltf::mesh::Mode::Triangles {
                if let Some(asset_mesh) = convert_primitive(&primitive, world_transform, buffers) {
                    meshes.push(asset_mesh);
                }
            }
        }
    }
    for child in node.children() {
        visit_node(&child, world_transform, buffers, meshes);
    }
}

fn convert_primitive(
    primitive: &gltf::Primitive<'_>,
    transform: Mat4,
    buffers: &[gltf::buffer::Data],
) -> Option<AssetMesh> {
    let reader =
        primitive.reader(|buffer| buffers.get(buffer.index()).map(|data| data.0.as_slice()));
    let positions: Vec<[f32; 3]> = reader.read_positions()?.collect();
    let normals: Vec<[f32; 3]> = reader
        .read_normals()
        .map(|normals| normals.collect())
        .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]);
    let uvs: Vec<[f32; 2]> = reader
        .read_tex_coords(0)
        .map(|uvs| uvs.into_f32().collect())
        .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);
    let tangents: Vec<[f32; 4]> = reader
        .read_tangents()
        .map(|tangents| tangents.collect())
        .unwrap_or_default();
    let indices: Vec<u32> = reader
        .read_indices()
        .map(|indices| indices.into_u32().collect())
        .unwrap_or_else(|| (0..positions.len() as u32).collect());

    let normal_transform = transform.inverse().transpose();
    let mut vertices = Vec::with_capacity(positions.len());
    for index in 0..positions.len() {
        let position = transform
            .transform_point3(Vec3::from_array(positions[index]))
            .to_array();
        let normal = normal_transform
            .transform_vector3(Vec3::from_array(normals[index]))
            .normalize_or_zero();
        let tangent = if let Some(raw_tangent) = tangents.get(index) {
            let transformed = normal_transform
                .transform_vector3(Vec3::new(raw_tangent[0], raw_tangent[1], raw_tangent[2]))
                .normalize_or_zero();
            [transformed.x, transformed.y, transformed.z, raw_tangent[3]]
        } else {
            [0.0, 0.0, 0.0, 1.0]
        };
        vertices.push(Vertex {
            position,
            normal: normal.to_array(),
            uv: uvs[index],
            tangent,
        });
    }
    if reader.read_normals().is_none() {
        compute_normals(&mut vertices, &indices);
    }
    if tangents.is_empty() {
        compute_tangents(&mut vertices, &indices);
    }

    let mesh = MeshData {
        vertices,
        indices,
        bounding_radius: 0.0,
    }
    .finish();
    Some(AssetMesh {
        mesh,
        material: convert_material(&primitive.material()),
    })
}

fn compute_normals(vertices: &mut [Vertex], indices: &[u32]) {
    let mut normals = vec![Vec3::ZERO; vertices.len()];
    for triangle in indices.chunks_exact(3) {
        let [a, b, c] = [
            triangle[0] as usize,
            triangle[1] as usize,
            triangle[2] as usize,
        ];
        if a >= vertices.len() || b >= vertices.len() || c >= vertices.len() {
            continue;
        }
        let p0 = Vec3::from_array(vertices[a].position);
        let p1 = Vec3::from_array(vertices[b].position);
        let p2 = Vec3::from_array(vertices[c].position);
        let normal = (p1 - p0).cross(p2 - p0);
        for index in [a, b, c] {
            normals[index] += normal;
        }
    }
    for (index, vertex) in vertices.iter_mut().enumerate() {
        vertex.normal = normals[index].normalize_or(Vec3::Y).to_array();
    }
}

fn compute_tangents(vertices: &mut [Vertex], indices: &[u32]) {
    let mut tangents = vec![Vec3::ZERO; vertices.len()];
    let mut bitangents = vec![Vec3::ZERO; vertices.len()];
    for triangle in indices.chunks_exact(3) {
        let [a, b, c] = [
            triangle[0] as usize,
            triangle[1] as usize,
            triangle[2] as usize,
        ];
        if a >= vertices.len() || b >= vertices.len() || c >= vertices.len() {
            continue;
        }
        let p0 = Vec3::from_array(vertices[a].position);
        let p1 = Vec3::from_array(vertices[b].position);
        let p2 = Vec3::from_array(vertices[c].position);
        let uv0 = Vec2::from_array(vertices[a].uv);
        let uv1 = Vec2::from_array(vertices[b].uv);
        let uv2 = Vec2::from_array(vertices[c].uv);
        let edge1 = p1 - p0;
        let edge2 = p2 - p0;
        let delta_uv1 = uv1 - uv0;
        let delta_uv2 = uv2 - uv0;
        let denominator = delta_uv1.x * delta_uv2.y - delta_uv2.x * delta_uv1.y;
        if denominator.abs() < 0.000001 {
            continue;
        }
        let scale = 1.0 / denominator;
        let tangent = (edge1 * delta_uv2.y - edge2 * delta_uv1.y) * scale;
        let bitangent = (edge2 * delta_uv1.x - edge1 * delta_uv2.x) * scale;
        for index in [a, b, c] {
            tangents[index] += tangent;
            bitangents[index] += bitangent;
        }
    }
    for (index, vertex) in vertices.iter_mut().enumerate() {
        let normal = Vec3::from_array(vertex.normal).normalize_or_zero();
        let tangent = (tangents[index] - normal * tangents[index].dot(normal)).normalize_or_zero();
        let fallback = if normal.cross(Vec3::Y).length_squared() > 0.01 {
            normal.cross(Vec3::Y).normalize()
        } else {
            Vec3::X
        };
        let tangent = if tangent.length_squared() > 0.000001 {
            tangent
        } else {
            fallback
        };
        let handedness = normal
            .cross(tangent)
            .dot(bitangents[index])
            .signum()
            .max(-1.0);
        vertex.tangent = [
            tangent.x,
            tangent.y,
            tangent.z,
            if handedness < 0.0 { -1.0 } else { 1.0 },
        ];
    }
}

fn convert_material(material: &gltf::Material<'_>) -> AssetMaterial {
    let pbr = material.pbr_metallic_roughness();
    let base = pbr.base_color_factor();
    let emissive = material.emissive_factor();
    AssetMaterial {
        base_color: Vec3::new(base[0], base[1], base[2]),
        metallic: pbr.metallic_factor(),
        roughness: pbr.roughness_factor(),
        emissive: Vec3::new(emissive[0], emissive[1], emissive[2]),
        base_color_texture: pbr
            .base_color_texture()
            .map(|texture| texture.texture().source().index()),
        normal_texture: material
            .normal_texture()
            .map(|texture| texture.texture().source().index()),
        metallic_roughness_texture: pbr
            .metallic_roughness_texture()
            .map(|texture| texture.texture().source().index()),
        emissive_texture: material
            .emissive_texture()
            .map(|texture| texture.texture().source().index()),
        occlusion_texture: material
            .occlusion_texture()
            .map(|texture| texture.texture().source().index()),
    }
}

fn convert_texture(image: &gltf::image::Data) -> Result<AssetTexture> {
    let mut rgba = Vec::with_capacity((image.width * image.height * 4) as usize);
    match image.format {
        gltf::image::Format::R8 => {
            for &r in &image.pixels {
                rgba.extend_from_slice(&[r, r, r, 255]);
            }
        }
        gltf::image::Format::R8G8 => {
            for pixel in image.pixels.chunks_exact(2) {
                rgba.extend_from_slice(&[pixel[0], pixel[1], 0, 255]);
            }
        }
        gltf::image::Format::R8G8B8 => {
            for pixel in image.pixels.chunks_exact(3) {
                rgba.extend_from_slice(&[pixel[0], pixel[1], pixel[2], 255]);
            }
        }
        gltf::image::Format::R8G8B8A8 => rgba.extend_from_slice(&image.pixels),
        gltf::image::Format::R16 => {
            for bytes in image.pixels.chunks_exact(2) {
                let value = u16::from_le_bytes([bytes[0], bytes[1]]) >> 8;
                rgba.extend_from_slice(&[value as u8, value as u8, value as u8, 255]);
            }
        }
        gltf::image::Format::R16G16 => {
            for bytes in image.pixels.chunks_exact(4) {
                let r = u16::from_le_bytes([bytes[0], bytes[1]]) >> 8;
                let g = u16::from_le_bytes([bytes[2], bytes[3]]) >> 8;
                rgba.extend_from_slice(&[r as u8, g as u8, 0, 255]);
            }
        }
        gltf::image::Format::R16G16B16 => {
            for bytes in image.pixels.chunks_exact(6) {
                for channel in 0..3 {
                    rgba.push(
                        (u16::from_le_bytes([bytes[channel * 2], bytes[channel * 2 + 1]]) >> 8)
                            as u8,
                    );
                }
                rgba.push(255);
            }
        }
        gltf::image::Format::R16G16B16A16 => {
            for bytes in image.pixels.chunks_exact(8) {
                for channel in 0..4 {
                    rgba.push(
                        (u16::from_le_bytes([bytes[channel * 2], bytes[channel * 2 + 1]]) >> 8)
                            as u8,
                    );
                }
            }
        }
        gltf::image::Format::R32G32B32FLOAT => {
            for bytes in image.pixels.chunks_exact(12) {
                for channel in 0..3 {
                    rgba.push(float_to_u8(bytes, channel));
                }
                rgba.push(255);
            }
        }
        gltf::image::Format::R32G32B32A32FLOAT => {
            for bytes in image.pixels.chunks_exact(16) {
                for channel in 0..4 {
                    rgba.push(float_to_u8(bytes, channel));
                }
            }
        }
    }
    Ok(AssetTexture {
        width: image.width,
        height: image.height,
        rgba,
    })
}

fn float_to_u8(bytes: &[u8], channel: usize) -> u8 {
    let offset = channel * 4;
    let value = f32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ]);
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn water_bottle_asset_loads_textured_pbr_geometry() {
        let asset = load_gltf(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/models/WaterBottle.glb"
        ))
        .expect("load bundled glTF");
        assert!(!asset.meshes.is_empty());
        assert!(!asset.textures.is_empty());
        assert!(
            asset
                .meshes
                .iter()
                .all(|mesh| mesh.mesh.indices.len() % 3 == 0)
        );
        assert!(
            asset
                .meshes
                .iter()
                .any(|mesh| mesh.material.base_color_texture.is_some())
        );
        assert!(
            asset
                .meshes
                .iter()
                .any(|mesh| mesh.material.normal_texture.is_some())
        );
    }
}
