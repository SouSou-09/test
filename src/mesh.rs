use glam::{Vec2, Vec3};

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub tangent: [f32; 4],
}

impl Vertex {
    pub fn new(position: Vec3, normal: Vec3, uv: Vec2, tangent_sign: f32) -> Self {
        let tangent = normal.cross(Vec3::Y).normalize_or_zero();
        let tangent = if tangent.length_squared() < 0.01 {
            Vec3::X
        } else {
            tangent
        };
        Self {
            position: position.to_array(),
            normal: normal.normalize().to_array(),
            uv: uv.to_array(),
            tangent: [tangent.x, tangent.y, tangent.z, tangent_sign],
        }
    }
}

#[derive(Clone, Debug)]
pub struct MeshData {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub bounding_radius: f32,
}

impl MeshData {
    pub fn finish(mut self) -> Self {
        self.bounding_radius = self
            .vertices
            .iter()
            .map(|v| Vec3::from_array(v.position).length())
            .fold(0.0_f32, f32::max);
        self
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum MeshKind {
    Cube,
    Sphere,
    Plane,
    Torus,
    Gltf(u32),
}

impl MeshKind {
    pub const ALL: [Self; 4] = [Self::Cube, Self::Sphere, Self::Plane, Self::Torus];

    pub fn generate(self) -> MeshData {
        match self {
            Self::Cube => cube(),
            Self::Sphere => uv_sphere(48, 24),
            Self::Plane => plane(24, 24, 1.0),
            Self::Torus => torus(48, 16, 0.72, 0.25),
            Self::Gltf(_) => unreachable!("glTF meshes are loaded from assets"),
        }
    }
}

pub fn cube() -> MeshData {
    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    let faces = [
        (Vec3::X, Vec3::Y, Vec3::Z),
        (-Vec3::X, -Vec3::Y, Vec3::Z),
        (Vec3::Y, Vec3::Z, Vec3::X),
        (-Vec3::Y, -Vec3::Z, Vec3::X),
        (Vec3::Z, -Vec3::X, Vec3::Y),
        (-Vec3::Z, Vec3::X, Vec3::Y),
    ];
    for (normal, u_axis, v_axis) in faces {
        let base = vertices.len() as u32;
        for (u, v) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            let position = normal + u_axis * u + v_axis * v;
            vertices.push(Vertex::new(
                position,
                normal,
                Vec2::new((u + 1.0) * 0.5, (v + 1.0) * 0.5),
                1.0,
            ));
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    MeshData {
        vertices,
        indices,
        bounding_radius: 0.0,
    }
    .finish()
}

pub fn plane(width: usize, depth: usize, size: f32) -> MeshData {
    let mut vertices = Vec::with_capacity((width + 1) * (depth + 1));
    let mut indices = Vec::with_capacity(width * depth * 6);
    for z in 0..=depth {
        for x in 0..=width {
            let u = x as f32 / width as f32;
            let v = z as f32 / depth as f32;
            vertices.push(Vertex::new(
                Vec3::new((u - 0.5) * size, 0.0, (v - 0.5) * size),
                Vec3::Y,
                Vec2::new(u, v),
                1.0,
            ));
        }
    }
    for z in 0..depth {
        for x in 0..width {
            let a = (z * (width + 1) + x) as u32;
            let b = a + 1;
            let c = a + width as u32 + 1;
            let d = c + 1;
            indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
    MeshData {
        vertices,
        indices,
        bounding_radius: 0.0,
    }
    .finish()
}

pub fn uv_sphere(segments: usize, rings: usize) -> MeshData {
    let mut vertices = Vec::with_capacity((segments + 1) * (rings + 1));
    let mut indices = Vec::with_capacity(segments * rings * 6);
    for ring in 0..=rings {
        let v = ring as f32 / rings as f32;
        let phi = v * std::f32::consts::PI;
        let (sin_phi, cos_phi) = phi.sin_cos();
        for segment in 0..=segments {
            let u = segment as f32 / segments as f32;
            let theta = u * std::f32::consts::TAU;
            let (sin_theta, cos_theta) = theta.sin_cos();
            let normal = Vec3::new(cos_theta * sin_phi, cos_phi, sin_theta * sin_phi);
            vertices.push(Vertex::new(normal, normal, Vec2::new(u, v), -1.0));
        }
    }
    for ring in 0..rings {
        for segment in 0..segments {
            let a = (ring * (segments + 1) + segment) as u32;
            let b = a + 1;
            let c = a + segments as u32 + 1;
            let d = c + 1;
            indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
    MeshData {
        vertices,
        indices,
        bounding_radius: 0.0,
    }
    .finish()
}

pub fn torus(segments: usize, sides: usize, major_radius: f32, minor_radius: f32) -> MeshData {
    let mut vertices = Vec::with_capacity((segments + 1) * (sides + 1));
    let mut indices = Vec::with_capacity(segments * sides * 6);
    for segment in 0..=segments {
        let u = segment as f32 / segments as f32;
        let theta = u * std::f32::consts::TAU;
        let (sin_theta, cos_theta) = theta.sin_cos();
        for side in 0..=sides {
            let v = side as f32 / sides as f32;
            let phi = v * std::f32::consts::TAU;
            let (sin_phi, cos_phi) = phi.sin_cos();
            let center = Vec3::new(cos_theta * major_radius, 0.0, sin_theta * major_radius);
            let outward = Vec3::new(cos_theta, 0.0, sin_theta);
            let normal = (outward * cos_phi + Vec3::Y * sin_phi).normalize();
            let position = center + normal * minor_radius;
            vertices.push(Vertex::new(position, normal, Vec2::new(u * 4.0, v), 1.0));
        }
    }
    for segment in 0..segments {
        for side in 0..sides {
            let a = (segment * (sides + 1) + side) as u32;
            let b = a + 1;
            let c = a + sides as u32 + 1;
            let d = c + 1;
            indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
    MeshData {
        vertices,
        indices,
        bounding_radius: 0.0,
    }
    .finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_meshes_have_valid_indices_and_unit_normals() {
        for kind in MeshKind::ALL {
            let mesh = kind.generate();
            assert!(!mesh.vertices.is_empty());
            assert!(!mesh.indices.is_empty());
            assert!(
                mesh.indices
                    .iter()
                    .all(|&i| (i as usize) < mesh.vertices.len())
            );
            assert!(mesh.vertices.iter().all(|v| {
                let length = Vec3::from_array(v.normal).length();
                (length - 1.0).abs() < 0.01
            }));
            assert!(mesh.bounding_radius > 0.0);
        }
    }
}
