use glam::{Mat4, Vec3, Vec4};

#[derive(Debug, Clone)]
pub struct Camera {
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y_radians: f32,
    pub aspect_ratio: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Vec3::new(9.0, 5.5, 11.0),
            yaw: -2.42,
            pitch: -0.28,
            fov_y_radians: 58.0_f32.to_radians(),
            aspect_ratio: 16.0 / 9.0,
            near: 0.08,
            far: 180.0,
        }
    }
}

impl Camera {
    pub fn forward(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(cy * cp, sp, sy * cp).normalize()
    }

    pub fn right(&self) -> Vec3 {
        self.forward().cross(Vec3::Y).normalize_or_zero()
    }

    pub fn up(&self) -> Vec3 {
        self.right().cross(self.forward()).normalize_or_zero()
    }

    pub fn rotate(&mut self, delta_yaw: f32, delta_pitch: f32) {
        self.yaw += delta_yaw;
        self.pitch = (self.pitch + delta_pitch).clamp(-1.53, 1.53);
    }

    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_to_rh(self.position, self.forward(), Vec3::Y)
    }

    pub fn projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov_y_radians, self.aspect_ratio, self.near, self.far)
    }

    pub fn view_projection_matrix(&self) -> Mat4 {
        self.projection_matrix() * self.view_matrix()
    }

    pub fn frustum(&self) -> Frustum {
        Frustum::from_matrix(self.view_projection_matrix())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Frustum {
    planes: [Vec4; 6],
}

impl Frustum {
    pub fn from_matrix(matrix: Mat4) -> Self {
        let rows = matrix.transpose().to_cols_array_2d();
        let row = |i: usize| Vec4::new(rows[i][0], rows[i][1], rows[i][2], rows[i][3]);
        let r0 = row(0);
        let r1 = row(1);
        let r2 = row(2);
        let r3 = row(3);
        let mut planes = [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r3 + r2, r3 - r2];
        for plane in &mut planes {
            let length = plane.truncate().length().max(1.0e-6);
            *plane /= length;
        }
        Self { planes }
    }

    pub fn intersects_sphere(&self, center: Vec3, radius: f32) -> bool {
        self.planes
            .iter()
            .all(|plane| plane.truncate().dot(center) + plane.w >= -radius)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frustum_contains_point_in_front_of_camera() {
        let camera = Camera::default();
        let target = camera.position + camera.forward() * 10.0;
        assert!(camera.frustum().intersects_sphere(target, 0.1));
    }

    #[test]
    fn frustum_rejects_point_behind_camera() {
        let camera = Camera::default();
        let target = camera.position - camera.forward() * 10.0;
        assert!(!camera.frustum().intersects_sphere(target, 0.1));
    }
}
