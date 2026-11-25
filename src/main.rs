use std::{collections::HashSet, sync::Arc, time::Instant};

use anyhow::Result;
use photon3d::{camera::Camera, renderer::Renderer, scene::Scene};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{DeviceEvent, ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

struct AppState {
    window: Arc<Window>,
    renderer: Renderer,
    scene: Scene,
    camera: Camera,
    keys: HashSet<KeyCode>,
    mouse_captured: bool,
    last_frame: Instant,
    fps_time: f32,
    fps_frames: u32,
}

impl AppState {
    async fn new(
        display: winit::event_loop::OwnedDisplayHandle,
        window: Arc<Window>,
    ) -> Result<Self> {
        let renderer = Renderer::new(display, window.clone()).await?;
        let size = window.inner_size();
        let camera = Camera {
            aspect_ratio: size.width.max(1) as f32 / size.height.max(1) as f32,
            ..Camera::default()
        };
        Ok(Self {
            window,
            renderer,
            scene: Scene::demo(),
            camera,
            keys: HashSet::new(),
            mouse_captured: false,
            last_frame: Instant::now(),
            fps_time: 0.0,
            fps_frames: 0,
        })
    }

    fn set_mouse_captured(&mut self, captured: bool) {
        self.mouse_captured = if captured {
            self.window
                .set_cursor_grab(CursorGrabMode::Confined)
                .or_else(|_| self.window.set_cursor_grab(CursorGrabMode::Locked))
                .is_ok()
        } else {
            let _ = self.window.set_cursor_grab(CursorGrabMode::None);
            false
        };
        self.window.set_cursor_visible(!self.mouse_captured);
    }

    fn update_camera(&mut self, dt: f32) {
        let mut movement = glam::Vec3::ZERO;
        let forward = self.camera.forward();
        let right = self.camera.right();
        if self.keys.contains(&KeyCode::KeyW) {
            movement += forward;
        }
        if self.keys.contains(&KeyCode::KeyS) {
            movement -= forward;
        }
        if self.keys.contains(&KeyCode::KeyD) {
            movement += right;
        }
        if self.keys.contains(&KeyCode::KeyA) {
            movement -= right;
        }
        if self.keys.contains(&KeyCode::KeyE) || self.keys.contains(&KeyCode::Space) {
            movement += glam::Vec3::Y;
        }
        if self.keys.contains(&KeyCode::KeyQ) || self.keys.contains(&KeyCode::ControlLeft) {
            movement -= glam::Vec3::Y;
        }
        if movement.length_squared() > 0.0 {
            let speed = if self.keys.contains(&KeyCode::ShiftLeft) {
                22.0
            } else {
                7.5
            };
            self.camera.position += movement.normalize() * speed * dt;
        }
    }

    fn frame(&mut self) {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;
        self.update_camera(dt);
        self.scene.update(dt);
        let stats = self.renderer.render(&self.scene, &self.camera);

        self.fps_time += dt;
        self.fps_frames += 1;
        if self.fps_time >= 0.5 {
            let fps = self.fps_frames as f32 / self.fps_time;
            self.window.set_title(&format!(
                "Photon3D — {:.0} fps | {} / {} instances | CPU submit {:.2} ms",
                fps,
                stats.visible_instances,
                stats.total_instances,
                stats.frame_microseconds as f32 / 1000.0,
            ));
            self.fps_time = 0.0;
            self.fps_frames = 0;
        }
    }
}

#[derive(Default)]
struct App {
    state: Option<AppState>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let attributes = Window::default_attributes()
            .with_title("Photon3D")
            .with_inner_size(LogicalSize::new(1440.0, 810.0));
        let window = Arc::new(event_loop.create_window(attributes).expect("create window"));
        let state = pollster::block_on(AppState::new(
            event_loop.owned_display_handle(),
            window.clone(),
        ))
        .expect("initialize Photon3D");
        self.state = Some(state);
        window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                state.camera.aspect_ratio = size.width.max(1) as f32 / size.height.max(1) as f32;
                state.renderer.resize(size.width.max(1), size.height.max(1));
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    match event.state {
                        ElementState::Pressed => {
                            state.keys.insert(code);
                            if code == KeyCode::Escape {
                                state.set_mouse_captured(false);
                            }
                            if code == KeyCode::KeyR {
                                state.camera = Camera::default();
                                let size = state.window.inner_size();
                                state.camera.aspect_ratio =
                                    size.width.max(1) as f32 / size.height.max(1) as f32;
                            }
                        }
                        ElementState::Released => {
                            state.keys.remove(&code);
                        }
                    }
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                state.set_mouse_captured(true);
            }
            WindowEvent::RedrawRequested => {
                state.frame();
                state.window.request_redraw();
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        if state.mouse_captured {
            if let DeviceEvent::MouseMotion { delta } = event {
                state
                    .camera
                    .rotate(delta.0 as f32 * 0.0022, -delta.1 as f32 * 0.0022);
            }
        }
    }
}

fn main() -> Result<()> {
    env_logger::init();
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::default();
    event_loop.run_app(&mut app)?;
    Ok(())
}
