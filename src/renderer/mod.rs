pub mod uniforms;

use std::{collections::HashMap, mem::size_of, sync::Arc, time::Instant};

use anyhow::{Context, Result};
use glam::Mat4;
use wgpu::util::DeviceExt;
use winit::{event_loop::OwnedDisplayHandle, window::Window};

use crate::{
    camera::Camera,
    mesh::{MeshKind, Vertex},
    scene::Scene,
};
use uniforms::{FrameUniform, InstanceUniform, PostUniform};

const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const NORMAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const VIEW_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const AO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const SHADOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const SHADOW_SIZE: u32 = 2048;
const MAX_INSTANCES: usize = 4096;

#[allow(dead_code)] // texture/instance handles are retained to keep dependent GPU resources alive
pub struct Renderer {
    instance: wgpu::Instance,
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    surface_view_format: wgpu::TextureFormat,
    meshes: HashMap<MeshKind, GpuMesh>,
    frame_buffer: wgpu::Buffer,
    instance_buffer: wgpu::Buffer,
    post_bright_buffer: wgpu::Buffer,
    post_horizontal_buffer: wgpu::Buffer,
    post_vertical_buffer: wgpu::Buffer,
    frame_bind_group_layout: wgpu::BindGroupLayout,
    shadow_bind_group_layout: wgpu::BindGroupLayout,
    ao_bind_group_layout: wgpu::BindGroupLayout,
    post_bind_group_layout: wgpu::BindGroupLayout,
    composite_bind_group_layout: wgpu::BindGroupLayout,
    bind_groups: BindGroups,
    scene_pipeline: wgpu::RenderPipeline,
    shadow_pipeline: wgpu::RenderPipeline,
    ao_pipeline: wgpu::RenderPipeline,
    bright_pipeline: wgpu::RenderPipeline,
    blur_pipeline: wgpu::RenderPipeline,
    composite_pipeline: wgpu::RenderPipeline,
    textures: RenderTextures,
    shadow_texture: wgpu::Texture,
    shadow_view: wgpu::TextureView,
    filter_sampler: wgpu::Sampler,
    shadow_sampler: wgpu::Sampler,
}

pub struct GpuMesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
}

#[allow(dead_code)] // owning textures beside their views is intentional resource lifetime management
struct RenderTextures {
    hdr: wgpu::Texture,
    hdr_view: wgpu::TextureView,
    normal: wgpu::Texture,
    normal_view: wgpu::TextureView,
    depth: wgpu::Texture,
    depth_view: wgpu::TextureView,
    view_depth: wgpu::Texture,
    view_depth_view: wgpu::TextureView,
    ao: wgpu::Texture,
    ao_view: wgpu::TextureView,
    bloom_a: wgpu::Texture,
    bloom_a_view: wgpu::TextureView,
    bloom_b: wgpu::Texture,
    bloom_b_view: wgpu::TextureView,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RenderStats {
    pub total_instances: usize,
    pub visible_instances: usize,
    pub batch_count: usize,
    pub frame_microseconds: u128,
}

struct Batch {
    mesh: MeshKind,
    start: u32,
    end: u32,
}

struct BindGroups {
    frame: wgpu::BindGroup,
    shadow: wgpu::BindGroup,
    ao: wgpu::BindGroup,
    bright: wgpu::BindGroup,
    blur_horizontal: wgpu::BindGroup,
    blur_vertical: wgpu::BindGroup,
    composite: wgpu::BindGroup,
}

impl BindGroups {
    #[allow(clippy::too_many_arguments)]
    fn new(
        device: &wgpu::Device,
        frame_layout: &wgpu::BindGroupLayout,
        shadow_layout: &wgpu::BindGroupLayout,
        ao_layout: &wgpu::BindGroupLayout,
        post_layout: &wgpu::BindGroupLayout,
        composite_layout: &wgpu::BindGroupLayout,
        frame_buffer: &wgpu::Buffer,
        post_bright_buffer: &wgpu::Buffer,
        post_horizontal_buffer: &wgpu::Buffer,
        post_vertical_buffer: &wgpu::Buffer,
        textures: &RenderTextures,
        shadow_view: &wgpu::TextureView,
        filter_sampler: &wgpu::Sampler,
        shadow_sampler: &wgpu::Sampler,
    ) -> Self {
        let frame = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frame bindings"),
            layout: frame_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(shadow_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(shadow_sampler),
                },
            ],
        });
        let shadow = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow frame bindings"),
            layout: shadow_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: frame_buffer.as_entire_binding(),
            }],
        });
        let ao = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ao bindings"),
            layout: ao_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&textures.view_depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&textures.normal_view),
                },
            ],
        });
        let post_group = |label: &str, input: &wgpu::TextureView, uniform_buffer: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: post_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(input),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(filter_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                ],
            })
        };
        let bright = post_group("bright input", &textures.hdr_view, post_bright_buffer);
        let blur_horizontal = post_group(
            "blur horizontal input",
            &textures.bloom_a_view,
            post_horizontal_buffer,
        );
        let blur_vertical = post_group(
            "blur vertical input",
            &textures.bloom_b_view,
            post_vertical_buffer,
        );
        let composite = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("composite bindings"),
            layout: composite_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&textures.hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&textures.ao_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&textures.bloom_a_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(filter_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: frame_buffer.as_entire_binding(),
                },
            ],
        });
        Self {
            frame,
            shadow,
            ao,
            bright,
            blur_horizontal,
            blur_vertical,
            composite,
        }
    }
}

impl Renderer {
    pub async fn new(display: OwnedDisplayHandle, window: Arc<Window>) -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_with_display_handle(
            Box::new(display),
        ));
        let surface = instance
            .create_surface(window.clone())
            .context("create surface")?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .context("request GPU adapter")?;
        log::info!(
            "graphics adapter: {} ({:?})",
            adapter.get_info().name,
            adapter.get_info().backend
        );
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .context("request GPU device")?;

        let size = window.inner_size();
        let capabilities = surface.get_capabilities(&adapter);
        let format = capabilities
            .formats
            .iter()
            .copied()
            .find(wgpu::TextureFormat::is_srgb)
            .unwrap_or(capabilities.formats[0]);
        let surface_view_format = format.add_srgb_suffix();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![surface_view_format],
        };
        surface.configure(&device, &config);

        let meshes = MeshKind::ALL
            .into_iter()
            .map(|kind| (kind, GpuMesh::new(&device, kind)))
            .collect();

        let frame_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame uniform"),
            size: size_of::<FrameUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instance stream"),
            size: (size_of::<InstanceUniform>() * MAX_INSTANCES) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let create_post_buffer = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size_of::<PostUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let post_bright_buffer = create_post_buffer("bright uniform");
        let post_horizontal_buffer = create_post_buffer("horizontal blur uniform");
        let post_vertical_buffer = create_post_buffer("vertical blur uniform");

        let filter_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("linear clamp sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow comparison sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        let shadow_texture = create_texture(
            &device,
            SHADOW_SIZE,
            SHADOW_SIZE,
            SHADOW_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            "shadow map",
        );
        let shadow_view = shadow_texture.create_view(&Default::default());

        let frame_bind_group_layout = create_frame_bind_group_layout(&device);
        let shadow_bind_group_layout = create_shadow_bind_group_layout(&device);
        let ao_bind_group_layout = create_ao_bind_group_layout(&device);
        let post_bind_group_layout = create_post_bind_group_layout(&device);
        let composite_bind_group_layout = create_composite_bind_group_layout(&device);

        let scene_pipeline = create_scene_pipeline(&device, &frame_bind_group_layout);
        let shadow_pipeline = create_shadow_pipeline(&device, &shadow_bind_group_layout);
        let ao_pipeline = create_ao_pipeline(&device, &ao_bind_group_layout);
        let bright_pipeline =
            create_post_pipeline(&device, &post_bind_group_layout, "fs_bright", HDR_FORMAT);
        let blur_pipeline =
            create_post_pipeline(&device, &post_bind_group_layout, "fs_blur", HDR_FORMAT);
        let composite_pipeline =
            create_composite_pipeline(&device, &composite_bind_group_layout, surface_view_format);

        let textures = RenderTextures::new(&device, config.width, config.height);
        let bind_groups = BindGroups::new(
            &device,
            &frame_bind_group_layout,
            &shadow_bind_group_layout,
            &ao_bind_group_layout,
            &post_bind_group_layout,
            &composite_bind_group_layout,
            &frame_buffer,
            &post_bright_buffer,
            &post_horizontal_buffer,
            &post_vertical_buffer,
            &textures,
            &shadow_view,
            &filter_sampler,
            &shadow_sampler,
        );

        let renderer = Self {
            instance,
            window,
            surface,
            device,
            queue,
            config,
            surface_view_format,
            meshes,
            frame_buffer,
            instance_buffer,
            post_bright_buffer,
            post_horizontal_buffer,
            post_vertical_buffer,
            frame_bind_group_layout,
            shadow_bind_group_layout,
            ao_bind_group_layout,
            post_bind_group_layout,
            composite_bind_group_layout,
            bind_groups,
            scene_pipeline,
            shadow_pipeline,
            ao_pipeline,
            bright_pipeline,
            blur_pipeline,
            composite_pipeline,
            textures,
            shadow_texture,
            shadow_view,
            filter_sampler,
            shadow_sampler,
        };
        Ok(renderer)
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 || (width == self.config.width && height == self.config.height)
        {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.textures = RenderTextures::new(&self.device, width, height);
        self.recreate_bind_groups();
    }

    pub fn render(&mut self, scene: &Scene, camera: &Camera) -> RenderStats {
        let frame_start = Instant::now();
        let light_view_proj = directional_light_matrix(scene);
        let frame = FrameUniform::new(
            camera,
            scene,
            self.config.width,
            self.config.height,
            light_view_proj,
        );
        self.queue
            .write_buffer(&self.frame_buffer, 0, bytemuck::bytes_of(&frame));

        let (instances, batches) = build_instances(scene, camera);
        if !instances.is_empty() {
            self.queue.write_buffer(
                &self.instance_buffer,
                0,
                bytemuck::cast_slice(&instances[..instances.len().min(MAX_INSTANCES)]),
            );
        }

        let surface_texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture) => texture,
            wgpu::CurrentSurfaceTexture::Occluded | wgpu::CurrentSurfaceTexture::Timeout => {
                return RenderStats::default();
            }
            wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.config);
                return RenderStats::default();
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                if let Ok(surface) = self.instance.create_surface(self.window.clone()) {
                    self.surface = surface;
                    self.surface.configure(&self.device, &self.config);
                }
                return RenderStats::default();
            }
            wgpu::CurrentSurfaceTexture::Validation => return RenderStats::default(),
        };
        let surface_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor {
                format: Some(self.surface_view_format),
                ..Default::default()
            });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("photon frame"),
            });

        self.shadow_pass(&mut encoder, &batches);
        self.scene_pass(&mut encoder, &batches);
        self.ao_pass(&mut encoder);
        self.bloom_passes(&mut encoder);
        self.composite_pass(&mut encoder, &surface_view);

        self.queue.submit([encoder.finish()]);
        self.window.pre_present_notify();
        self.queue.present(surface_texture);

        RenderStats {
            total_instances: scene.objects.len(),
            visible_instances: instances.len(),
            batch_count: batches.len(),
            frame_microseconds: frame_start.elapsed().as_micros(),
        }
    }

    fn recreate_bind_groups(&mut self) {
        self.bind_groups = BindGroups::new(
            &self.device,
            &self.frame_bind_group_layout,
            &self.shadow_bind_group_layout,
            &self.ao_bind_group_layout,
            &self.post_bind_group_layout,
            &self.composite_bind_group_layout,
            &self.frame_buffer,
            &self.post_bright_buffer,
            &self.post_horizontal_buffer,
            &self.post_vertical_buffer,
            &self.textures,
            &self.shadow_view,
            &self.filter_sampler,
            &self.shadow_sampler,
        );
    }

    fn shadow_pass(&self, encoder: &mut wgpu::CommandEncoder, batches: &[Batch]) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("directional shadow pass"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.shadow_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.shadow_pipeline);
        pass.set_bind_group(0, &self.bind_groups.shadow, &[]);
        self.draw_batches(&mut pass, batches);
    }

    fn scene_pass(&self, encoder: &mut wgpu::CommandEncoder, batches: &[Batch]) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("pbr scene pass"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.textures.hdr_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.textures.normal_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.textures.view_depth_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.textures.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.scene_pipeline);
        pass.set_bind_group(0, &self.bind_groups.frame, &[]);
        self.draw_batches(&mut pass, batches);
    }

    fn draw_batches<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>, batches: &[Batch]) {
        pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        for batch in batches {
            let Some(mesh) = self.meshes.get(&batch.mesh) else {
                continue;
            };
            pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..mesh.index_count, 0, batch.start..batch.end);
        }
    }

    fn ao_pass(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("ssao pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.textures.ao_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.ao_pipeline);
        pass.set_bind_group(0, &self.bind_groups.ao, &[]);
        pass.draw(0..3, 0..1);
    }

    fn bloom_passes(&self, encoder: &mut wgpu::CommandEncoder) {
        self.write_post_uniforms();
        full_screen_pass(
            encoder,
            &self.bright_pipeline,
            &self.bind_groups.bright,
            &self.textures.bloom_a_view,
            "bloom bright pass",
        );
        full_screen_pass(
            encoder,
            &self.blur_pipeline,
            &self.bind_groups.blur_horizontal,
            &self.textures.bloom_b_view,
            "bloom horizontal blur",
        );
        full_screen_pass(
            encoder,
            &self.blur_pipeline,
            &self.bind_groups.blur_vertical,
            &self.textures.bloom_a_view,
            "bloom vertical blur",
        );
    }

    fn composite_pass(&self, encoder: &mut wgpu::CommandEncoder, surface_view: &wgpu::TextureView) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("aces composite pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: surface_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.composite_pipeline);
        pass.set_bind_group(0, &self.bind_groups.composite, &[]);
        pass.draw(0..3, 0..1);
    }

    fn write_post_uniforms(&self) {
        let texel = [
            1.0 / self.textures.bloom_width(),
            1.0 / self.textures.bloom_height(),
            0.0,
            0.0,
        ];
        let bright = PostUniform {
            direction_threshold: [0.0, 0.0, 2.05, 0.5],
            texel_size: [0.0; 4],
        };
        let horizontal = PostUniform {
            direction_threshold: [1.0, 0.0, 0.0, 0.0],
            texel_size: texel,
        };
        let vertical = PostUniform {
            direction_threshold: [0.0, 1.0, 0.0, 0.0],
            texel_size: texel,
        };
        self.queue
            .write_buffer(&self.post_bright_buffer, 0, bytemuck::bytes_of(&bright));
        self.queue.write_buffer(
            &self.post_horizontal_buffer,
            0,
            bytemuck::bytes_of(&horizontal),
        );
        self.queue
            .write_buffer(&self.post_vertical_buffer, 0, bytemuck::bytes_of(&vertical));
    }
}

impl GpuMesh {
    fn new(device: &wgpu::Device, kind: MeshKind) -> Self {
        let mesh = kind.generate();
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh vertices"),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh indices"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        Self {
            vertex_buffer,
            index_buffer,
            index_count: mesh.indices.len() as u32,
        }
    }
}

impl RenderTextures {
    fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let hdr = create_texture(
            device,
            width,
            height,
            HDR_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            "hdr scene",
        );
        let normal = create_texture(
            device,
            width,
            height,
            NORMAL_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            "view normals",
        );
        let depth = create_texture(
            device,
            width,
            height,
            DEPTH_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            "scene depth",
        );
        let view_depth = create_texture(
            device,
            width,
            height,
            VIEW_DEPTH_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            "view depth",
        );
        let ao = create_texture(
            device,
            width,
            height,
            AO_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            "ssao",
        );
        let bloom_width = (width / 4).max(1);
        let bloom_height = (height / 4).max(1);
        let bloom_a = create_texture(
            device,
            bloom_width,
            bloom_height,
            HDR_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            "bloom a",
        );
        let bloom_b = create_texture(
            device,
            bloom_width,
            bloom_height,
            HDR_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            "bloom b",
        );
        Self {
            hdr_view: hdr.create_view(&Default::default()),
            hdr,
            normal_view: normal.create_view(&Default::default()),
            normal,
            depth_view: depth.create_view(&Default::default()),
            depth,
            view_depth_view: view_depth.create_view(&Default::default()),
            view_depth,
            ao_view: ao.create_view(&Default::default()),
            ao,
            bloom_a_view: bloom_a.create_view(&Default::default()),
            bloom_a,
            bloom_b_view: bloom_b.create_view(&Default::default()),
            bloom_b,
        }
    }

    fn bloom_width(&self) -> f32 {
        self.bloom_a.width() as f32
    }

    fn bloom_height(&self) -> f32 {
        self.bloom_a.height() as f32
    }
}

fn create_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
    label: &str,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

fn directional_light_matrix(scene: &Scene) -> Mat4 {
    let direction = scene.sun_direction.normalize();
    let eye = -direction * 48.0;
    let view = Mat4::look_to_rh(eye, direction, glam::Vec3::Y);
    Mat4::orthographic_rh(-38.0, 38.0, -38.0, 38.0, 0.1, 110.0) * view
}

fn build_instances(scene: &Scene, camera: &Camera) -> (Vec<InstanceUniform>, Vec<Batch>) {
    let visible = scene.visible_indices(&camera.frustum());
    let mut instances = Vec::with_capacity(visible.len().min(MAX_INSTANCES));
    let mut batches = Vec::new();
    let mut active_mesh: Option<MeshKind> = None;
    let mut batch_start = 0_u32;

    for index in visible.into_iter().take(MAX_INSTANCES) {
        let object = &scene.objects[index];
        if let Some(mesh) = active_mesh {
            if mesh != object.mesh {
                let end = instances.len() as u32;
                if end > batch_start {
                    batches.push(Batch {
                        mesh,
                        start: batch_start,
                        end,
                    });
                }
                batch_start = end;
                active_mesh = Some(object.mesh);
            }
        } else {
            active_mesh = Some(object.mesh);
        }
        instances.push(InstanceUniform::new(
            object.transform.matrix(),
            object.material,
        ));
    }
    if let Some(mesh) = active_mesh {
        let end = instances.len() as u32;
        if end > batch_start {
            batches.push(Batch {
                mesh,
                start: batch_start,
                end,
            });
        }
    }
    (instances, batches)
}

fn create_shadow_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("shadow frame layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

fn create_frame_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("frame bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                count: None,
            },
        ],
    })
}

fn create_ao_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ao bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    })
}

fn create_post_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("post input layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

fn create_composite_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let texture = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("composite layout"),
        entries: &[
            texture(0),
            texture(1),
            texture(2),
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

fn vertex_attributes() -> [wgpu::VertexAttribute; 4] {
    wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x3,
        2 => Float32x2,
        3 => Float32x4,
    ]
}

fn instance_attributes() -> [wgpu::VertexAttribute; 11] {
    wgpu::vertex_attr_array![
        4 => Float32x4,
        5 => Float32x4,
        6 => Float32x4,
        7 => Float32x4,
        8 => Float32x4,
        9 => Float32x4,
        10 => Float32x4,
        11 => Float32x4,
        12 => Float32x4,
        13 => Float32x4,
        14 => Float32x4,
    ]
}

fn vertex_buffer_layouts<'a>(
    vertex_attributes: &'a [wgpu::VertexAttribute],
    instance_attributes: &'a [wgpu::VertexAttribute],
) -> [Option<wgpu::VertexBufferLayout<'a>>; 2] {
    [
        Some(wgpu::VertexBufferLayout {
            array_stride: size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: vertex_attributes,
        }),
        Some(wgpu::VertexBufferLayout {
            array_stride: size_of::<InstanceUniform>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: instance_attributes,
        }),
    ]
}

fn create_pipeline_layout(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    label: &str,
) -> wgpu::PipelineLayout {
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    })
}

fn create_scene_pipeline(
    device: &wgpu::Device,
    frame_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/scene.wgsl"));
    let layout = create_pipeline_layout(device, frame_layout, "scene pipeline layout");
    let vertex_attrs = vertex_attributes();
    let instance_attrs = instance_attributes();
    let buffers = vertex_buffer_layouts(&vertex_attrs, &instance_attrs);
    let targets = [
        Some(wgpu::ColorTargetState {
            format: HDR_FORMAT,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        }),
        Some(wgpu::ColorTargetState {
            format: NORMAL_FORMAT,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        }),
        Some(wgpu::ColorTargetState {
            format: VIEW_DEPTH_FORMAT,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        }),
    ];
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("cook-torrance scene pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &buffers,
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &targets,
        }),
        multiview_mask: None,
        cache: None,
    })
}

fn create_shadow_pipeline(
    device: &wgpu::Device,
    frame_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/shadow.wgsl"));
    let layout = create_pipeline_layout(device, frame_layout, "shadow pipeline layout");
    let vertex_attrs = vertex_attributes();
    let instance_attrs = instance_attributes();
    let buffers = vertex_buffer_layouts(&vertex_attrs, &instance_attrs);
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("shadow depth pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_shadow"),
            compilation_options: Default::default(),
            buffers: &buffers,
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: SHADOW_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: Default::default(),
            bias: wgpu::DepthBiasState {
                constant: 2,
                slope_scale: 1.75,
                clamp: 0.0,
            },
        }),
        multisample: Default::default(),
        fragment: None,
        multiview_mask: None,
        cache: None,
    })
}

fn fullscreen_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    fragment_entry: &str,
    target_format: wgpu::TextureFormat,
    shader: &wgpu::ShaderModule,
    label: &str,
) -> wgpu::RenderPipeline {
    let pipeline_layout = create_pipeline_layout(device, layout, label);
    let targets = [Some(wgpu::ColorTargetState {
        format: target_format,
        blend: None,
        write_mask: wgpu::ColorWrites::ALL,
    })];
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_fullscreen"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(fragment_entry),
            compilation_options: Default::default(),
            targets: &targets,
        }),
        multiview_mask: None,
        cache: None,
    })
}

fn create_ao_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/ao.wgsl"));
    fullscreen_pipeline(device, layout, "fs_ao", AO_FORMAT, &shader, "ssao pipeline")
}

fn create_post_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    fragment_entry: &str,
    target_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/post.wgsl"));
    fullscreen_pipeline(
        device,
        layout,
        fragment_entry,
        target_format,
        &shader,
        fragment_entry,
    )
}

fn create_composite_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    target_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/post.wgsl"));
    fullscreen_pipeline(
        device,
        layout,
        "fs_composite",
        target_format,
        &shader,
        "composite pipeline",
    )
}

fn full_screen_pass(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    target: &wgpu::TextureView,
    label: &str,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.draw(0..3, 0..1);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::Scene;

    #[test]
    fn instance_buffer_layout_is_std430_friendly() {
        assert_eq!(size_of::<InstanceUniform>(), 176);
        assert_eq!(size_of::<Vertex>(), 48);
    }

    #[test]
    fn batches_are_contiguous_and_grouped_by_mesh() {
        let scene = Scene::demo();
        let camera = Camera::default();
        let (instances, batches) = build_instances(&scene, &camera);
        assert!(!instances.is_empty());
        assert!(!batches.is_empty());
        assert!(batches.iter().all(|batch| batch.end > batch.start));
        assert!(batches.windows(2).all(|pair| pair[0].end <= pair[1].start));
    }
}
