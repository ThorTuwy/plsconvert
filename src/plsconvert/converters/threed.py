from pathlib import Path
from plsconvert.converters.abstract import Converter
from plsconvert.utils.graph import ConversionAdj, conversionFromToAdj
from plsconvert.converters.registry import register_converter
from plsconvert.utils.dependency import Dependencies, LibDependency as Lib

@register_converter
class threeDConverter(Converter):
    """
    Converter for 3D models.
    """

    @property
    def name(self) -> str:
        return "3D Converter"
    
    @property
    def dependencies(self) -> Dependencies:
        return Dependencies([Lib("trimesh"), Lib("moderngl"), Lib("pyrr"), Lib("PIL"), Lib("imageio"), Lib("imageio_ffmpeg"), Lib("pygltflib")])

    def adjConverter(self) -> ConversionAdj:
        model_conversion = conversionFromToAdj(
            ["glb", "gltf", "obj"],
            ["glb", "gltf", "obj"]
        )
        video_conversion = conversionFromToAdj(
            ["glb", "gltf", "obj"],
            ["mp4"]
        )
        image_conversion = conversionFromToAdj(
            ["glb", "gltf", "obj"],
            ["png"]
        )
        gif_conversion = conversionFromToAdj(
            ["glb", "gltf", "obj"],
            ["gif"]
        )
        return model_conversion + video_conversion + image_conversion + gif_conversion

    def convert(
        self, input: Path, output: Path, input_extension: str, output_extension: str
    ) -> None:
        if output_extension == "mp4":
            self._create_spinning_video(input, output, input_extension)
        elif output_extension == "png":
            self._create_png_render(input, output, input_extension)
        elif output_extension == "gif":
            self._create_spinning_gif(input, output, input_extension)
        else:
            self._convert_3d_format(input, output, input_extension, output_extension)

    def _convert_3d_format(self, input: Path, output: Path, input_ext: str, output_ext: str) -> None:
        import trimesh
        
        mesh = trimesh.load(str(input))
        
        if hasattr(mesh, 'dump'):
            try:
                mesh = mesh.dump(concatenate=True)
            except:
                if hasattr(mesh, 'geometry') and mesh.geometry:
                    mesh = list(mesh.geometry.values())[0]
        
        if output_ext.lower() == "obj":
            mesh.export(str(output))
        else:
            exported_data = mesh.export(file_type=output_ext.lower())
            with open(output, 'wb') as f:
                f.write(exported_data)

    def _create_png_render(self, input: Path, output: Path, input_ext: str) -> None:
        import trimesh
        import moderngl
        import numpy as np
        from PIL import Image
        import pyrr
        
        scene_or_mesh = trimesh.load(str(input))
        
        if hasattr(scene_or_mesh, 'geometry') and scene_or_mesh.geometry:
            geometries = list(scene_or_mesh.geometry.values())
            if geometries:
                mesh = geometries[0]
            else:
                mesh = scene_or_mesh.dump(concatenate=True)
        else:
            mesh = scene_or_mesh
        
        vertices = mesh.vertices.astype(np.float32)
        faces = mesh.faces.astype(np.uint32)
        
        if hasattr(mesh, 'vertex_normals'):
            normals = mesh.vertex_normals.astype(np.float32)
        else:
            mesh.fix_normals()
            normals = mesh.vertex_normals.astype(np.float32)
        
        # Get UV coordinates (same as MP4 method)
        uvs = None
        try:
            if hasattr(scene_or_mesh, 'geometry'):
                for geom in scene_or_mesh.geometry.values():
                    if hasattr(geom.visual, 'uv') and geom.visual.uv is not None:
                        uvs = geom.visual.uv.astype(np.float32)
                        uvs[:, 1] = 1.0 - uvs[:, 1]
                        uvs = np.clip(uvs, 0.0, 1.0)
                        break
            elif hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
                uvs = mesh.visual.uv.astype(np.float32)
                uvs[:, 1] = 1.0 - uvs[:, 1]
                uvs = np.clip(uvs, 0.0, 1.0)
        except Exception:
            pass
        
        if uvs is None:
            uvs = np.zeros((len(vertices), 2), dtype=np.float32)
            x_range = vertices[:, 0].max() - vertices[:, 0].min()
            z_range = vertices[:, 2].max() - vertices[:, 2].min()
            if x_range > 0:
                uvs[:, 0] = (vertices[:, 0] - vertices[:, 0].min()) / x_range
            if z_range > 0:
                uvs[:, 1] = (vertices[:, 2] - vertices[:, 2].min()) / z_range
            uvs = np.clip(uvs, 0.0, 1.0)
        
        # Extract texture (same as MP4 method)
        texture_data = None
        has_texture = False
        
        try:
            if hasattr(scene_or_mesh, 'geometry'):
                for geom in scene_or_mesh.geometry.values():
                    if hasattr(geom.visual, 'material'):
                        material = geom.visual.material
                        if hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
                            texture_data = material.baseColorTexture
                            has_texture = True
                            break
                        elif hasattr(material, 'image') and material.image is not None:
                            texture_data = material.image
                            has_texture = True
                            break
            elif hasattr(mesh.visual, 'material'):
                material = mesh.visual.material
                if hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
                    texture_data = material.baseColorTexture
                    has_texture = True
                elif hasattr(material, 'image') and material.image is not None:
                    texture_data = material.image
                    has_texture = True
        except Exception:
            has_texture = False
        
        # Get vertex colors
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            colors = mesh.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
        else:
            base_color = [1.0, 1.0, 1.0] if has_texture else [0.6, 0.7, 0.8]
            colors = np.ones((len(vertices), 3), dtype=np.float32)
            colors[:, :] = base_color
        
        vertex_data = np.column_stack([vertices, normals, colors, uvs])
        
        # Center and scale model
        center = vertices.mean(axis=0)
        vertices_centered = vertices - center
        scale = 1.0 / np.max(np.linalg.norm(vertices_centered, axis=1))
        
        ctx = moderngl.create_context(standalone=True, require=330)
        
        # Same shaders as MP4 but with alpha support
        vertex_shader = """
        #version 330 core
        
        in vec3 in_position;
        in vec3 in_normal;
        in vec3 in_color;
        in vec2 in_uv;
        
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        uniform mat3 normal_matrix;
        
        out vec3 world_pos;
        out vec3 normal;
        out vec3 color;
        out vec2 uv;
        
        void main() {
            vec4 world_position = model * vec4(in_position, 1.0);
            world_pos = world_position.xyz;
            normal = normalize(normal_matrix * in_normal);
            color = in_color;
            uv = in_uv;
            gl_Position = projection * view * world_position;
        }
        """
        
        fragment_shader = """
        #version 330 core
        
        in vec3 world_pos;
        in vec3 normal;
        in vec3 color;
        in vec2 uv;
        
        uniform vec3 light_pos;
        uniform vec3 light_color;
        uniform vec3 camera_pos;
        uniform float ambient_strength;
        uniform float specular_strength;
        uniform sampler2D texture_sampler;
        uniform bool has_texture;
        
        out vec4 fragColor;
        
        void main() {
            vec2 clamped_uv = clamp(uv, 0.0, 1.0);
            
            vec3 base_color;
            if (has_texture) {
                vec3 texture_color = texture(texture_sampler, clamped_uv).rgb;
                float color_intensity = (color.r + color.g + color.b) / 3.0;
                if (color_intensity > 0.8) {
                    base_color = texture_color;
                } else {
                    base_color = texture_color * color;
                }
            } else {
                base_color = color;
            }
            
            vec3 norm = normalize(normal);
            if (length(norm) < 0.1) {
                norm = vec3(0.0, 1.0, 0.0);
            }
            
            vec3 ambient = ambient_strength * light_color;
            
            vec3 light_dir = normalize(light_pos - world_pos);
            float diff = max(dot(norm, light_dir), 0.0);
            vec3 diffuse = diff * light_color;
            
            vec3 view_dir = normalize(camera_pos - world_pos);
            vec3 reflect_dir = reflect(-light_dir, norm);
            float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 64);
            vec3 specular = specular_strength * spec * light_color;
            
            vec3 result = (ambient + diffuse + specular) * base_color;
            result = clamp(result, 0.0, 1.0);
            
            fragColor = vec4(result, 1.0);
        }
        """
        
        program = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
        
        # Create texture (same as MP4)
        texture = None
        if has_texture and texture_data is not None:
            try:
                if hasattr(texture_data, 'tobytes'):
                    width, height = texture_data.size
                    texture_bytes = texture_data.tobytes('raw', 'RGB')
                else:
                    from PIL import Image
                    if isinstance(texture_data, np.ndarray):
                        texture_pil = Image.fromarray(texture_data)
                    else:
                        texture_pil = texture_data
                    width, height = texture_pil.size
                    texture_bytes = texture_pil.tobytes('raw', 'RGB')
                
                texture = ctx.texture((width, height), 3, texture_bytes)
                texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
                texture.repeat_x = False
                texture.repeat_y = False
                texture.build_mipmaps()
            except Exception:
                has_texture = False
                texture = None
        
        if not has_texture:
            default_texture_data = np.ones((4, 4, 3), dtype=np.uint8) * 255
            texture = ctx.texture((4, 4), 3, default_texture_data.tobytes())
        
        vbo = ctx.buffer(vertex_data.tobytes())
        ibo = ctx.buffer(faces.tobytes())
        vao = ctx.vertex_array(program, [(vbo, '3f 3f 3f 2f', 'in_position', 'in_normal', 'in_color', 'in_uv')], ibo)
        
        width, height = 1024, 1024
        fbo = ctx.framebuffer(
            color_attachments=ctx.texture((width, height), 4),  # RGBA for transparency
            depth_attachment=ctx.depth_texture((width, height))
        )
        
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.CULL_FACE)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        
        projection = pyrr.matrix44.create_perspective_projection_matrix(45.0, 1.0, 0.1, 100.0)
        
        # Single render at nice angle
        camera_distance = 3.0
        camera_pos = np.array([
            camera_distance * 0.707,  # 45 degrees
            camera_distance * 0.5,    # elevated
            camera_distance * 0.707
        ])
        
        view = pyrr.matrix44.create_look_at(
            camera_pos, np.array([0, 0, 0]), np.array([0, 1, 0])
        )
        
        model = np.eye(4, dtype=np.float32)
        model[:3, 3] = -center * scale
        model[0, 0] = scale
        model[1, 1] = scale
        model[2, 2] = scale
        
        normal_matrix = np.linalg.inv(model[:3, :3]).T
        light_pos = camera_pos + np.array([1, 2, 1])
        
        program['model'].write(model.astype(np.float32).tobytes())
        program['view'].write(view.astype(np.float32).tobytes())
        program['projection'].write(projection.astype(np.float32).tobytes())
        program['normal_matrix'].write(normal_matrix.astype(np.float32).tobytes())
        program['light_pos'].value = tuple(light_pos)
        program['light_color'].value = (1.0, 1.0, 1.0)
        program['camera_pos'].value = tuple(camera_pos)
        program['ambient_strength'].value = 0.3
        program['specular_strength'].value = 0.5
        program['has_texture'].value = has_texture
        
        if texture:
            texture.use(location=0)
            program['texture_sampler'].value = 0
        
        fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 0.0)  # Transparent background
        ctx.viewport = (0, 0, width, height)
        vao.render()
        
        # Save as PNG with alpha
        raw_data = fbo.read(components=4)  # RGBA
        image = Image.frombytes('RGBA', (width, height), raw_data).transpose(Image.FLIP_TOP_BOTTOM)
        image.save(str(output))
        
        ctx.release()

    def _create_spinning_gif(self, input: Path, output: Path, input_ext: str) -> None:
        import trimesh
        import moderngl
        import numpy as np
        import imageio
        import tempfile
        from PIL import Image
        import pyrr
        
        scene_or_mesh = trimesh.load(str(input))
        
        if hasattr(scene_or_mesh, 'geometry') and scene_or_mesh.geometry:
            geometries = list(scene_or_mesh.geometry.values())
            if geometries:
                mesh = geometries[0]
            else:
                mesh = scene_or_mesh.dump(concatenate=True)
        else:
            mesh = scene_or_mesh
        
        vertices = mesh.vertices.astype(np.float32)
        faces = mesh.faces.astype(np.uint32)
        
        if hasattr(mesh, 'vertex_normals'):
            normals = mesh.vertex_normals.astype(np.float32)
        else:
            mesh.fix_normals()
            normals = mesh.vertex_normals.astype(np.float32)
        
        # Get UV coordinates (same as other methods)
        uvs = None
        try:
            if hasattr(scene_or_mesh, 'geometry'):
                for geom in scene_or_mesh.geometry.values():
                    if hasattr(geom.visual, 'uv') and geom.visual.uv is not None:
                        uvs = geom.visual.uv.astype(np.float32)
                        uvs[:, 1] = 1.0 - uvs[:, 1]
                        uvs = np.clip(uvs, 0.0, 1.0)
                        break
            elif hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
                uvs = mesh.visual.uv.astype(np.float32)
                uvs[:, 1] = 1.0 - uvs[:, 1]
                uvs = np.clip(uvs, 0.0, 1.0)
        except Exception:
            pass
        
        if uvs is None:
            uvs = np.zeros((len(vertices), 2), dtype=np.float32)
            x_range = vertices[:, 0].max() - vertices[:, 0].min()
            z_range = vertices[:, 2].max() - vertices[:, 2].min()
            if x_range > 0:
                uvs[:, 0] = (vertices[:, 0] - vertices[:, 0].min()) / x_range
            if z_range > 0:
                uvs[:, 1] = (vertices[:, 2] - vertices[:, 2].min()) / z_range
            uvs = np.clip(uvs, 0.0, 1.0)
        
        # Extract texture (same as other methods)
        texture_data = None
        has_texture = False
        
        try:
            if hasattr(scene_or_mesh, 'geometry'):
                for geom in scene_or_mesh.geometry.values():
                    if hasattr(geom.visual, 'material'):
                        material = geom.visual.material
                        if hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
                            texture_data = material.baseColorTexture
                            has_texture = True
                            break
                        elif hasattr(material, 'image') and material.image is not None:
                            texture_data = material.image
                            has_texture = True
                            break
            elif hasattr(mesh.visual, 'material'):
                material = mesh.visual.material
                if hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
                    texture_data = material.baseColorTexture
                    has_texture = True
                elif hasattr(material, 'image') and material.image is not None:
                    texture_data = material.image
                    has_texture = True
        except Exception:
            has_texture = False
        
        # Get vertex colors
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            colors = mesh.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
        else:
            base_color = [1.0, 1.0, 1.0] if has_texture else [0.6, 0.7, 0.8]
            colors = np.ones((len(vertices), 3), dtype=np.float32)
            colors[:, :] = base_color
        
        vertex_data = np.column_stack([vertices, normals, colors, uvs])
        
        # Center and scale model
        center = vertices.mean(axis=0)
        vertices_centered = vertices - center
        scale = 1.0 / np.max(np.linalg.norm(vertices_centered, axis=1))
        
        ctx = moderngl.create_context(standalone=True, require=330)
        
        # Same shaders as other methods
        vertex_shader = """
        #version 330 core
        
        in vec3 in_position;
        in vec3 in_normal;
        in vec3 in_color;
        in vec2 in_uv;
        
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        uniform mat3 normal_matrix;
        
        out vec3 world_pos;
        out vec3 normal;
        out vec3 color;
        out vec2 uv;
        
        void main() {
            vec4 world_position = model * vec4(in_position, 1.0);
            world_pos = world_position.xyz;
            normal = normalize(normal_matrix * in_normal);
            color = in_color;
            uv = in_uv;
            gl_Position = projection * view * world_position;
        }
        """
        
        fragment_shader = """
        #version 330 core
        
        in vec3 world_pos;
        in vec3 normal;
        in vec3 color;
        in vec2 uv;
        
        uniform vec3 light_pos;
        uniform vec3 light_color;
        uniform vec3 camera_pos;
        uniform float ambient_strength;
        uniform float specular_strength;
        uniform sampler2D texture_sampler;
        uniform bool has_texture;
        
        out vec4 fragColor;
        
        void main() {
            vec2 clamped_uv = clamp(uv, 0.0, 1.0);
            
            vec3 base_color;
            if (has_texture) {
                vec3 texture_color = texture(texture_sampler, clamped_uv).rgb;
                float color_intensity = (color.r + color.g + color.b) / 3.0;
                if (color_intensity > 0.8) {
                    base_color = texture_color;
                } else {
                    base_color = texture_color * color;
                }
            } else {
                base_color = color;
            }
            
            vec3 norm = normalize(normal);
            if (length(norm) < 0.1) {
                norm = vec3(0.0, 1.0, 0.0);
            }
            
            vec3 ambient = ambient_strength * light_color;
            
            vec3 light_dir = normalize(light_pos - world_pos);
            float diff = max(dot(norm, light_dir), 0.0);
            vec3 diffuse = diff * light_color;
            
            vec3 view_dir = normalize(camera_pos - world_pos);
            vec3 reflect_dir = reflect(-light_dir, norm);
            float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 64);
            vec3 specular = specular_strength * spec * light_color;
            
            vec3 result = (ambient + diffuse + specular) * base_color;
            result = clamp(result, 0.0, 1.0);
            
            fragColor = vec4(result, 1.0);
        }
        """
        
        program = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
        
        # Create texture (same as other methods)
        texture = None
        if has_texture and texture_data is not None:
            try:
                if hasattr(texture_data, 'tobytes'):
                    width, height = texture_data.size
                    texture_bytes = texture_data.tobytes('raw', 'RGB')
                else:
                    from PIL import Image
                    if isinstance(texture_data, np.ndarray):
                        texture_pil = Image.fromarray(texture_data)
                    else:
                        texture_pil = texture_data
                    width, height = texture_pil.size
                    texture_bytes = texture_pil.tobytes('raw', 'RGB')
                
                texture = ctx.texture((width, height), 3, texture_bytes)
                texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
                texture.repeat_x = False
                texture.repeat_y = False
                texture.build_mipmaps()
            except Exception:
                has_texture = False
                texture = None
        
        if not has_texture:
            default_texture_data = np.ones((4, 4, 3), dtype=np.uint8) * 255
            texture = ctx.texture((4, 4), 3, default_texture_data.tobytes())
        
        vbo = ctx.buffer(vertex_data.tobytes())
        ibo = ctx.buffer(faces.tobytes())
        vao = ctx.vertex_array(program, [(vbo, '3f 3f 3f 2f', 'in_position', 'in_normal', 'in_color', 'in_uv')], ibo)
        
        width, height = 720, 720
        fbo = ctx.framebuffer(
            color_attachments=ctx.texture((width, height), 4),  # RGBA for transparency
            depth_attachment=ctx.depth_texture((width, height))
        )
        
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.CULL_FACE)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        
        projection = pyrr.matrix44.create_perspective_projection_matrix(45.0, 1.0, 0.1, 100.0)
        
        # Create frames for GIF
        gif_frames = []
        
        for i in range(60):  # More frames for smoother animation
            angle = i * (2 * np.pi / 60)
            camera_distance = 3.0
            camera_pos = np.array([
                camera_distance * np.cos(angle),
                camera_distance * 0.5,
                camera_distance * np.sin(angle)
            ])
            
            view = pyrr.matrix44.create_look_at(
                camera_pos, np.array([0, 0, 0]), np.array([0, 1, 0])
            )
            
            model = np.eye(4, dtype=np.float32)
            model[:3, 3] = -center * scale
            model[0, 0] = scale
            model[1, 1] = scale
            model[2, 2] = scale
            
            normal_matrix = np.linalg.inv(model[:3, :3]).T
            light_pos = camera_pos + np.array([1, 2, 1])
            
            program['model'].write(model.astype(np.float32).tobytes())
            program['view'].write(view.astype(np.float32).tobytes())
            program['projection'].write(projection.astype(np.float32).tobytes())
            program['normal_matrix'].write(normal_matrix.astype(np.float32).tobytes())
            program['light_pos'].value = tuple(light_pos)
            program['light_color'].value = (1.0, 1.0, 1.0)
            program['camera_pos'].value = tuple(camera_pos)
            program['ambient_strength'].value = 0.3
            program['specular_strength'].value = 0.5
            program['has_texture'].value = has_texture
            
            if texture:
                texture.use(location=0)
                program['texture_sampler'].value = 0
            
            fbo.use()
            ctx.clear(0.0, 0.0, 0.0, 0.0)  # Transparent background
            ctx.viewport = (0, 0, width, height)
            vao.render()
            
            # Read RGBA data and convert to PIL image
            raw_data = fbo.read(components=4)
            image = Image.frombytes('RGBA', (width, height), raw_data).transpose(Image.FLIP_TOP_BOTTOM)
            gif_frames.append(np.array(image))
        
        # Create animated GIF with transparency and automatic looping
        imageio.mimsave(str(output), gif_frames, fps=24, transparency=0, disposal=2, loop=0)
        
        ctx.release()

    def _create_spinning_video(self, input: Path, output: Path, input_ext: str) -> None:
        import trimesh
        import moderngl
        import numpy as np
        import imageio
        import tempfile
        from PIL import Image
        import pyrr
        
        scene_or_mesh = trimesh.load(str(input))
        
        if hasattr(scene_or_mesh, 'geometry') and scene_or_mesh.geometry:
            geometries = list(scene_or_mesh.geometry.values())
            if geometries:
                mesh = geometries[0]
            else:
                mesh = scene_or_mesh.dump(concatenate=True)
        else:
            mesh = scene_or_mesh
        
        vertices = mesh.vertices.astype(np.float32)
        faces = mesh.faces.astype(np.uint32)
        
        if hasattr(mesh, 'vertex_normals'):
            normals = mesh.vertex_normals.astype(np.float32)
        else:
            mesh.fix_normals()
            normals = mesh.vertex_normals.astype(np.float32)
        
        # Get UV coordinates
        uvs = None
        try:
            if hasattr(scene_or_mesh, 'geometry'):
                for geom in scene_or_mesh.geometry.values():
                    if hasattr(geom.visual, 'uv') and geom.visual.uv is not None:
                        uvs = geom.visual.uv.astype(np.float32)
                        uvs[:, 1] = 1.0 - uvs[:, 1]  # Flip Y coordinate
                        uvs = np.clip(uvs, 0.0, 1.0)
                        break
            elif hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
                uvs = mesh.visual.uv.astype(np.float32)
                uvs[:, 1] = 1.0 - uvs[:, 1]
                uvs = np.clip(uvs, 0.0, 1.0)
        except Exception:
            pass
        
        if uvs is None:
            uvs = np.zeros((len(vertices), 2), dtype=np.float32)
            x_range = vertices[:, 0].max() - vertices[:, 0].min()
            z_range = vertices[:, 2].max() - vertices[:, 2].min()
            if x_range > 0:
                uvs[:, 0] = (vertices[:, 0] - vertices[:, 0].min()) / x_range
            if z_range > 0:
                uvs[:, 1] = (vertices[:, 2] - vertices[:, 2].min()) / z_range
            uvs = np.clip(uvs, 0.0, 1.0)
        
        # Extract texture
        texture_data = None
        has_texture = False
        
        try:
            if hasattr(scene_or_mesh, 'geometry'):
                for geom in scene_or_mesh.geometry.values():
                    if hasattr(geom.visual, 'material'):
                        material = geom.visual.material
                        if hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
                            texture_data = material.baseColorTexture
                            has_texture = True
                            break
                        elif hasattr(material, 'image') and material.image is not None:
                            texture_data = material.image
                            has_texture = True
                            break
            elif hasattr(mesh.visual, 'material'):
                material = mesh.visual.material
                if hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
                    texture_data = material.baseColorTexture
                    has_texture = True
                elif hasattr(material, 'image') and material.image is not None:
                    texture_data = material.image
                    has_texture = True
        except Exception:
            has_texture = False
        
        # Get vertex colors
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            colors = mesh.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
        else:
            base_color = [1.0, 1.0, 1.0] if has_texture else [0.6, 0.7, 0.8]
            colors = np.ones((len(vertices), 3), dtype=np.float32)
            colors[:, :] = base_color
        
        vertex_data = np.column_stack([vertices, normals, colors, uvs])
        
        # Center and scale model
        center = vertices.mean(axis=0)
        vertices_centered = vertices - center
        scale = 1.0 / np.max(np.linalg.norm(vertices_centered, axis=1))
        
        ctx = moderngl.create_context(standalone=True, require=330)
        
        vertex_shader = """
        #version 330 core
        
        in vec3 in_position;
        in vec3 in_normal;
        in vec3 in_color;
        in vec2 in_uv;
        
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        uniform mat3 normal_matrix;
        
        out vec3 world_pos;
        out vec3 normal;
        out vec3 color;
        out vec2 uv;
        
        void main() {
            vec4 world_position = model * vec4(in_position, 1.0);
            world_pos = world_position.xyz;
            normal = normalize(normal_matrix * in_normal);
            color = in_color;
            uv = in_uv;
            gl_Position = projection * view * world_position;
        }
        """
        
        fragment_shader = """
        #version 330 core
        
        in vec3 world_pos;
        in vec3 normal;
        in vec3 color;
        in vec2 uv;
        
        uniform vec3 light_pos;
        uniform vec3 light_color;
        uniform vec3 camera_pos;
        uniform float ambient_strength;
        uniform float specular_strength;
        uniform sampler2D texture_sampler;
        uniform bool has_texture;
        
        out vec4 fragColor;
        
        void main() {
            vec2 clamped_uv = clamp(uv, 0.0, 1.0);
            
            vec3 base_color;
            if (has_texture) {
                vec3 texture_color = texture(texture_sampler, clamped_uv).rgb;
                float color_intensity = (color.r + color.g + color.b) / 3.0;
                if (color_intensity > 0.8) {
                    base_color = texture_color;
                } else {
                    base_color = texture_color * color;
                }
            } else {
                base_color = color;
            }
            
            vec3 norm = normalize(normal);
            if (length(norm) < 0.1) {
                norm = vec3(0.0, 1.0, 0.0);
            }
            
            vec3 ambient = ambient_strength * light_color;
            
            vec3 light_dir = normalize(light_pos - world_pos);
            float diff = max(dot(norm, light_dir), 0.0);
            vec3 diffuse = diff * light_color;
            
            vec3 view_dir = normalize(camera_pos - world_pos);
            vec3 reflect_dir = reflect(-light_dir, norm);
            float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 64);
            vec3 specular = specular_strength * spec * light_color;
            
            vec3 result = (ambient + diffuse + specular) * base_color;
            result = clamp(result, 0.0, 1.0);
            
            fragColor = vec4(result, 1.0);
        }
        """
        
        program = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
        
        # Create texture
        texture = None
        if has_texture and texture_data is not None:
            try:
                if hasattr(texture_data, 'tobytes'):
                    width, height = texture_data.size
                    texture_bytes = texture_data.tobytes('raw', 'RGB')
                else:
                    from PIL import Image
                    if isinstance(texture_data, np.ndarray):
                        texture_pil = Image.fromarray(texture_data)
                    else:
                        texture_pil = texture_data
                    width, height = texture_pil.size
                    texture_bytes = texture_pil.tobytes('raw', 'RGB')
                
                texture = ctx.texture((width, height), 3, texture_bytes)
                texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
                texture.repeat_x = False
                texture.repeat_y = False
                texture.build_mipmaps()
            except Exception:
                has_texture = False
                texture = None
        
        if not has_texture:
            default_texture_data = np.ones((4, 4, 3), dtype=np.uint8) * 255
            texture = ctx.texture((4, 4), 3, default_texture_data.tobytes())
        
        vbo = ctx.buffer(vertex_data.tobytes())
        ibo = ctx.buffer(faces.tobytes())
        vao = ctx.vertex_array(program, [(vbo, '3f 3f 3f 2f', 'in_position', 'in_normal', 'in_color', 'in_uv')], ibo)
        
        width, height = 1024, 1024
        fbo = ctx.framebuffer(
            color_attachments=ctx.texture((width, height), 4),  # RGBA for transparency
            depth_attachment=ctx.depth_texture((width, height))
        )
        
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.CULL_FACE)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        
        projection = pyrr.matrix44.create_perspective_projection_matrix(45.0, 1.0, 0.1, 100.0)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            frames = []
            
            for i in range(72):
                angle = i * (2 * np.pi / 72)
                camera_distance = 3.0
                camera_pos = np.array([
                    camera_distance * np.cos(angle),
                    camera_distance * 0.5,
                    camera_distance * np.sin(angle)
                ])
                
                view = pyrr.matrix44.create_look_at(
                    camera_pos, np.array([0, 0, 0]), np.array([0, 1, 0])
                )
                
                model = np.eye(4, dtype=np.float32)
                model[:3, 3] = -center * scale
                model[0, 0] = scale
                model[1, 1] = scale
                model[2, 2] = scale
                
                normal_matrix = np.linalg.inv(model[:3, :3]).T
                light_pos = camera_pos + np.array([1, 2, 1])
                
                program['model'].write(model.astype(np.float32).tobytes())
                program['view'].write(view.astype(np.float32).tobytes())
                program['projection'].write(projection.astype(np.float32).tobytes())
                program['normal_matrix'].write(normal_matrix.astype(np.float32).tobytes())
                program['light_pos'].value = tuple(light_pos)
                program['light_color'].value = (1.0, 1.0, 1.0)
                program['camera_pos'].value = tuple(camera_pos)
                program['ambient_strength'].value = 0.3
                program['specular_strength'].value = 0.5
                program['has_texture'].value = has_texture
                
                if texture:
                    texture.use(location=0)
                    program['texture_sampler'].value = 0
                
                fbo.use()
                ctx.clear(0.0, 0.0, 0.0, 0.0)  # Transparent background
                ctx.viewport = (0, 0, width, height)
                vao.render()
                
                raw_data = fbo.read(components=4)  # RGBA
                image = Image.frombytes('RGBA', (width, height), raw_data).transpose(Image.FLIP_TOP_BOTTOM)
                # Convert to RGB with green screen background for easy chroma key removal
                rgb_image = Image.new('RGB', (width, height), (0, 255, 0))  # Pure green
                rgb_image.paste(image, mask=image.split()[-1])  # Use alpha as mask
                image = rgb_image
                
                frame_path = Path(temp_dir) / f"frame_{i:03d}.png"
                image.save(frame_path)
                frames.append(str(frame_path))
            
            with imageio.get_writer(str(output), fps=24) as writer:
                for frame_path in frames:
                    image = imageio.imread(frame_path)
                    writer.append_data(image)
        
        ctx.release()
