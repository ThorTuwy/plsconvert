from pathlib import Path
from plsconvert.converters.abstract import Converter
from plsconvert.converters.registry import addMethodData, registerConverter
from plsconvert.utils.dependency import Dependencies, LibDependency as Lib
from plsconvert.utils.graph import PairList

@registerConverter
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

    @addMethodData(PairList.all2all(["glb", "gltf", "obj"], ["glb", "gltf", "obj", "mp4", "png", "gif"]), False)
    def model_to_frames(
        self, input: Path, output: Path, input_extension: str, output_extension: str
    ) -> None:
        if output_extension == "mp4":
            self._create_video(input, output, input_extension, transparent=False)
        elif output_extension == "png":
            self._create_png_render(input, output, input_extension)
        elif output_extension == "gif":
            self._create_gif(input, output, input_extension)
        else:
            self._convert_3d_format(input, output, input_extension, output_extension)

    def _convert_3d_format(self, input: Path, output: Path, input_ext: str, output_ext: str) -> None:
        import trimesh
        
        mesh = trimesh.load(str(input))
        
        if hasattr(mesh, 'dump'):
            try:
                mesh = mesh.dump(concatenate=True)
            except Exception:
                if hasattr(mesh, 'geometry') and mesh.geometry:
                    mesh = list(mesh.geometry.values())[0]
        
        if output_ext.lower() == "obj":
            mesh.export(str(output))
        else:
            exported_data = mesh.export(file_type=output_ext.lower())
            with open(output, 'wb') as f:
                f.write(exported_data)

    def _load_model_data(self, input: Path) -> dict:
        """
        Load 3D model data once and return all necessary information for rendering.
        
        Args:
            input: Path to the 3D model file
            
        Returns:
            Dictionary containing all model data needed for rendering
        """
        import trimesh
        import numpy as np
        
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
        
        return {
            'vertex_data': vertex_data,
            'faces': faces,
            'center': center,
            'scale': scale,
            'texture_data': texture_data,
            'has_texture': has_texture
        }

    def _generate_frame(self, model_data: dict, transparent: bool = True, rotation_degrees: float = 0.0) -> bytes:
        """
        Generate a single frame from pre-loaded 3D model data.
        
        Args:
            model_data: Pre-loaded model data from _load_model_data
            transparent: If True, renders with transparent background. If False, uses green chroma key
            rotation_degrees: Rotation angle in degrees (0-360)
            
        Returns:
            Image data as bytes
        """
        import moderngl
        import numpy as np
        from PIL import Image
        import pyrr
        
        vertex_data = model_data['vertex_data']
        faces = model_data['faces']
        center = model_data['center']
        scale = model_data['scale']
        texture_data = model_data['texture_data']
        has_texture = model_data['has_texture']
        
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
            color_attachments=ctx.texture((width, height), 4),
            depth_attachment=ctx.depth_texture((width, height))
        )
        
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.CULL_FACE)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        
        projection = pyrr.matrix44.create_perspective_projection_matrix(45.0, 1.0, 0.1, 100.0)
        
        # Calculate camera position based on rotation
        rotation_rad = np.radians(rotation_degrees)
        camera_distance = 3.0
        camera_pos = np.array([
            camera_distance * np.cos(rotation_rad),
            camera_distance * 0.5,
            camera_distance * np.sin(rotation_rad)
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
        if transparent:
            ctx.clear(0.0, 0.0, 0.0, 0.0)  # Transparent background
        else:
            ctx.clear(0.0, 1.0, 0.0, 1.0)  # Green chroma key background
        ctx.viewport = (0, 0, width, height)
        vao.render()
        
        # Read image data
        raw_data = fbo.read(components=4)
        image = Image.frombytes('RGBA', (width, height), raw_data)
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        
        # Convert to appropriate format based on transparent flag
        if not transparent:
            # Convert to RGB with green background for chroma key
            rgb_image = Image.new('RGB', (width, height), (0, 255, 0))
            rgb_image.paste(image, mask=image.split()[-1])
            image = rgb_image
        
        ctx.release()
        
        # Return image data as bytes
        import io
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()

    def _create_png_render(self, input: Path, output: Path, input_ext: str) -> None:
        """Create a single PNG render with transparent background."""
        model_data = self._load_model_data(input)
        frame_data = self._generate_frame(model_data, transparent=True, rotation_degrees=45.0)
        with open(output, 'wb') as f:
            f.write(frame_data)

    def _create_video(self, input: Path, output: Path, input_ext: str, transparent: bool = False) -> None:
        """Create a spinning video (MP4) with specified transparency."""
        import imageio
        import tempfile
        
        # Load model data once
        model_data = self._load_model_data(input)
        
        # MP4 settings: 72 frames, 24 fps, 3 seconds duration
        num_frames = 72
        fps = 24
        
        with tempfile.TemporaryDirectory() as temp_dir:
            frames = []
            
            for i in range(num_frames):
                angle = i * (360.0 / num_frames)
                frame_data = self._generate_frame(model_data, transparent=transparent, rotation_degrees=angle)
                
                frame_path = Path(temp_dir) / f"frame_{i:03d}.png"
                with open(frame_path, 'wb') as f:
                    f.write(frame_data)
                frames.append(str(frame_path))
            
            with imageio.get_writer(str(output), fps=fps) as writer:
                for frame_path in frames:
                    image = imageio.imread(frame_path)
                    writer.append_data(image)

    def _create_gif(self, input: Path, output: Path, input_ext: str) -> None:
        """Create a spinning GIF with transparent background."""
        import imageio
        import tempfile
        
        # Load model data once
        model_data = self._load_model_data(input)
        
        # GIF settings: 60 frames, 24 fps, 2.5 seconds duration
        num_frames = 60
        fps = 24
        
        with tempfile.TemporaryDirectory() as temp_dir:
            frames = []
            
            for i in range(num_frames):
                angle = i * (360.0 / num_frames)
                frame_data = self._generate_frame(model_data, transparent=True, rotation_degrees=angle)
                
                frame_path = Path(temp_dir) / f"frame_{i:03d}.png"
                with open(frame_path, 'wb') as f:
                    f.write(frame_data)
                frames.append(str(frame_path))
            
            # Create animated GIF with transparency
            gif_frames = []
            for frame_path in frames:
                image = imageio.imread(frame_path)
                gif_frames.append(image)
            
            imageio.mimsave(str(output), gif_frames, fps=fps, transparency=0, disposal=2, loop=0) 