import pygame
from pygame.locals import *
from OpenGL.GL import *
import numpy as np
import glm
import ctypes
import argparse
import sys
import os

def load_obj(filename):
    vertices = []
    normals = []
    textures = []
    faces = []

    with open(filename, "r") as file:
        for line in file:
            if line.startswith("v "):  
                parts = line.strip().split()
                vertices.append(tuple(map(float, parts[1:4])))
            elif line.startswith("vn "):  
                parts = line.strip().split()
                normals.append(tuple(map(float, parts[1:4])))
            elif line.startswith("vt "):  
                parts = line.strip().split()
                textures.append(tuple(map(float, parts[1:3])))
            elif line.startswith("f "):  
                parts = line.strip().split()
                face = []
                for p in parts[1:]:
                    vals = p.split('/')
                    vertex_idx = int(vals[0]) - 1
                    texture_idx = int(vals[1]) - 1 if len(vals) > 1 and vals[1] else None
                    normal_idx = int(vals[2]) - 1 if len(vals) > 2 and vals[2] else None
                    face.append((vertex_idx, texture_idx, normal_idx))
                faces.append(face)

    return vertices, normals, textures, faces

def calculate_normals(vertices, faces):
    normals = np.zeros((len(vertices), 3), dtype=np.float32)
    counts = np.zeros(len(vertices), dtype=np.int32)

    for face in faces:
        idxs = [vertex[0] for vertex in face]
        v0 = np.array(vertices[idxs[0]])
        for i in range(1, len(idxs) - 1):
            v1 = np.array(vertices[idxs[i]])
            v2 = np.array(vertices[idxs[i + 1]])
            normal = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(normal)
            if norm != 0:
                normal /= norm  
                for idx in [idxs[0], idxs[i], idxs[i + 1]]:
                    normals[idx] += normal
                    counts[idx] += 1

    for i in range(len(normals)):
        if counts[i] > 0:
            normals[i] /= counts[i]
            norm = np.linalg.norm(normals[i])
            if norm != 0:
                normals[i] /= norm

    return normals

def calculate_bounding_sphere(vertices):
    vertices_np = np.array(vertices)
    center = np.mean(vertices_np, axis=0)
    radius = np.max(np.linalg.norm(vertices_np - center, axis=1))
    return center, radius

def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)

    result = glGetShaderiv(shader, GL_COMPILE_STATUS)
    if not result:
        error = glGetShaderInfoLog(shader).decode()
        shader_type_str = 'vertex' if shader_type == GL_VERTEX_SHADER else 'fragment'
        raise RuntimeError(f"Erreur de compilation du shader {shader_type_str}: {error}")
    return shader

def create_shader_program(vertex_source, fragment_source):
    vertex_shader = compile_shader(vertex_source, GL_VERTEX_SHADER)
    fragment_shader = compile_shader(fragment_source, GL_FRAGMENT_SHADER)
    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)

    result = glGetProgramiv(program, GL_LINK_STATUS)
    if not result:
        error = glGetProgramInfoLog(program).decode()
        raise RuntimeError(f"Erreur de linkage du shader: {error}")

    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    return program

def main():
    parser = argparse.ArgumentParser(description="Visualiseur OBJ")
    parser.add_argument('-i', '--input', type=str, required=True, help="Chemin vers le fichier OBJ à charger")
    args = parser.parse_args()

    obj_file = args.input

    if not os.path.exists(obj_file):
        print(f"Erreur : le fichier {obj_file} n'existe pas.")
        sys.exit(1)

    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Visualiseur OBJ")
    glViewport(0, 0, display[0], display[1])

    glEnable(GL_DEPTH_TEST)
    glClearColor(0.1, 0.1, 0.1, 1.0)

    vertices, normals_file, textures, faces = load_obj(obj_file)

    vertices = np.array(vertices, dtype=np.float32)
    normals_file = np.array(normals_file, dtype=np.float32)
    textures = np.array(textures, dtype=np.float32)

    if normals_file.size == 0:
        normals = calculate_normals(vertices, faces)
    else:
        normals = normals_file

    center, radius = calculate_bounding_sphere(vertices)
    camera_distance = radius * 3
    camera_position = center + np.array([0, 0, camera_distance])

    vertex_positions = []
    vertex_normals = []
    vertex_texcoords = []

    for face in faces:
        if len(face) < 3:
            continue
        v0 = face[0]
        for i in range(1, len(face) -1):
            v1 = face[i]
            v2 = face[i+1]
            for vertex in [v0, v1, v2]:
                v_idx, t_idx, n_idx = vertex
                vertex_positions.append(vertices[v_idx])
                if normals.size > 0:
                    if n_idx is not None and n_idx < len(normals):
                        vertex_normals.append(normals[n_idx])
                    else:
                        vertex_normals.append(normals[v_idx])
                else:
                    vertex_normals.append((0.0, 0.0, 0.0))
                if textures.size > 0 and t_idx is not None and t_idx < len(textures):
                    vertex_texcoords.append(textures[t_idx])
                else:
                    vertex_texcoords.append((0.0, 0.0))

    vertex_positions = np.array(vertex_positions, dtype=np.float32)
    vertex_normals = np.array(vertex_normals, dtype=np.float32)
    vertex_texcoords = np.array(vertex_texcoords, dtype=np.float32)

    vertex_data = np.hstack((
        vertex_positions,
        vertex_normals,
        vertex_texcoords
    ))

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data.flatten(), GL_STATIC_DRAW)

    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    stride = 8 * 4  
    offset_position = ctypes.c_void_p(0)
    offset_normal = ctypes.c_void_p(3 * 4)
    offset_texcoord = ctypes.c_void_p(6 * 4)

    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, offset_position)

    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, offset_normal)

    glEnableVertexAttribArray(2)
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, offset_texcoord)

    try:
        with open('vertex_shader.glsl', 'r') as f:
            vertex_shader_source = f.read()
        with open('fragment_shader.glsl', 'r') as f:
            fragment_shader_source = f.read()
    except FileNotFoundError as e:
        print(f"Erreur : {e}")
        pygame.quit()
        return

    shader_program = create_shader_program(vertex_shader_source, fragment_shader_source)

    glUseProgram(shader_program)
    model_loc = glGetUniformLocation(shader_program, 'model')
    view_loc = glGetUniformLocation(shader_program, 'view')
    projection_loc = glGetUniformLocation(shader_program, 'projection')
    light_pos_loc = glGetUniformLocation(shader_program, 'lightPos')
    view_pos_loc = glGetUniformLocation(shader_program, 'viewPos')
    light_color_loc = glGetUniformLocation(shader_program, 'lightColor')
    object_color_loc = glGetUniformLocation(shader_program, 'objectColor')

    if light_color_loc == -1 or object_color_loc == -1:
        print("Erreur : Les uniformes 'lightColor' ou 'objectColor' n'ont pas été trouvées dans le shader.")
        pygame.quit()
        return

    glUniform3f(light_color_loc, 1.0, 1.0, 1.0)

    glUniform3f(object_color_loc, 0.6, 0.6, 0.6)

    rotation_x, rotation_y = 0.0, 0.0
    last_mouse_pos = None
    wireframe_mode = False

    clock = pygame.time.Clock()
    running = True
    while running:
        clock.tick(60)  
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 4:  
                    camera_distance -= radius * 0.1
                    if camera_distance < radius * 0.1:
                        camera_distance = radius * 0.1
                    camera_position = center + np.array([0, 0, camera_distance])
                elif event.button == 5:  
                    camera_distance += radius * 0.1
                    camera_position = center + np.array([0, 0, camera_distance])
                elif event.button == 3:  
                    last_mouse_pos = event.pos
            elif event.type == MOUSEBUTTONUP:
                if event.button == 3:
                    last_mouse_pos = None
            elif event.type == MOUSEMOTION:
                if pygame.mouse.get_pressed()[2]:
                    if last_mouse_pos:
                        dx = event.pos[0] - last_mouse_pos[0]
                        dy = event.pos[1] - last_mouse_pos[1]
                        rotation_x += dy * 0.5
                        rotation_y += dx * 0.5
                    last_mouse_pos = event.pos
            elif event.type == KEYDOWN:
                if event.key == K_SPACE:
                    wireframe_mode = not wireframe_mode

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(shader_program)

        model = glm.mat4(1.0)
        model = glm.rotate(model, glm.radians(rotation_x), glm.vec3(1, 0, 0))
        model = glm.rotate(model, glm.radians(rotation_y), glm.vec3(0, 1, 0))

        view = glm.lookAt(
            glm.vec3(camera_position[0], camera_position[1], camera_position[2]),
            glm.vec3(center[0], center[1], center[2]),
            glm.vec3(0, 1, 0)
        )

        projection = glm.perspective(
            glm.radians(45),
            display[0] / display[1],
            radius * 0.1,
            radius * 10
        )

        glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm.value_ptr(model))
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm.value_ptr(view))
        glUniformMatrix4fv(projection_loc, 1, GL_FALSE, glm.value_ptr(projection))

        glUniform3f(light_pos_loc, 4.0, 4.0, 4.0)
        glUniform3f(view_pos_loc, camera_position[0], camera_position[1], camera_position[2])

        if wireframe_mode:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        glBindVertexArray(vao)
        glDrawArrays(GL_TRIANGLES, 0, len(vertex_positions))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()