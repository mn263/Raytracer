import math
import struct
import zlib


def blend(c1, c2):
    # # print c1.x
    # # print c1.y
    # # print c1.z
    #
    # c1.x = int(c1.x)
    # c1.y = int(c1.y)
    # c1.z = int(c1.z)
    # for i in range(3):
    #     c2[i] = int(c2[i])
    # # print c2[3]
    # x = c1.x * (0xFF - c2[3]) + c2[0] * c2[3] >> 8
    # y = c1.y * (0xFF - c2[3]) + c2[1] * c2[3] >> 8
    # z = c1.z * (0xFF - c2[3]) + c2[2] * c2[3] >> 8
    # return [x, y, z]
    return [c1[i] * (0xFF - c2[3]) + c2[i] * c2[3] >> 8 for i in range(3)]


def vector_mult(v1, mult_val):
    x = v1.x * mult_val
    y = v1.y * mult_val
    z = v1.z * mult_val
    return Vector3(x, y, z)


def multiply_vectors(v1, v2):
    return Vector3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z)


def divide_vectors(v1, v2):
    return Vector3(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z)


def add_vectors(v1, v2):
    return Vector3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z)


def subtract_vectors(v1, v2):
    return Vector3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z)


class PNG(object):
    def __init__(self, height, width, background_color):
        self.height = height
        self.width = width
        self.background_color = [int(x) for x in background_color]
        # create canvas
        self.canvas = []
        for row in range(height):
            row = []
            for column in range(width):
                row.append(self.background_color)
            self.canvas.append(row)
        # initially fill canvas with background color
        for x in range(0, self.width):
            for y in range(0, self.width):
                self.point(x, y, self.background_color)

    def point(self, x, y, color=None):
        if x < 0 or y < 0 or x > self.width - 1 or y > self.height - 1:
            return
        if not color:
            color = self.background_color
        self.canvas[y][x] = blend(self.canvas[y][x], color)

    def dump(self):
        raw_list = []
        for y in range(self.height):
            raw_list.append(chr(0))  # filter type 0 (None)
            for x in range(self.width):
                raw_list.append(struct.pack("!3B", *self.canvas[y][x]))
        raw_data = ''.join(raw_list)

        return struct.pack("8B", 137, 80, 78, 71, 13, 10, 26, 10) + self.pack_chunk('IHDR', struct.pack("!2I5B", self.width, self.height, 8, 2, 0, 0, 0)) + self.pack_chunk('tRNS', struct.pack("!6B", 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF)) + self.pack_chunk('IDAT', zlib.compress(raw_data, 9)) + self.pack_chunk('IEND', '')

    def pack_chunk(self, tag, data):
        to_check = tag + data
        return struct.pack("!I", len(data)) + to_check + struct.pack("!I", zlib.crc32(to_check) & 0xFFFFFFFF)


class Vector3(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.magnitude = math.sqrt(sum([math.pow(x, 2), math.pow(y, 2), math.pow(z, 2)]))
        self.data = [x, y, z]
        self.setup()

    def setup(self):
        self.magnitude = math.sqrt(sum([math.pow(x, 2) for x in self.data]))
        self.x = self.data[0]
        self.y = self.data[1]
        self.z = self.data[2]

    def set_x(self, val):
        self.data[0] = val
        self.setup()

    def set_y(self, val):
        self.data[1] = val
        self.setup()

    def set_z(self, val):
        self.data[2] = val
        self.setup()

    def dot(self, other):
        # This will provide the dot product of the vector.
        result = []
        for i in range(len(self.data)):
            result.append(self.data[i] * other.data[i])
        return sum(result)

    def cross(self, other):
        # This should only be used with a 3 dimensional vector.
        # Returns the vector result of the cross product.
        new_x = (self.y * other.z) - (self.z * other.y)
        new_y = (self.z * other.x) - (self.x * other.z)
        new_z = (self.x * other.y) - (self.y * other.x)
        return Vector3(new_x, new_y, new_z)

    def normalize(self):
        mag = float(self.magnitude)
        self.set_x(float(self.x / mag))
        self.set_y(float(self.y / mag))
        self.set_z(float(self.z / mag))
        return self

    def get_coords(self):
        return self.x, self.y, self.z

    def __str__(self):
        return "({}, {}, {})".format(self.x, self.y, self.z)


class Camera(object):
    def __init__(self, clarity, l_at, l_from, l_up):
        self.clarity = clarity
        self.look_at_direction = l_at
        self.look_from = l_from
        self.look_up = l_up
        self.g = Vector3(0, 0, -1)
        self.t = Vector3(0, 100, 0)
        # TODO: fix some of the variables in camera
        self.normal = 2
        self.angle = 90
        self.width = clarity
        self.height = clarity
        self.w = None
        self.u = None
        self.v = None
        self.top = None
        self.bottom = None
        self.left = None
        self.right = None
        self.calc()

    def calc(self):
        self.w = Vector3(self.g.x * -1, self.g.y * -1, self.g.z * -1)
        self.u = self.t.cross(self.w).normalize()
        self.v = self.w.cross(self.u)
        self.top = abs(self.normal) * math.tan(math.radians(self.angle / 2))
        self.bottom = -self.top
        self.right = self.top * self.width / self.height
        self.left = -self.right

    def look_at(self, point):
        direction = subtract_vectors(point, self.look_from)
        self.g = Vector3(direction.x, direction.y, direction.z)
        self.g.normalize()
        self.calc()


class Sphere(object):
    def __init__(self, center, radius, reflective_color, diffuse_color, spec_highlight, phong_const):
        self.center = center
        self.radius = radius
        if reflective_color:
            self.reflective = True
            self.diffuse = False
            self.color = reflective_color
        if diffuse_color:
            self.reflective = False
            self.diffuse = True
            self.color = diffuse_color

        self.diffuse_color = diffuse_color
        self.reflective_color = reflective_color
#     TODO: write code that uses specular_highlight and phong_const
        self.spec_highlight = spec_highlight
        self.phong_const = phong_const

    def is_sphere(self):
        return True

    def get_normal_v(self, intersecting_point):
        return vector_mult(add_vectors(intersecting_point, vector_mult(self.center, -1)), (1/self.radius))


class Triangle(object):
    def __init__(self, first, second, third, reflective_color, diffuse_color, spec_highlight, phong_const):
        self.first = first
        self.second = second
        self.third = third

        if reflective_color:
            self.reflective = True
            self.diffuse = False
            self.color = reflective_color
        if diffuse_color:
            self.reflective = False
            self.diffuse = True
            self.color = diffuse_color

        self.diffuse_color = diffuse_color
        self.reflective_color = reflective_color
        self.spec_highlight = spec_highlight
        self.phong_const = phong_const

    def is_sphere(self):
        return False

    def get_normal_v(self):
        v = add_vectors(self.first, vector_mult(self.second, -1))
        x_v, y_v, z_v = v.get_coords()
        w = add_vectors(self.third, vector_mult(self.first, -1))
        x_w, y_w, z_w = w.get_coords()
        return Vector3((y_v * z_w) - (z_v * y_w), (z_v * x_w) - (x_v * z_w), (z_v * y_w) - (y_v * x_w))


class Ray(object):
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction.normalize()


class Hit(object):  # A simple object representing a ray hit with a sphere.
    def __init__(self, dist, world_hit, normal, obj_color, spec_highlight, phong_const):
        # Dist should be a scalar, and normal should be a vector object.
        self.dist = dist
        self.world_hit = world_hit
        self.normal = normal
        self.obj_color = obj_color
        self.spec_highlight = spec_highlight
        self.phong_const = phong_const


class Light(object):
    def __init__(self, position, color, intensity):
        self.position = position
        self.color = color
        self.intensity = intensity


def create_ray(camera, x_pixel, y_pixel):
    x_cam = camera.left + ((camera.right - camera.left) * (x_pixel + 0.5) / camera.width)
    y_cam = camera.bottom + ((camera.top - camera.bottom) * (y_pixel + 0.5) / camera.height)
    result = add_vectors(Vector3(camera.w.x * -camera.normal, camera.w.y * -camera.normal, camera.w.z * -camera.normal),
                         Vector3(camera.u.x * x_cam, camera.u.y * x_cam, camera.u.z * x_cam))

    direction = add_vectors(result, Vector3(camera.v.x * y_cam, camera.v.y * y_cam, camera.v.z * y_cam))
    return Ray(camera.look_from, direction)


def load_camera(clarity, at_line, from_line, up_line, angle_line):

    look_at = Vector3(float(at_line[1]), float(at_line[2]), float(at_line[3]))
    look_from = Vector3(float(from_line[1]), float(from_line[2]), float(from_line[3]))
    look_up = Vector3(float(up_line[1]), float(up_line[2]), float(up_line[3]))

    camera = Camera(clarity, look_at, look_from, look_up)
    # TODO: figure out what is wrong with the angle (it works better with 80)
    camera.angle = float(angle_line[1])
    camera.angle = 61
    camera.look_at(look_at)
    camera.calc()
    return camera


# TODO: rename "Hit" to "Intersection"
def check_for_intersection(ray, shape):
    if shape.is_sphere():
        sphere = shape
        x_d, y_d, z_d = ray.direction.get_coords()
        x_o, y_o, z_o = ray.origin.get_coords()
        x_c, y_c, z_c = sphere.center.get_coords()

        b = 2 * ((x_d * x_o) - (x_d * x_c) + (y_d * y_o) - (y_d * y_c) + (z_d * z_o) - (z_d * z_c))
        c = (
            pow(x_d, 2) + pow(x_c, 2) - (2 * x_o * x_c) +
            pow(y_d, 2) + pow(y_c, 2) - (2 * y_o * y_c) +
            pow(z_d, 2) + pow(z_c, 2) - (2 * z_o * z_c) - pow(sphere.radius, 2))
        discriminant = pow(b, 2) - (4 * c)

        t = None
        if discriminant == 0:
            t = (-0.5 * b)
        elif discriminant < 0:
            return  # missed
        elif discriminant > 0:
            t0 = (-b - pow(discriminant, 0.5))/2
            t1 = (-b + pow(discriminant, 0.5))/2
            if t0 < 0 < t1:
                t = t1
            elif t0 > 0 and t1 > 0:
                t = t1

        intersecting_point = add_vectors(ray.origin, vector_mult(ray.direction, t))
        x_i, y_i, z_i = intersecting_point.get_coords()
        normal = sphere.get_normal_v(intersecting_point)
        undr_sqr = pow(x_o - x_i, 2) + pow(y_o - y_i, 2) + pow(z_o - z_i, 2)
        if undr_sqr < 0:
            undr_sqr *= -1
        distance = pow(undr_sqr, 0.5)

        if sphere.diffuse:
            return Hit(distance, intersecting_point, normal, sphere.diffuse_color, sphere.spec_highlight, sphere.phong_const)
        elif sphere.reflective:
            return Hit(distance, intersecting_point, normal, sphere.reflective_color, sphere.spec_highlight, sphere.phong_const)

    elif not shape.is_sphere():
        # TODO: make hit for triangles
        triangle = shape
        u = add_vectors(triangle.second, vector_mult(triangle.first, -1))
        v = add_vectors(triangle.third, vector_mult(triangle.first, -1))
        n = u.cross(v)
        if n.x == 0 and n.y == 0 and n.z == 0:
            print "THEY ALL EQUAL ZERO"
            return
        w_o = add_vectors(ray.origin, vector_mult(triangle.second, -1))
        a = -1 * n.dot(w_o)
        b = n.dot(ray.direction)
        if abs(b) < 0.0000001:
            if a == 0:
                print "In Same Plane"
            else:
                print "No Intersect"
            return

        r = a / b
        # if r < 0:
        #     print "No Intersect 2"
        #     return
        intersecting_point = add_vectors(ray.origin, vector_mult(ray.direction, r))

        uu = u.dot(u)
        uv = u.dot(v)
        vv = v.dot(v)
        w = add_vectors(intersecting_point, vector_mult(triangle.first, -1))
        wu = w.dot(u)
        wv = w.dot(v)
        # D = uv.cross(uv) - uu.cross(vv)
        D = (uv * uv) - (uu * vv)
        S = ((uv * wv) - (vv * wu)) / D
        if S < 0 or S > 1:
            # print "Outside Triangle"
            return
        t = ((uv * wu) - (uu * wv)) / D
        if t < 0 or (S + t) > 1:
            # print "Outside Triangle 2"
            return

        # print "Intersected!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        # print intersecting_point
        #


        # get distance
        undr_sqr = pow(ray.origin.x - intersecting_point.x, 2) + pow(ray.origin.y - intersecting_point.y, 2) + pow(ray.origin.z - intersecting_point.z, 2)
        if undr_sqr < 0:
            undr_sqr *= -1
        distance = pow(undr_sqr, 0.5)

        normal = triangle.get_normal_v()

        if triangle.diffuse:
            return Hit(distance, intersecting_point, normal, triangle.diffuse_color, triangle.spec_highlight, triangle.phong_const)
        elif triangle.reflective:
            return Hit(distance, intersecting_point, normal, triangle.reflective_color, triangle.spec_highlight, triangle.phong_const)
        # normal = triangle.get_normal_v()
        # a, b, c, = normal.get_coords()
        # normal.normalize()
        # # a, b, c, = normal.get_coords()
        # x_d, y_d, z_d = ray.direction.get_coords()
        # x_o, y_o, z_o = ray.origin.get_coords()
        #
        # thingy = normal.cross(ray.origin)




        # t = -(ax0 + by0 + cz0 + d) / (axd + byd + czd)
        # t = -((a * x_o) + (b * y_o) + (c * z_o) + distance ) / ( () + () + () )
        # return


def load_light_and_colors(direct_line, color_line, ambient_line, background_line):
    dir_to_light = Vector3(float(direct_line[1]), float(direct_line[2]), float(direct_line[3]))
    light_color = [float(color_line[1]), float(color_line[2]), float(color_line[3])]
    ambient_light = Vector3(float(ambient_line[1]), float(ambient_line[2]), float(ambient_line[3]))
    background_color = [float(background_line[1])*255, float(background_line[2])*255, float(background_line[3])*255, 255]
    return dir_to_light, light_color, ambient_light, background_color


def load_spheres(sphere_lines):
    # center, radius, reflective_color, diffuse_color
    spheres = []
    for line in sphere_lines:
        center = Vector3(float(line[2]), float(line[3]), float(line[4]))
        radius = float(line[6])
        if line[8] == "Diffuse":
            reflective_color = None
            diffuse_color = [float(line[9])*255, float(line[10])*255, float(line[11])*255, 255]
        else:
            reflective_color = [float(line[9])*255, float(line[10])*255, float(line[11])*255, 255]
            diffuse_color = None
        spec_highlight = Vector3(float(line[13]), float(line[14]), float(line[15]))
        phong_const = float(line[17])
        spheres.append(Sphere(center, radius, reflective_color, diffuse_color, spec_highlight, phong_const))
    return spheres


def load_triangles(triangle_lines):
    triangles = []
    for line in triangle_lines:
        first = Vector3(float(line[1]), float(line[2]), float(line[3]))
        second = Vector3(float(line[5]), float(line[6]), float(line[7]))
        third = Vector3(float(line[9]), float(line[10]), float(line[11]))
        if line[13] == "Diffuse":
            reflective_color = None
            diffuse_color = [float(line[14])*255, float(line[15])*255, float(line[16])*255, 255]
        else:
            reflective_color = [float(line[14])*255, float(line[15])*255, float(line[16])*255, 255]
            diffuse_color = None
        spec_highlight = Vector3(float(line[18]), float(line[19]), float(line[20]))
        phong_const = float(line[22])
        triangles.append(Triangle(first, second, third, reflective_color, diffuse_color, spec_highlight, phong_const))
    return triangles


def diffuse(hit, light):
    color = hit.obj_color
    new_color = [0, 0, 0, 255]
    for i in xrange(3):
        brightness = light.intensity * hit.normal.dot(subtract_vectors(light.position, hit.world_hit).normalize())
        new_color[i] = new_color[i] + max(0, int(color[i] * brightness))
    return new_color


def load_file_objects(input_file, image_size):
    f = open(input_file, 'r')
    lines = f.readlines()
    camera = load_camera(image_size,
                         lines[0].strip().split(" "),
                         lines[1].strip().split(" "),
                         lines[2].strip().split(" "),
                         lines[3].strip().split(" "))
    # TODO: figure out how to use ambient light
    dir_to_light, light_color, ambient_light, background_color = load_light_and_colors(
        lines[4].strip().split(" ")[:4],
        lines[4].strip().split(" ")[4:],
        lines[5].strip().split(" "),
        lines[6].strip().split(" ")
    )
    light = Light(dir_to_light, light_color, 1)

    sphere_lines = []
    triangle_lines = []
    for index in range(7, len(lines)):
        line = lines[index].strip().split(" ")
        if line[0] == "Sphere":
            sphere_lines.append(line)
        elif line[0] == "Triangle":
            triangle_lines.append(line)
    spheres = load_spheres(sphere_lines)
    triangles = load_triangles(triangle_lines)
    png = PNG(camera.clarity, camera.clarity, background_color)
    return camera, light, spheres, triangles, png


def run(image, image_size):
    camera, light, spheres, triangles, ppm = load_file_objects(image, image_size)
    shapes = []
    # for sphere in spheres:
    #     shapes.append(sphere)
    for triangle in triangles:
        shapes.append(triangle)

    for row_index in xrange(0, camera.width - 1):
        # Run across all pixels in the row
        for column_index in xrange(0, camera.height - 1):
            # draw save pixel at row and column
            ray = create_ray(camera, row_index, camera.height - column_index)
            closest_hit = None
            for shape in shapes:
                hit = check_for_intersection(ray, shape)
                if hit:
                    if not closest_hit:
                        closest_hit = hit
                    elif hit.dist < closest_hit.dist:
                        closest_hit = hit
            if closest_hit:  # Image already filled with bg-color, only update if we hit a shape
                color = diffuse(closest_hit, light)
                ppm.point(row_index, column_index, color)

    # Create PNG
    f = open("outfile.png", "wb")
    f.write(ppm.dump())
    f.close()

run(image="diffuse.rayTracing", image_size=513)
