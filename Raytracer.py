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
    # TODO: re-write in a way I understand
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
    def __init__(self, height, width, background_color, color):
        self.height = height
        self.width = width
        # TODO: figure what bgcolor and color really are
        self.background_color = background_color
        self.color = color
        pixel = self.background_color
        self.canvas = []
        for row in range(height):
            row = []
            for column in range(width):
                row.append(pixel)
            self.canvas.append(row)

####### DON'T UNDERSTAND######################################################

    def point(self, x, y, color=None):
        if x < 0 or y < 0 or x > self.width - 1 or y > self.height - 1:
            return
        if not color:
            color = self.color
        self.canvas[y][x] = blend(self.canvas[y][x], color)

    def _rect_helper(self, x0, y0, x1, y1):
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0
        return [x0, y0, x1, y1]

    def filled_rectangle(self, x0, y0, x1, y1):
        x0, y0, x1, y1 = self._rect_helper(x0, y0, x1, y1)
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                self.point(x, y, self.color)

    def dump(self):
        raw_list = []
        for y in range(self.height):
            raw_list.append(chr(0))  # filter type 0 (None)
            for x in range(self.width):
                raw_list.append(struct.pack("!3B", *self.canvas[y][x]))
        raw_data = ''.join(raw_list)

        # 8-bit image represented as RGB tuples
        # simple transparency, alpha is pure white
        return struct.pack("8B", 137, 80, 78, 71, 13, 10, 26, 10) + self.pack_chunk('IHDR', struct.pack("!2I5B", self.width, self.height, 8, 2, 0, 0, 0)) + self.pack_chunk('tRNS', struct.pack("!6B", 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF)) + self.pack_chunk('IDAT', zlib.compress(raw_data, 9)) + self.pack_chunk('IEND', '')

    def pack_chunk(self, tag, data):
        to_check = tag + data
        return struct.pack("!I", len(data)) + to_check + struct.pack("!I", zlib.crc32(to_check) & 0xFFFFFFFF)
########################################################################################


class Vector3(object):
    def __init__(self, x, y, z, h=None):
        self.x = x
        self.y = y
        self.z = z
        if h:
            print "there was an h"
            self.magnitude = math.sqrt(sum([math.pow(x, 2), math.pow(y, 2), math.pow(z, 2), math.pow(h, 2)]))
            self.data = [x, y, z, h]
        else:
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
########################################################################################


class Camera(object):
    def __init__(self, clarity, l_at, l_from, l_up):
        self.clarity = clarity
        self.look_at_direction = l_at
        self.look_from = l_from
        self.look_up = l_up
# ###### DON'T UNDERSTAND######################################################
        self.g = Vector3(0, 0, -1)
        self.t = Vector3(0, 100, 0)
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

########################################################################################

####### DON'T UNDERSTAND######################################################
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
########################################################################################


class Sphere(object):
    def __init__(self, center, radius, reflective_color, diffuse_color):
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


class Triangle(object):
    def __init__(self, first, second, third, reflective_color, diffuse_color, spec_highlight, phong_const):
        self.first = first
        self.second = second
        self.third = third
        self.reflective_color = reflective_color
        self.diffuse_color = diffuse_color
        self.spec_highlight = spec_highlight
        self.phong_const = phong_const


####### DON'T UNDERSTAND######################################################
class Ray(object):
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction.normalize()

    def hits(self, sphere):
        # Tests for intersection with sphere in world space.
        # If no intersection occurs, returns None.
        # Else return multiplier t to reach closest hit point.
        # t is a multiplier as follows: self.origin + (self.direction * t).
        oc = subtract_vectors(sphere.center, self.origin)
        tca = oc.dot(self.direction)
        d = self.find_opposite(tca, oc.magnitude)
        # If the point of closest approach is farther away from the center than
        # the radius is long, there is no hit.
        if d > sphere.radius or d < 0:
            return None
        # Find the distance from the camera to the collision point:
        thc = self.find_opposite(d, sphere.radius)
        t = tca - thc
        # Find the hit point in world space:
        world_hit = add_vectors(vector_mult(self.direction, t), self.origin)
        #Construct the surface normal of the hit point.
        normal = subtract_vectors(world_hit, sphere.center).normalize()
        # Build the hit object to return.
        result = Hit(tca - thc, world_hit, normal, sphere.color)
        return result

    def find_opposite(self, adj, hyp):
        # Use the pythagorean theorem to find the length
        # of the opposite side.
        return math.sqrt(math.pow(hyp, 2) - math.pow(adj, 2))

########################################################################################


####### DON'T UNDERSTAND######################################################
class Hit(object):  # A simple object representing a ray hit with a sphere.
    def __init__(self, dist, world_hit, normal, obj_color):
        # Dist should be a scalar, and normal should be a vector object.
        self.dist = dist
        self.world_hit = world_hit
        self.normal = normal
        self.obj_color = obj_color
########################################################################################


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

    result2 = add_vectors(result, Vector3(camera.v.x * y_cam, camera.v.y * y_cam, camera.v.z * y_cam))
    return Ray(camera.look_from, result2)


def load_camera(at_line, from_line, up_line, angle_line):

    look_at = Vector3(float(at_line[1]), float(at_line[2]), float(at_line[3]))
    look_from = Vector3(float(from_line[1]), float(from_line[2]), float(from_line[3]))
    look_up = Vector3(float(up_line[1]), float(up_line[2]), float(up_line[3]))
    clarity = 200  # whatever we want

    camera = Camera(clarity, look_at, look_from, look_up)
    camera.angle = float(angle_line[1])
    camera.look_at(look_at)
    camera.calc()
    return camera


def load_light_and_colors(direct_line, color_line, ambient_line, background_line):
    dir_to_light = Vector3(float(direct_line[1]), float(direct_line[2]), float(direct_line[3]))
    light_color = [float(color_line[1]), float(color_line[2]), float(color_line[3])]
    ambient_light = Vector3(float(ambient_line[1]), float(ambient_line[2]), float(ambient_line[3]))
    background_color = Vector3(float(background_line[1]), float(background_line[2]), float(background_line[3]))
    return dir_to_light, light_color, ambient_light, background_color


def load_field_of_view(input_file):
    field_of_view = 55
    return field_of_view


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
        spheres.append(Sphere(center, radius, reflective_color, diffuse_color))
    return spheres


def load_triangles(input_file):
    triangles = []
    first = Vector3(0, -0.5, 0.5)
    second = Vector3(1, 0.5, 0)
    third = Vector3(0, -0.5, -0.5)
    reflective = None
    diffuse_color = Vector3(0, 0, 1)
    specular_highlight = Vector3(1, 1, 1)
    phong_constant = 4
    triangle1 = Triangle(first, second, third, reflective, diffuse_color, specular_highlight, phong_constant)
    triangles.append(triangle1)

    first = Vector3(0, -0.5, 0.5)
    second = Vector3(0, -0.5, -0.5)
    third = Vector3(-1, 0.5, 0)
    reflective = None
    diffuse_color = Vector3(1, 1, 0)
    specular_highlight = Vector3(1, 1, 1)
    phong_constant = 4
    triangle2 = Triangle(first, second, third, reflective, diffuse_color, specular_highlight, phong_constant)
    triangles.append(triangle2)

    return triangles


####### DON'T UNDERSTAND######################################################
def diffuse(hit, light):
    # TODO: replace with my diffuse
    color = hit.obj_color
    new_color = [0, 0, 0, 255]
    for i in xrange(3):
        brightness = light.intensity * hit.normal.dot(subtract_vectors(light.position, hit.world_hit).normalize())
        new_color[i] = new_color[i] + max(0, int(color[i] * brightness))
    return new_color
########################################################################################


def load_file_objects(input):
    f = open(input, 'r')
    lines = f.readlines()

    camera = load_camera(lines[0].strip().split(" "),
                         lines[1].strip().split(" "),
                         lines[2].strip().split(" "),
                         lines[3].strip().split(" "))

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

    return camera, light, spheres, triangles


def run():
    input = "diffuse.rayTracing"
    camera, light, spheres, triangles = load_file_objects(input)

    png = PNG(camera.clarity, camera.clarity, [255, 255, 255], [34, 34, 34, 255])
    # Draw background.
    png.filled_rectangle(0, 0, camera.width - 1, camera.height - 1)

    # Draw shapes.
    for i in xrange(0, camera.width - 1):
        for j in xrange(0, camera.height - 1):
            ray = create_ray(camera, i, camera.height - j)
            # This is to tell what's in the front:
            front = None
            for s in spheres:
                hit = ray.hits(s)
                if hit:
                    if not front:
                        front = hit
                    elif hit.dist < front.dist:
                        front = hit
            if front:  # Do some shading:
                png.point(i, j, diffuse(front, light))

    # Create PNG
    f = open("outfile.png", "wb")
    f.write(png.dump())
    f.close()

run()
