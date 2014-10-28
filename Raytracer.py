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
#         self.e = Vector3(0, 0, 0)
#         self.g = Vector3(0, 0, -1)
#         t = ray-sphere intersection
#         self.t = Vector3(0, 100, 0)
        self.e = Vector3(0, 0, 0)
        self.g = Vector3(0, 0, -1)
        # t = ray-sphere intersection
        self.t = Vector3(0, 100, 0)
        # n = normal?
        self.n = 2
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
        self.top = abs(self.n) * math.tan(math.radians(self.angle / 2))
        self.bottom = -self.top
        self.right = self.top * self.width / self.height
        self.left = -self.right

    def ray(self, x_pixel, y_pixel):
        x_cam = self.left + ((self.right - self.left) * (x_pixel + 0.5) / self.width)
        y_cam = self.bottom + ((self.top - self.bottom) * (y_pixel + 0.5) / self.height)
        result = add_vectors(Vector3(self.w.x * -self.n, self.w.y * -self.n, self.w.z * -self.n),
                             Vector3(self.u.x * x_cam, self.u.y * x_cam, self.u.z * x_cam))

        result2 = add_vectors(result, Vector3(self.v.x * y_cam, self.v.y * y_cam, self.v.z * y_cam))
        # result.normalize()
        return Ray(self.e, result2)

    def look_at(self, x, y, z):
        point = Vector3(x, y, z)
        direction = subtract_vectors(point, self.e)
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


def load_camera(input_file):
    look_at = Vector3(0, -0.2, 0)
    look_from = Vector3(0, 0.2, 1.2)
    look_up = Vector3(0, 1, 0)
    # whatever we want
    clarity = 200
    return Camera(clarity, look_at, look_from, look_up)


def load_light_and_colors(input_file):
    dir_to_light = Vector3(0, 1, 0)
    light_color = [1, 1, 1]
    ambient_light = Vector3(0, 0, 0)
    background_color = Vector3(0.2, 0.2, 0.2)

    # TODO: remove next two lines
    dir_to_light = Vector3(0, 55, -4)
    light_color = [255, 255, 255]
    return dir_to_light, light_color, ambient_light, background_color


def load_field_of_view(input_file):
    field_of_view = 55
    return field_of_view


def load_spheres(input_file):
    spheres = []
    center = Vector3(0, 0.3, 0)
    radius = 0.2
    reflective = Vector3(0.75, 0.75, 0.75)
    diffuse_color = None
    # sphere1 = Sphere(center, radius, reflective, diffuse_color)
    # spheres.append(sphere1)

    # TODO: remove this later on
    spheres = [
        # MSphere((0, 0, -4), 1, [0xff, 0x00, 0x00, 0xff]),
        # MSphere((-2.5, 0, -4), 1, [0x00, 0xff, 0x00, 0xff]),
        # MSphere((-1, 0.5, -3), 1, [0xff, 0xff, 0x00, 0xff]),
        Sphere(Vector3(2.5, -0.7, -4), 1, None, [0, 0, 255, 255])]
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


####### DON'T UNDERSTAND######################################################
def shoot_ray(x, y, camera):
    x_cam = camera.left + ((camera.right - camera.left) * (x + 0.5) / camera.width)
    y_cam = camera.bottom + ((camera.top - camera.bottom) * (y + 0.5) / camera.height)

    negative = Vector3(-camera.n, -camera.n, -camera.n)
    mult = multiply_vectors(camera.w, negative)
    mult2 = multiply_vectors(camera.u, Vector3(x_cam, x_cam, x_cam))
    mult3 = multiply_vectors(camera.v, Vector3(y_cam, y_cam, y_cam))
    add = add_vectors(mult, mult2)
    direction = add_vectors(add, mult3)
    return Ray(camera.e, direction)
########################################################################################


def run():
    input = None
    c = load_camera(input)
    c.e = Vector3(-2, 5, -1)
    c.look_at(0, 0, -4)
    c.angle = 60
    c.calc()

    # triangles = load_triangles(input)
    spheres = load_spheres(input)

    dir_to_light, light_color, ambient_light, background_color = load_light_and_colors(input)
    light = Light(dir_to_light, light_color, 1)

    png = PNG(200, 200, [255, 255, 255], [34, 34, 34, 255])
    # Draw BG.
    png.filled_rectangle(0, 0, c.width - 1, c.height - 1)
    # Draw shapes.
    for i in xrange(0, c.width - 1):
        for j in xrange(0, c.height - 1):
            ray = c.ray(i, c.height - j)
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
