import math


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


class ppm(object):
    def __init__(self, height, width, background_color):
        self.height = height
        self.width = width
        self.background_color = background_color
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
                self.pixel(x, y, self.background_color)

    def pixel(self, x, y, color=None):
        if x < 0 or y < 0 or x > self.width - 1 or y > self.height - 1:
            return
        if not color:
            color = self.background_color
        self.canvas[y][x] = [color.x, color.y, color.z]

    def output(self):
        output = "P3\n{} {}\n{}\n".format(len(self.canvas), len(self.canvas[0]), 255)
        for row in self.canvas:
            out_row = ""
            for column in row:
                out_row = out_row + " " + str(int(column[0])) + " " + str(int(column[1])) + " " + str(int(column[2])) + " "
            output = "{}{}\n".format(output, out_row)
        return output


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
    def __init__(self, clarity, l_at, l_from, l_up, angle):
        self.clarity = clarity
        self.look_at = l_at
        self.look_from = subtract_vectors(l_from, self.look_at)
        self.look_up = l_up
        self.normal = 2
        self.angle = angle
        self.width = clarity
        self.height = clarity
        self.u = None
        self.v = None
        self.top = None
        self.bottom = None
        self.left = None
        self.right = None
        self.calibrate()

    def calibrate(self):
        self.u = self.look_up.cross(self.look_from)
        self.v = self.look_from.cross(self.u)
        self.top = abs(self.normal) * math.tan(math.radians(self.angle))
        self.bottom = -self.top
        self.right = self.top * self.width / self.height
        self.left = -self.right


class Sphere(object):
    def __init__(self, center, radius, reflective_color, diffuse_color, spec_highlight, phong_const):
        self.center = center
        self.radius = radius
        if reflective_color:
            self.reflective = True
            self.diffuse = False
            self.color = reflective_color
            self.reflective_color = reflective_color
            self.diffuse_color = Vector3(255, 255, 255)
        elif diffuse_color:
            self.reflective = False
            self.diffuse = True
            self.diffuse_color = diffuse_color
            self.color = diffuse_color
            self.reflective_color = Vector3(255, 255, 255)

        self.spec_highlight = spec_highlight
        self.phong_const = phong_const

    def is_sphere(self):
        return True

    def get_normal_v(self, intersecting_point):
        return vector_mult(subtract_vectors(intersecting_point, self.center), (1/self.radius))


class Triangle(object):
    def __init__(self, first, second, third, reflective_color, diffuse_color, spec_highlight, phong_const):
        self.first = first
        self.second = second
        self.third = third
        if reflective_color:
            self.reflective = True
            self.diffuse = False
            self.color = reflective_color
            self.reflective_color = reflective_color
            self.diffuse_color = Vector3(255, 255, 255)
        if diffuse_color:
            self.reflective = False
            self.diffuse = True
            self.color = diffuse_color
            self.diffuse_color = diffuse_color
            self.reflective_color = Vector3(255, 255, 255)

        self.spec_highlight = spec_highlight
        self.phong_const = phong_const

    def is_sphere(self):
        return False


class Ray(object):
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction.normalize()


class Intersection(object):  # A simple object representing a ray intersection with a sphere.
    def __init__(self, dist, intersection_point, normal, shape):
        # Dist should be a scalar, and normal should be a vector object.
        self.dist = dist
        self.intersection_point = intersection_point
        self.normal = normal
        self.shape = shape


class Light(object):
    def __init__(self, position, ambient_light, color, intensity, background_color):
        self.position = position
        self.color = color
        self.intensity = intensity
        self.ambient = ambient_light
        self.background_color = background_color


def create_ray(camera, x_pixel, y_pixel):
    x_cam = camera.left + ((camera.right - camera.left) * (x_pixel) / camera.width)
    y_cam = camera.bottom + ((camera.top - camera.bottom) * (y_pixel) / camera.height)
    result = add_vectors(
        vector_mult(camera.look_from, -camera.normal),
        vector_mult(camera.u, x_cam)
    )
    direction = add_vectors(result, Vector3(camera.v.x * y_cam, camera.v.y * y_cam, camera.v.z * y_cam))
    return Ray(camera.look_from, direction)


def load_camera(clarity, at_line, from_line, up_line, angle_line):

    look_at = Vector3(float(at_line[1]), float(at_line[2]), float(at_line[3]))
    look_from = Vector3(float(from_line[1]), float(from_line[2]), float(from_line[3]))
    look_up = Vector3(float(up_line[1]), float(up_line[2]), float(up_line[3]))
    angle = float(angle_line[1])

    camera = Camera(clarity, look_at, look_from, look_up, angle)
    return camera


def check_for_intersection(ray, shape, c):
    if shape.is_sphere():
        sphere = shape
        x_o, y_o, z_o = ray.origin.get_coords()
        if c.look_from.z > 1:
            x_o, y_o, z_o = ray.origin.normalize().get_coords()
        x_d, y_d, z_d = ray.direction.normalize().get_coords()
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
                t = t0
        if not t:
            return

        intersecting_point = add_vectors(ray.origin, vector_mult(ray.direction, t))
        x_i, y_i, z_i = intersecting_point.get_coords()
        normal = sphere.get_normal_v(intersecting_point)
        undr_sqr = pow(x_o - x_i, 2) + pow(y_o - y_i, 2) + pow(z_o - z_i, 2)
        if undr_sqr < 0:
            undr_sqr *= -1
        distance = pow(undr_sqr, 0.5)

        return Intersection(distance, intersecting_point, normal.normalize(), sphere)

    elif not shape.is_sphere():
        triangle = shape
        u = subtract_vectors(triangle.second, triangle.first)
        v = subtract_vectors(triangle.third, triangle.first)
        normal = u.cross(v)
        if normal.x == 0 and normal.y == 0 and normal.z == 0:
            return
        w_o = subtract_vectors(ray.origin, triangle.second)
        a = -1 * normal.dot(w_o)
        b = normal.dot(ray.direction)
        if abs(b) < 0.0000001:
            return

        r = a / b
        if r < 0:
            return
        intersecting_point = add_vectors(ray.origin, vector_mult(ray.direction, r))

        uu = u.dot(u)
        uv = u.dot(v)
        vv = v.dot(v)
        w = subtract_vectors(intersecting_point, triangle.first)
        wu = w.dot(u)
        wv = w.dot(v)
        d = (uv * uv) - (uu * vv)
        s = ((uv * wv) - (vv * wu)) / d
        if s < 0 or s > 1:
            return
        t = ((uv * wu) - (uu * wv)) / d
        if t < 0 or (s + t) > 1:
            return

        # get distance
        undr_sqr = pow(ray.origin.x - intersecting_point.x, 2) + pow(ray.origin.y - intersecting_point.y, 2) + pow(ray.origin.z - intersecting_point.z, 2)
        if undr_sqr < 0:
            undr_sqr *= -1
        distance = pow(undr_sqr, 0.5)

        return Intersection(distance, intersecting_point, normal.normalize(), triangle)


def load_light_and_colors(direct_line, color_line, ambient_line, background_line):
    dir_to_light = Vector3(float(direct_line[1]), float(direct_line[2]), float(direct_line[3]))
    light_color = Vector3(float(color_line[1]), float(color_line[2]), float(color_line[3]))
    ambient_light = Vector3(float(ambient_line[1]), float(ambient_line[2]), float(ambient_line[3]))
    background_color = Vector3(float(background_line[1])*255, float(background_line[2])*255, float(background_line[3])*255)
    return dir_to_light, light_color, ambient_light, background_color


def load_spheres(sphere_lines):
    # center, radius, reflective_color, diffuse_color
    spheres = []
    for line in sphere_lines:
        center = Vector3(float(line[2]), float(line[3]), float(line[4]))
        radius = float(line[6])
        if line[8] == "Diffuse":
            reflective_color = None
            diffuse_color = Vector3(float(line[9])*255, float(line[10])*255, float(line[11])*255)
            spec_highlight = Vector3(float(line[13]), float(line[14]), float(line[15]))
            phong_const = float(line[17])
        else:
            reflective_color = Vector3(float(line[9]), float(line[10]), float(line[11]))
            diffuse_color = None
            spec_highlight = Vector3(255, 255, 255)
            phong_const = 0
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
            diffuse_color = Vector3(float(line[14])*255, float(line[15])*255, float(line[16])*255)
        else:
            reflective_color = Vector3(float(line[14])*255, float(line[15])*255, float(line[16])*255)
            diffuse_color = None
        spec_highlight = Vector3(float(line[18]), float(line[19]), float(line[20]))
        phong_const = float(line[22])
        triangles.append(Triangle(first, second, third, reflective_color, diffuse_color, spec_highlight, phong_const))
    return triangles


def check_if_in_shadow(intersection, light, shapes, camera):
    ray = Ray(intersection.intersection_point, light.position)
    for shape in shapes:
        inter = check_for_intersection(ray, shape, camera)
        if inter:
            difference = subtract_vectors(intersection.intersection_point, inter.intersection_point)
            difference = abs(difference.x) + abs(difference.y) + abs(difference.z)
            if difference > 0.00000000001:
                return True
    return False


def get_reflection(intersection, light, shapes, camera):
    direction = subtract_vectors(intersection.normal, intersection.intersection_point)
    ray = Ray(intersection.intersection_point, direction)
    for shape in shapes:
        inter = check_for_intersection(ray, shape, camera)
        if inter:
            difference = subtract_vectors(intersection.intersection_point, inter.intersection_point)
            difference = abs(difference.x) + abs(difference.y) + abs(difference.z)
            if difference > 0.00000000001:
                return shape.diffuse_color
    return light.background_color


def calculate_pixel_color(intersection, light, camera, shapes):
    is_in_shadow = check_if_in_shadow(intersection, light, shapes, camera)
    if is_in_shadow:
        c_l = Vector3(0, 0, 0)
    else:
        c_l = Vector3(light.color.x, light.color.y, light.color.z)

    # ###########
    if intersection.shape.reflective:
        shp = intersection.shape
        reflection_color = get_reflection(intersection, light, shapes, camera)
        da_color = Vector3(reflection_color.x * shp.reflective_color.x, reflection_color.y * shp.reflective_color.y, reflection_color.z * shp.reflective_color.z)
    else:
        c_r = Vector3(intersection.shape.diffuse_color.x, intersection.shape.diffuse_color.y, intersection.shape.diffuse_color.z)
        c_a = light.ambient
        ambient = multiply_vectors(c_r, c_a)
        direct = multiply_vectors(c_r, vector_mult(c_l, max(0, intersection.normal.dot(light.position))))

        diffuse_term = add_vectors(ambient, direct)
        # ###########
        c_p = intersection.shape.spec_highlight
        specular = multiply_vectors(c_l, c_p)
        reflective = vector_mult(intersection.normal, 2)
        reflective = vector_mult(reflective, intersection.normal.dot(light.position))
        reflective = subtract_vectors(reflective, light.position)

        eye = subtract_vectors(camera.look_from, intersection.intersection_point)
        phong = max(0, pow(eye.dot(reflective), intersection.shape.phong_const))
        phong_term = vector_mult(specular, phong)
        # ############
        da_color = add_vectors(diffuse_term, phong_term)

    if int(da_color.x) > 255:
        da_color.x = 255
    if int(da_color.y) > 255:
        da_color.y = 255
    if int(da_color.z) > 255:
        da_color.z = 255
    new_color = Vector3(int(da_color.x), int(da_color.y), int(da_color.z))
    return new_color


def load_file_objects(input_file, image_size):
    f = open(input_file, 'r')
    lines = f.readlines()
    camera = load_camera(image_size,
                         lines[0].strip().split(" "),
                         lines[1].strip().split(" "),
                         lines[2].strip().split(" "),
                         lines[3].strip().split(" "))
    dir_to_light, light_color, ambient_light, background_color = load_light_and_colors(
        lines[4].strip().split(" ")[:4],
        lines[4].strip().split(" ")[4:],
        lines[5].strip().split(" "),
        lines[6].strip().split(" ")
    )
    light = Light(dir_to_light, ambient_light, light_color, 1, background_color)

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
    png = ppm(camera.clarity, camera.clarity, background_color)
    return camera, light, spheres, triangles, png


def run(image, image_size):
    camera, light, spheres, triangles, ppm = load_file_objects(image, image_size)
    shapes = []
    for sphere in spheres:
        shapes.append(sphere)
    for triangle in triangles:
        shapes.append(triangle)

    for row_index in xrange(0, camera.width - 1):
        # Run across all pixels in the row
        for column_index in xrange(0, camera.height - 1):
            # draw save pixel at row and column
            ray = create_ray(camera, row_index, camera.height - column_index)
            closest_intersection = None
            for shape in shapes:
                intersection = check_for_intersection(ray, shape, camera)
                if intersection:
                    if not closest_intersection:
                        closest_intersection = intersection
                    elif intersection.dist < closest_intersection.dist:
                        closest_intersection = intersection
            if closest_intersection:  # Image already filled with bg-color, only update if we intersection a shape
                color = calculate_pixel_color(closest_intersection, light, camera, shapes)
                ppm.pixel(row_index, column_index, color)

    # Create PPM
    f = open("outfile.ppm", "wb")
    f.write(ppm.output())
    f.close()

run(image="my_scene.rayTracing", image_size=400)
