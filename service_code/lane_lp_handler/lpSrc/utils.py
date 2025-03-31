import cv2
import numpy as np
import math
import collections
from typing import List, Union, Tuple, Any, Dict
from scipy.spatial import ConvexHull
from collections import OrderedDict


def resize_aspect_ratio(img, long_size, interpolation):
    height, width, channel = img.shape

    # set target image size
    target_size = long_size

    ratio = target_size / max(height, width)

    target_h, target_w = int(height * ratio), int(width * ratio)
    proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

    # make canvas and paste image
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)
    resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
    resized[0:target_h, 0:target_w, :] = proc
    target_h, target_w = target_h32, target_w32

    size_heatmap = (int(target_w / 2), int(target_h / 2))

    return resized, ratio, size_heatmap


def cvt2HeatmapImg(img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img


def normalizeMeanVariance(
    in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)
):
    # should be RGB order
    img = in_img.copy().astype(np.float32)

    img -= np.array(
        [mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32
    )
    img /= np.array(
        [variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0],
        dtype=np.float32,
    )
    return img


def adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net=2):
    filtered_polys = []
    for poly in polys:
        if not(poly is None):
            filtered_polys.append(poly)
    polys = filtered_polys
    if len(polys) > 0:
        polys = np.array(polys)
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
    return polys


def fline(p0: List, p1: List, debug: bool = False) -> List:
    """
    Вычесление угла наклона прямой по 2 точкам
    """
    x1 = float(p0[0])
    y1 = float(p0[1])

    x2 = float(p1[0])
    y2 = float(p1[1])

    if debug:
        print("Уравнение прямой, проходящей через эти точки:")
    if x1 - x2 == 0:
        k = math.inf
        b = y2
    else:
        k = (y1 - y2) / (x1 - x2)
        b = y2 - k*x2
    if debug:
        print(" y = %.4f*x + %.4f" % (k, b))
    r = math.atan(k)
    a = math.degrees(r)
    a180 = a
    if a < 0:
        a180 = 180 + a
    return [k, b, a, a180, r]


def distance(p0, p1) -> float:
    """
    distance between two points p0 and p1
    """
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)


def linear_line_matrix(p0: List, p1: List, verbode: bool = False) -> np.ndarray:
    """
    Вычесление коефициентов матрицы, описывающей линию по двум точкам
    """
    x1 = float(p0[0])
    y1 = float(p0[1])

    x2 = float(p1[0])
    y2 = float(p1[1])

    matrix_a = y1 - y2
    matrix_b = x2 - x1
    matrix_c = x2*y1-x1*y2
    if verbode:
        print("Уравнение прямой, проходящей через эти точки:")
        print("%.4f*x + %.4fy = %.4f" % (matrix_a, matrix_b, matrix_c))
        print(matrix_a, matrix_b, matrix_c)
    return np.array([matrix_a, matrix_b, matrix_c])


def find_distances(points) -> List:
    """
    TODO: describe function
    """
    distanses = []
    cnt = len(points)

    for i in range(cnt):
        p0 = i
        if i < cnt - 1:
            p1 = i + 1
        else:
            p1 = 0
        distanses.append({"d": distance(points[p0], points[p1]), "p0": p0, "p1": p1,
                          "matrix": linear_line_matrix(points[p0], points[p1]),
                          "coef": fline(points[p0], points[p1])})
    return distanses


def build_perspective(img: np.ndarray, rect: list, w: int, h: int) -> List:
    """
    image perspective transformation
    """
    img_h, img_w, img_c = img.shape
    if img_h < h:
        h = img_h
    if img_w < w:
        w = img_w
    pts1 = np.float32(rect)
    pts2 = np.float32(np.array([[0, 0], [w, 0], [w, h], [0, h]]))
    moment = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, moment, (w, h))


def get_cv_zone_rgb(img: np.ndarray, rect: list, gw: float = 0, gh: float = 0,
                    coef: float = 4.6, auto_width_height: bool = True) -> List:
    """
    TODO: describe function
    """
    if gw == 0 or gh == 0:
        distanses = find_distances(rect)
        h = (distanses[0]['d'] + distanses[2]['d']) / 2
        if auto_width_height:
            w = int(h*coef)
        else:
            w = (distanses[1]['d'] + distanses[3]['d']) / 2
    else:
        w, h = gw, gh
    return build_perspective(img, rect, int(w), int(h))


def crop_image(image, target_box):
    x = int(min(target_box[0], target_box[2]))
    w = int(abs(target_box[2] - target_box[0]))
    y = int(min(target_box[1], target_box[3]))
    h = int(abs(target_box[3] - target_box[1]))

    image_part = image[y:y + h, x:x + w]
    return image_part, (x, w, y, h)


def minimum_bounding_rectangle(points: np.ndarray) -> np.ndarray:
    """
    Find the smallest bounding rectangle for a set of points.
    detail: https://gis.stackexchange.com/questions/22895/finding-minimum-area-rectangle-for-given-points
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    pi2 = np.pi / 2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = hull_points[1:] - hull_points[:-1]
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles - pi2),
        np.cos(angles + pi2),
        np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval


def reshape_points(target_points, start_idx: int) -> List:
    """
    TODO: describe function
    """
    if start_idx > 0:
        part1 = target_points[:start_idx]
        part2 = target_points[start_idx:]
        target_points = np.concatenate((part2, part1))
    return target_points


def add_coordinates_offset(points, x: float, y: float) -> List:
    """
    TODO: describe function
    """
    return [[point[0] + x, point[1] + y] for point in points]


def get_y_by_matrix(matrix: np.ndarray, x: float) -> np.ndarray:
    """
    TODO: describe function
    """
    matrix_a = matrix[0]
    matrix_b = matrix[1]
    matrix_c = matrix[2]
    if matrix_b != 0:
        return (matrix_c - matrix_a * x) / matrix_b
    

def add_point_offset(point: List, x: float, y: float) -> List:
    """
    TODO: describe function
    """
    return [point[0] + x, point[1] + y]


def add_point_offsets(points: List, dx: float, dy: float) -> List:
    """
    TODO: describe function
    """
    return [
        add_point_offset(points[0], -dx, -dy),
        add_point_offset(points[1], dx, dy),
        add_point_offset(points[2], dx, dy),
        add_point_offset(points[3], -dx, -dy),
    ]


def make_rect_variants(propably_points: List, quality_profile: List = None) -> List:
    """
    TODO: describe function
    """
    if quality_profile is None:
        quality_profile = [3, 1, 0, 0]

    steps = quality_profile[0]
    steps_plus = quality_profile[1]
    steps_minus = quality_profile[2]
    step = 1
    if len(quality_profile) > 3:
        step_adaptive = quality_profile[3] > 0
    else:
        step_adaptive = False

    distanses = find_distances(propably_points)

    point_centre_left = [propably_points[0][0] + (propably_points[1][0] - propably_points[0][0]) / 2,
                         propably_points[0][1] + (propably_points[1][1] - propably_points[0][1]) / 2]

    if distanses[3]["matrix"][1] == 0:
        return [propably_points]
    point_bottom_left = [point_centre_left[0], get_y_by_matrix(distanses[3]["matrix"], point_centre_left[0])]
    dx = propably_points[0][0] - point_bottom_left[0]
    dy = propably_points[0][1] - point_bottom_left[1]

    dx_step = dx / steps
    dy_step = dy / steps

    if step_adaptive:
        d_max = distance(point_centre_left, propably_points[0])
        dd = math.sqrt(dx ** 2 + dy ** 2)
        steps_all = int(d_max / dd)

        step = int((steps_all * 2) / steps)
        if step < 1:
            step = 1
        steps_minus = steps_all + steps_minus * step
        steps_plus = steps_all + steps_plus * step

    points_arr = []
    for i in range(-steps_minus, steps + steps_plus + 1, step):
        points_arr.append(add_point_offsets(propably_points, i * dx_step, i * dy_step))
    return points_arr


def detect_best_perspective(bw_images: List[np.ndarray]) -> int:
    """
    TODO: describe function
    """
    res = []
    idx = 0
    diff = 1000000
    diff_cnt = 0
    for i, img in enumerate(bw_images):
        s = np.sum(img, axis=0)
        img_stat = collections.Counter(s)
        img_stat_dict = OrderedDict(img_stat.most_common())
        max_stat = max(img_stat_dict, key=int)
        max_stat_count = img_stat_dict[max_stat]
        min_stat = min(img_stat_dict, key=int)
        min_stat_count = img_stat_dict[min_stat]
        res.append({'max': max_stat, 'min': min_stat, 'maxCnt': max_stat_count, 'minCnt': min_stat_count})

        if min_stat < diff:
            idx = i
            diff = min_stat
        if min_stat == diff and max_stat_count + min_stat_count > diff_cnt:
            idx = i
            diff_cnt = max_stat_count + min_stat_count
    return idx


def prepare_image_text(img: np.ndarray) -> np.ndarray:
    """
    сперва переведём изображение из RGB в чёрно серый
    значения пикселей будут от 0 до 255
    """
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray_image = cv2.normalize(gray_image, None, alpha=0, beta=255,
                               norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    (thresh, black_and_white_image) = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    return black_and_white_image


def normalize_perspective_images(images) -> List[np.ndarray]:
    """
    TODO: describe function
    """
    new_images = []
    for img in images:
        new_images.append(prepare_image_text(img))
    return new_images


def get_det_boxes(textmap, linkmap, text_threshold, link_threshold, low_text, use_cpp_bindings=True):
    """
    get det boxes
    """
    # prepare data
    linkmap = linkmap.copy()
    textmap = textmap.copy()
    img_h, img_w = textmap.shape

    """ labeling method """
    ret, text_score = cv2.threshold(textmap, low_text, 1, 0)
    ret, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)

    text_score_comb = np.clip(text_score + link_score, 0, 1)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8),
                                                                          connectivity=4)
    det = []
    mapper = []
    for k in range(1, n_labels):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10:
            continue

        # thresholding
        if np.max(textmap[labels == k]) < text_threshold:
            continue

        # make segmentation map
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels == k] = 255
        segmap[np.logical_and(link_score == 1, text_score == 0)] = 0  # remove link area
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        # boundary check
        if sx < 0:
            sx = 0
        if sy < 0:
            sy = 0
        if ex >= img_w:
            ex = img_w
        if ey >= img_h:
            ey = img_h
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        # make box
        np_contours = np.roll(np.array(np.where(segmap != 0)), 1, axis=0).transpose().reshape(-1, 2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        # align diamond-shape
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
            t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        box = np.array(box)

        det.append(box)
        mapper.append(k)

    return det


def make_offsets(
        bbox: Tuple,
        distanses_offset_left_max_percentage: float,
        offset_top_max_percentage: float,
        offset_right_max_percentage: float,
        offset_bottom_max_percentage: float):

    distanses_offset_left_percentage = distanses_offset_left_max_percentage
    offset_top_percentage = offset_top_max_percentage
    offset_right_percentage = offset_right_max_percentage
    offset_bottom_percentage = offset_bottom_max_percentage

    k = bbox[1] / bbox[0]

    if k < 2:
        offset_top_percentage = offset_top_percentage / 2
        offset_bottom_percentage = offset_bottom_percentage / 2

    if k < 1:
        offset_top_percentage = 0
        offset_bottom_percentage = 0

    offsets = [
        distanses_offset_left_percentage,
        offset_top_percentage,
        offset_right_percentage,
        offset_bottom_percentage
    ]
    return offsets



def detect_intersection(matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
    """
    www.math.by/geometry/eqline.html
    xn--80ahcjeib4ac4d.xn--p1ai/information/solving_systems_of_linear_equations_in_python/
    """
    x = np.array([matrix1[:2], matrix2[:2]])
    y = np.array([matrix1[2], matrix2[2]])
    return np.linalg.solve(x, y)


def detect_intersection_norm_dd(matrix1: np.ndarray, matrix2: np.ndarray, d1: float, d2: float) -> np.ndarray:
    """
    TODO: describe function
    """
    x = np.array([matrix1[:2], matrix2[:2]])
    c0 = matrix1[2] - d1 * (matrix1[0] ** 2 + matrix1[1] ** 2) ** 0.5
    c1 = matrix2[2] - d2 * (matrix2[0] ** 2 + matrix2[1] ** 2) ** 0.5
    y = np.array([c0, c1])
    return np.linalg.solve(x, y)


def detect_distance_from_point_to_line(matrix: List[np.ndarray],
                                       point) -> float:
    """
    Определение растояния от точки к линии
    https://ru.onlinemschool.com/math/library/analytic_geometry/p_line1/
    """
    a = matrix[0]
    b = matrix[1]
    c = matrix[2]
    x = point[0]
    y = point[1]
    return abs(a * x + b * y - c) / math.sqrt(a ** 2 + b ** 2)



def addopt_rect_to_bbox_make_points(
        distanses: List,
        bbox: Tuple,
        distanses_offset_left_max_percentage: float,
        offset_top_max_percentage: float,
        offset_right_max_percentage: float,
        offset_bottom_max_percentage: float):
    points = []
    offsets = make_offsets(bbox,
                           distanses_offset_left_max_percentage,
                           offset_top_max_percentage,
                           offset_right_max_percentage,
                           offset_bottom_max_percentage)

    cnt = len(distanses)
    for i in range(cnt):
        i_next = i + 1
        if i_next == cnt:
            i_next = 0
        offsets[i] = distanses[i_next]['d'] * offsets[i] / 100
    for i in range(cnt):
        i_prev = i
        i_next = i + 1
        if i_next == cnt:
            i_next = 0
        offset1 = offsets[i_prev]
        offset2 = offsets[i_next]
        points.append(
            detect_intersection_norm_dd(distanses[i_prev]['matrix'], distanses[i_next]['matrix'], offset1, offset2))
    return points


def detect_distance_from_point_to_line(matrix: List[np.ndarray],
                                       point) -> float:
    """
    Определение растояния от точки к линии
    https://ru.onlinemschool.com/math/library/analytic_geometry/p_line1/
    """
    a = matrix[0]
    b = matrix[1]
    c = matrix[2]
    x = point[0]
    y = point[1]
    return abs(a * x + b * y - c) / math.sqrt(a ** 2 + b ** 2)


def addopt_rect_to_bbox(target_points: List,
                        bbox: Tuple,
                        distanses_offset_left_max_percentage: float,
                        offset_top_max_percentage: float,
                        offset_right_max_percentage: float,
                        offset_bottom_max_percentage: float) -> np.ndarray:
    """
    TODO: describe function
    """
    distanses = find_distances(target_points)
    points = addopt_rect_to_bbox_make_points(
        distanses,
        bbox,
        distanses_offset_left_max_percentage,
        offset_top_max_percentage,
        offset_right_max_percentage,
        offset_bottom_max_percentage)

    points = reshape_points(points, 3)

    distanses = find_distances(points)

    if distanses[3]['coef'][2] == 90:
        return np.array(points)

    h = bbox[0]
    w = bbox[1]

    matrix_left = linear_line_matrix([0, 0], [0, h])
    matrix_right = linear_line_matrix([w, 0], [w, h])

    p_left_top = detect_intersection(matrix_left, distanses[1]['matrix'])
    p_left_bottom = detect_intersection(matrix_left, distanses[3]['matrix'])
    p_right_top = detect_intersection(matrix_right, distanses[1]['matrix'])
    p_right_bottom = detect_intersection(matrix_right, distanses[3]['matrix'])

    offset_left_bottom = distance(points[0], p_left_bottom)
    offset_left_top = distance(points[1], p_left_top)
    offset_right_top = distance(points[2], p_right_top)
    offset_right_bottom = distance(points[3], p_right_bottom)

    over_left_top = points[1][0] < 0
    over_left_bottom = points[0][0] < 0
    if not over_left_top and not over_left_bottom:
        if offset_left_top > offset_left_bottom:
            points[0] = p_left_bottom
            left_distance = detect_distance_from_point_to_line(distanses[0]['matrix'], p_left_bottom)
            points[1] = detect_intersection_norm_dd(distanses[0]['matrix'], distanses[1]['matrix'], left_distance, 0)
        else:
            points[1] = p_left_top
            left_distance = detect_distance_from_point_to_line(distanses[0]['matrix'], p_left_top)
            points[0] = detect_intersection_norm_dd(distanses[3]['matrix'], distanses[0]['matrix'], 0, left_distance)

    over_right_top = points[2][0] > w
    over_right_bottom = points[3][0] > w
    if not over_right_top and not over_right_bottom:
        if offset_right_top > offset_right_bottom:
            points[3] = p_right_bottom
            right_distance = detect_distance_from_point_to_line(distanses[2]['matrix'], p_right_bottom)
            points[2] = detect_intersection_norm_dd(distanses[1]['matrix'], distanses[2]['matrix'], 0, right_distance)
        else:
            points[2] = p_right_top
            right_distance = detect_distance_from_point_to_line(distanses[2]['matrix'], p_right_top)
            points[3] = detect_intersection_norm_dd(distanses[2]['matrix'], distanses[3]['matrix'], right_distance, 0)

    return np.array(points)


def split_boxes(bboxes: List[Union[np.ndarray, np.ndarray]], dimensions: List[Dict],
                similarity_range: int = 0.5) -> Tuple[List[int], List[int]]:
    """
    TODO: describe function
    """
    np_bboxes_idx = []
    garbage_bboxes_idx = []
    max_dy = 0
    if len(bboxes):
        max_dy = max([dimension['dy'] for dimension in dimensions])
    for i, (bbox, dimension) in enumerate(zip(bboxes, dimensions)):
        if max_dy * similarity_range <= dimension['dy']:
            np_bboxes_idx.append(i)
        else:
            garbage_bboxes_idx.append(i)
    return np_bboxes_idx, garbage_bboxes_idx


def order_points_old(pts: np.ndarray):
    # initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    lp = np.argmin(s)

    # fix original code by Oleg Cherniy
    rp = lp + 2
    if rp > 3:
        rp = rp - 4
    rect[0] = pts[lp]
    rect[2] = pts[rp]
    pts_crop = [pts[idx] for idx in filter(lambda i: (i != lp) and (i != rp), range(len(pts)))]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    # Определяется так.
    # Предположим, у нас есть 3 точки: А(х1,у1), Б(х2,у2), С(х3,у3).
    # Через точки А и Б проведена прямая. И нам надо определить, как расположена точка С относительно прямой АБ.
    # Для этого вычисляем значение:
    # D = (х3 - х1) * (у2 - у1) - (у3 - у1) * (х2 - х1)
    # - Если D = 0 - значит, точка С лежит на прямой АБ.
    # - Если D < 0 - значит, точка С лежит слева от прямой.
    # - Если D > 0 - значит, точка С лежит справа от прямой.
    x1 = rect[0][0]
    y1 = rect[0][1]
    x2 = rect[2][0]
    y2 = rect[2][1]
    x3 = pts_crop[0][0]
    y3 = pts_crop[0][1]
    d = (x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)

    if d > 0:
        rect[1] = pts_crop[0]
        rect[3] = pts_crop[1]
    else:
        rect[1] = pts_crop[1]
        rect[3] = pts_crop[0]

    # return the ordered coordinates
    return rect


def fix_clockwise2(target_points) -> np.ndarray:
    return order_points_old(np.array(target_points))


def find_min_x_idx(target_points) -> int:
    """
    TODO: describe function
    """
    min_x_idx = 3
    for i in range(0, len(target_points)):
        if target_points[i][0] < target_points[min_x_idx][0]:
            min_x_idx = i
        if target_points[i][0] == target_points[min_x_idx][0] and target_points[i][1] < target_points[min_x_idx][1]:
            min_x_idx = i
    return min_x_idx


def normalize_rect(rect: List):
    """
    TODO: describe function
    """
    rect = fix_clockwise2(rect)
    min_x_idx = find_min_x_idx(rect)
    rect = reshape_points(rect, min_x_idx)
    coef_ccw = fline(rect[0], rect[3])
    angle_ccw = round(coef_ccw[2], 2)
    d_bottom = distance(rect[0], rect[3])
    d_left = distance(rect[0], rect[1])
    k = d_bottom / d_left
    if not round(rect[0][0], 4) == round(rect[1][0], 4):
        if d_bottom < d_left:
            k = d_left / d_bottom
            if k > 1.5 or angle_ccw < 0 or angle_ccw > 45:
                rect = reshape_points(rect, 3)
        else:
            if k < 1.5 and (angle_ccw < 0 or angle_ccw > 45):
                rect = reshape_points(rect, 3)
    return rect


def filter_boxes(bboxes: List[Union[np.ndarray, np.ndarray]], dimensions: List[Dict],
                 target_points: Any,
                 np_bboxes_idx: List[int], filter_range: int = 0.7) -> Tuple[List[int], List[int], int]:
    """
    TODO: describe function
    """
    target_points = normalize_rect(target_points)
    dy = distance(target_points[0], target_points[1])
    new_np_bboxes_idx = []
    garbage_bboxes_idx = []
    max_dy = 0
    if len(bboxes):
        max_dy = max([dimension['dy'] for dimension in dimensions])
    for i, (bbox, dimension) in enumerate(zip(bboxes, dimensions)):
        if i in np_bboxes_idx:
            coef = dimension['dy']/max_dy
            if coef > filter_range:
                new_np_bboxes_idx.append(i)
            else:
                boxify_factor = dimension['dx']/dimension['dy']
                dx_offset = round(dimension['dx']/2)
                if bbox[0][0] <= dx_offset and 0.7 < boxify_factor < 1.7:
                    garbage_bboxes_idx.append(i)
                else:
                    new_np_bboxes_idx.append(i)
        else:
            garbage_bboxes_idx.append(i)

    probably_count_lines = round(dy/max_dy)
    probably_count_lines = 1 if probably_count_lines < 1 else probably_count_lines
    probably_count_lines = 3 if probably_count_lines > 3 else probably_count_lines
    return new_np_bboxes_idx, garbage_bboxes_idx, probably_count_lines


def unzip(zipped):
    return list(zip(*zipped))


def crop_number_plate_zones_from_images(images, images_points):
    zones = []
    image_ids = []
    for i, (image, image_points) in enumerate(zip(images, images_points)):
        image_zones = [get_cv_zone_rgb(image, reshape_points(rect, 1)) for rect in image_points]
        for zone in image_zones:
            zones.append(zone)
            image_ids.append(i)
    return zones, image_ids