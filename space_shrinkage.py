from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import math


def generate_radius_sequence(start_value, step_value, length):
    """
    :param start_value: smallest radius from center
    :param step_value: ratio from radius i+1 / radius i
    :param length: length of the sequence generated
    :return: list contains generated sequences of radius

    Generate sequence of radius, the different between two
    radius should be increasing since normal distribution
    has steeper curve when closer to mean
    """

    temp = start_value
    r_list = [temp]
    for i in range(1, length):
        temp += step_value * i
        r_list.append(temp)
    return r_list


def generate_circle(center_x, center_y, spin_angle, radius):
    """
    :param center_x: float or int, x coordinate for center
    :param center_y: float or int, y coordinate for center
    :param spin_angle: multiple fractions of pi, like: math.pi/6
    :param radius: float or int, radius of generated circle
    :return: x_list, y_list, lists contains x and y coordinates for
    generate points

    Generate circle of points that represent normal distribution
    observations
    """
    current_angle = 0
    x_list = []
    y_list = []

    while current_angle < 2 * math.pi:
        x_temp = center_x + math.cos(current_angle) * radius
        y_temp = center_y + math.sin(current_angle) * radius
        current_angle += spin_angle
        x_list.append(x_temp)
        y_list.append(y_temp)

    return x_list, y_list


def james_stein_shrinkage_2d(x_list, y_list):
    """
    :param x_list: list, contains points' x-coordinate
    :param y_list: list, contains points' y-coordinate
    :return: x y updated lists contains after-shrinkage
    coordinates

    Shrinkage Coefficient:
    1 - 1/||x||_l2 norm
    """
    assert len(x_list) == len(y_list)
    updated_x_list = []
    updated_y_list = []

    for idx in range(len(x_list)):
        x_temp = x_list[idx]
        y_temp = y_list[idx]
        shrinkage_coeff = 1 - 1 / (x_temp ** 2 + y_temp ** 2)
        updated_x_list.append(shrinkage_coeff * x_temp)
        updated_y_list.append(shrinkage_coeff * y_temp)
    return updated_x_list, updated_y_list


def illustration_2d(shrink_flag=False):
    """
    :param shrink_flag: bool, whether to perform J-S Shrinkage
    :return: None, generate 2D J-S shrinkage plot
    """
    r_list = generate_radius_sequence(1, 1 / 2, 4)
    center_x = 1
    center_y = 1
    theta = math.pi / 16
    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']

    # plot points
    # points will be performed shrinkage if
    # shrink_flag is True
    for idx in range(len(r_list)):
        r = r_list[idx]
        c = color_list[idx]
        x, y = generate_circle(center_x, center_y, theta, r)
        if shrink_flag:
            x, y = james_stein_shrinkage_2d(x, y)
        plt.scatter(x, y, c=c, s=6)

    # plot divergence area ball
    area_center = (center_x / 2, center_y / 2)
    area_radius = ((center_x / 2) ** 2 + (center_y / 2) ** 2) ** (1 / 2)

    ax = plt.gca()
    ax.add_patch(plt.Circle(area_center, area_radius, fill=False, edgecolor='b'))
    plt.scatter(center_x, center_y, s=10)
    ax.set_aspect('equal', adjustable='box')
    plt.xlim([-4, 6])
    plt.ylim([-4, 6])
    plt.grid()
    plt.show()


def generate_ball(center_x, center_y, center_z, spin_angle_xy, spin_angle_z, radius):
    """
    :param center_x: numerical, center's x coordinate for the ball
    :param center_y: numerical, center's y coordinate for the ball
    :param center_z: numerical, center's z coordinate for the ball
    :param spin_angle_xy: numerical, spin angle for dot rotate around xy plane
    :param spin_angle_z: numerical, spin angle for xy plane dot circle by z axis
    :param radius: numerical, radius for the generated ball
    :return: x_list, y_list, z_list, contains x y z coordinates of generated points

    Generate 3d balls of points that represent observations from 3 independent
    gaussian distribution
    """
    current_angle_z = - math.pi / 2
    x_list = []
    y_list = []
    z_list = []

    while current_angle_z < math.pi / 2:
        current_angle_xy = 0
        while current_angle_xy < 2 * math.pi:
            x_temp = center_x + math.cos(current_angle_z) \
                     * math.cos(current_angle_xy) * radius
            y_temp = center_y + math.cos(current_angle_z) \
                     * math.sin(current_angle_xy) * radius
            z_temp = center_z + math.sin(current_angle_z) \
                     * radius
            current_angle_xy += spin_angle_xy
            x_list.append(x_temp)
            y_list.append(y_temp)
            z_list.append(z_temp)
        current_angle_z += spin_angle_z

    return x_list, y_list, z_list


def james_stein_shrinkage_3d(x_list, y_list, z_list):
    """
    :param x_list: list, contains points' x-coordinate
    :param y_list: list, contains points' y-coordinate
    :param z_list: list, contains points' z-coordinate
    :return: x y z updated lists contains after-shrinkage
    coordinates

    Shrinkage Coefficient:
    1 - 1/||x||_l2 norm
    """
    assert len(x_list) == len(y_list)
    assert len(x_list) == len(z_list)
    updated_x_list = []
    updated_y_list = []
    updated_z_list = []

    for idx in range(len(x_list)):
        x_temp = x_list[idx]
        y_temp = y_list[idx]
        z_temp = z_list[idx]
        shrinkage_coeff = 1 - 1 / (x_temp ** 2 + y_temp ** 2 + z_temp ** 2)
        updated_x_list.append(shrinkage_coeff * x_temp)
        updated_y_list.append(shrinkage_coeff * y_temp)
        updated_z_list.append(shrinkage_coeff * z_temp)
    return updated_x_list, updated_y_list, updated_z_list


def illustration_3d(shrink_flag=False):
    """
    :param shrink_flag: bool, whether to perform J-S Shrinkage
    :return: None, generate 3D J-S shrinkage plot
    """
    center_x = 1
    center_y = 1
    center_z = 1
    spin_angle_xy = math.pi / 10
    spin_angle_z = math.pi / 10
    r_list = generate_radius_sequence(1, 1 / 2, 3)
    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')

    # plot points
    # points will be performed shrinkage if
    # shrink_flag is True
    for idx in range(len(r_list)):
        r = r_list[idx]
        c = color_list[idx]
        x, y, z = generate_ball(center_x, center_y, center_z,
                                spin_angle_xy, spin_angle_z, r)
        if shrink_flag:
            x, y, z = james_stein_shrinkage_3d(x, y, z)
        ax.scatter(x, y, z, c=c)

    # plot divergence area ball
    area_center = (center_x / 2, center_y / 2, center_z / 2)
    area_radius = ((center_x / 2) ** 2 + (center_y / 2) ** 2 + (center_z / 2) ** 2) ** (1 / 2)
    u, v = np.mgrid[0:2 * np.pi:50j, 0:np.pi:50j]
    area_x = area_radius * np.cos(u) * np.sin(v)
    area_y = area_radius * np.sin(u) * np.sin(v)
    area_z = area_radius * np.cos(v)
    ax.plot_surface(area_x + area_center[0], area_y + area_center[1], area_z + area_center[2], alpha=0.3)

    ax.set_xlim(-2, 3)
    ax.set_ylim(-2, 3)
    ax.set_zlim(-2, 3)
    plt.show()


if __name__ == "__main__":
    illustration_2d(shrink_flag=False)
    illustration_2d(shrink_flag=True)

    illustration_3d(shrink_flag=False)
    illustration_3d(shrink_flag=True)
