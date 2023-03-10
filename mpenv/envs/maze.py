import os
import numpy as np
from gym import spaces
import hppfcl
import pinocchio as pin
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as collections

from mpenv.core.mesh import Mesh
from mpenv.envs.base import Base
from mpenv.envs.maze_generator import Maze
from mpenv.envs import utils as envs_utils
from mpenv.envs.utils import ROBOTS_PROPS
from mpenv.core import utils
from mpenv.core.geometry import Geometries

from mpenv.observers.robot_links import RobotLinksObserver
from mpenv.observers.point_cloud import PointCloudObserver
from mpenv.observers.ray_tracing import RayTracingObserver
from mpenv.observers.maze import MazeObserver


class MazeGoal(Base):
    def __init__(self, grid_size):
        super().__init__(robot_name="sphere")

        self.thickness = 0.02
        self.grid_size = grid_size
        self.robot_name = "sphere"
        self.freeflyer_bounds = np.array(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
        )
        self.robot_props = ROBOTS_PROPS["sphere2d"]
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.robot_props["action_dim"],), dtype=np.float32
        )

        self.fig, self.ax, self.pos = None, None, None

    def _reset(self, idx_env=None, start=None, goal=None):
        model_wrapper = self.model_wrapper
        self.robot = self.add_robot("sphere2d", self.freeflyer_bounds)
        self.geoms, self.idx_env = self.get_obstacles_geoms(idx_env)
        for geom_obj in self.geoms.geom_objs:
            self.add_obstacle(geom_obj, static=True)
        model_wrapper.create_data()

        valid_sample = False
        while not valid_sample:
            self.state = self.random_configuration()
            self.goal_state = self.random_configuration()
            valid_sample = self.validate_sample(self.state, self.goal_state)
        if start is not None:
            self.set_state(start)
        if goal is not None:
            self.set_goal_state(goal)

        if self.fig:
            plt.close()
        self.fig, self.ax, self.pos = None, None, None

        return self.observation()

    def validate_sample(self, state, goal_state):
        "Filter start and goal with straight path solution"
        straight_path = self.model_wrapper.arange(
            state, goal_state, self.delta_collision_check
        )
        _, collide = self.stopping_configuration(straight_path)
        return collide.any()

    def get_obstacles_geoms(self, idx_env):
        np_random = self._np_random
        self.maze = Maze(self.grid_size, self.grid_size)
        self.maze.make_maze()
        geom_objs = extract_obstacles(self.maze, self.thickness)
        geoms = Geometries(geom_objs)
        return geoms, idx_env

    def set_eval(self):
        pass

    def render(self, *unused_args, **unused_kwargs):
        if self.fig is None:
            self.init_matplotlib()
            self.pos = self.ax.scatter(
                self.state.q[0], self.state.q[1], color="orange", s=200
            )
        else:
            prev_pos = self.pos.get_offsets().data
            new_pos = self.state.q[:2][None, :]
            x = np.vstack((prev_pos, new_pos))
            self.ax.plot(x[:, 0], x[:, 1], c=(0, 0, 0, 0.5))
            self.pos.set_offsets(new_pos)
        plt.draw()
        plt.pause(0.01)

    def init_matplotlib(self):
        plt.ion()

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, aspect="equal")
        ax.set_xlim(0.0 - self.thickness, 1.0 + self.thickness)
        ax.set_ylim(0.0 - self.thickness, 1.0 + self.thickness)
        ax.set_xticks([])
        ax.set_yticks([])

        obstacles = self.geoms.geom_objs
        rects = []
        for i, obst in enumerate(obstacles):
            x, y = obst.placement.translation[:2]
            half_side = obst.geometry.halfSide
            w, h = 2 * half_side[:2]
            rects.append(
                patches.Rectangle(
                    (x - w / 2, y - h / 2), w, h  # (x,y)  # width  # height
                )
            )
        coll = collections.PatchCollection(rects, zorder=1)
        coll.set_alpha(0.6)
        ax.add_collection(coll)

        size = self.robot_props["dist_goal"]
        offsets = np.stack((self.state.q, self.goal_state.q), 0)[:, :2]
        sg = collections.EllipseCollection(
            widths=size,
            heights=size,
            facecolors=[(1, 0, 0, 0.8), (0, 1, 0, 0.8)],
            angles=0,
            units="xy",
            offsets=offsets,
            transOffset=ax.transData,
        )
        ax.add_collection(sg)

        plt.tight_layout()
        self.fig = fig
        self.ax = ax


class MazeGoalDistanceCurriculum(MazeGoal):
    def _reset(self, idx_env=None, start=None, goal=None, curriculum_difficulty=1.0):
        self.difficulty = curriculum_difficulty
        self.MAX_PATH_LENGTH = max(self.difficulty * np.sqrt(2), 0.2)
        self.MIN_PATH_LENGTH = max(self.difficulty / 2.0, 0.0) * np.sqrt(2)
        kwargs = {"idx_env": idx_env, "start": start, "goal": goal}
        return super(MazeGoalDistanceCurriculum, self)._reset(**kwargs)

    def validate_sample(self, state, goal_state):
        "Filter start and goal with straight path solution"
        """
        success, path, _, _ = Base.solve_rrt(self, simplify=True, max_iterations=2000)

        if success:
            path_length = 0

            start = path["points"][0].q[:2]
            for point in path["points"][1:]:
                end = point.q[:2]
                path_length += np.sqrt(np.sum(np.power(start - end, 2.0)))
                start = end

            if path_length > self.MAX_PATH_LENGTH:
                print("Rejected environment because path length was too big:")
                print("\t{} > {}".format(path_length, min(self.difficulty + 0.25, 1.0) * self.MAX_PATH_LENGTH))
                return False
        """
        path_length = np.sqrt(np.sum(np.power(state.q[:2] - goal_state.q[:2], 2.0)))

        if path_length > self.MAX_PATH_LENGTH or path_length < self.MIN_PATH_LENGTH:
            return False

        # Allow straight path solutions for the easiest levels of difficulty.
        if self.difficulty > 0.25:
            straight_path = self.model_wrapper.arange(
                state, goal_state, self.delta_collision_check
            )

            _, collide = self.stopping_configuration(straight_path)
            return collide.any()
        else:
            return True


class MazeGoalObstaclesCurriculum(MazeGoal):
    def _reset(self, idx_env=None, start=None, goal=None, curriculum_difficulty=1.0):
        self.difficulty = curriculum_difficulty
        self.HARD_MAX_OBSTACLES = self.grid_size ** 2.0
        self.MAX_OBSTACLES = int(self.difficulty * self.HARD_MAX_OBSTACLES)
        kwargs = {"idx_env": idx_env, "start": start, "goal": goal}
        return super(MazeGoalObstaclesCurriculum, self)._reset(**kwargs)

    def validate_sample(self, state, goal_state):
        "Filter start and goal with straight path solution"
        # Allow straight path solutions for the easiest levels of difficulty.
        if self.difficulty < 0.25:
            return True

        straight_path = self.model_wrapper.arange(
            state, goal_state, self.delta_collision_check
        )

        _, collide = self.stopping_configuration(straight_path)
        return collide.any()

    def get_obstacles_geoms(self, idx_env):
        np_random = self._np_random
        self.maze = Maze(self.grid_size, self.grid_size)
        self.maze.make_maze()
        geom_objs = extract_obstacles(self.maze, self.thickness)
        geoms = Geometries(geom_objs)

        # Truncating obstacles to maximum depending on the curriculum difficulty.
        # We need to keep the at least (4 * grid_size) walls surrounding the whole maze.
        geoms.geom_objs = geoms.geom_objs[:4 * self.grid_size + self.MAX_OBSTACLES]
        return geoms, idx_env


def extract_obstacles(maze, thickness):
    scx = 1 / maze.nx
    scy = 1 / maze.ny

    obstacles_coord = []
    # Draw top and left walls.
    for x in range(maze.nx):
        obstacles_coord.append((x / maze.nx, 0, (x + 1) / maze.nx, 0))
    for y in range(maze.ny):
        obstacles_coord.append((0, y / maze.ny, 0, (y + 1) / maze.ny))

    # Draw right and bottom walls.
    for x in range(maze.nx):
        obstacles_coord.append((x / maze.nx, 1.0, (x + 1) / maze.nx, 1.0))
    for y in range(maze.ny):
        obstacles_coord.append((1.0, y / maze.ny, 1.0, (y + 1) / maze.ny))

    # Draw the "South" and "East" walls of each cell, if present (these
    # are the "North" and "West" walls of a neighbouring cell in
    # general, of course).
    # We ignore last row & last column of cells since the right & bottom walls have already been drawn.
    for x in range(maze.nx - 1):
        for y in range(maze.ny - 1):
            if maze.cell_at(x, y).walls["S"]:
                x1, y1, x2, y2 = (
                    x * scx,
                    (y + 1) * scy,
                    (x + 1) * scx,
                    (y + 1) * scy,
                )
                obstacles_coord.append((x1, y1, x2, y2))
            if maze.cell_at(x, y).walls["E"]:
                x1, y1, x2, y2 = (
                    (x + 1) * scx,
                    y * scy,
                    (x + 1) * scx,
                    (y + 1) * scy,
                )
                obstacles_coord.append((x1, y1, x2, y2))
    obstacles = []
    for i, obst_coord in enumerate(obstacles_coord):
        x1, y1, x2, y2 = obst_coord[0], obst_coord[1], obst_coord[2], obst_coord[3]
        x1 -= thickness / 2
        x2 += thickness / 2
        y1 -= thickness / 2
        y2 += thickness / 2
        box_size = [x2 - x1, y2 - y1, 0.1]
        pos = [(x1 + x2) / 2, (y1 + y2) / 2, 0]
        placement = pin.SE3(np.eye(3), np.array(pos))
        mesh = Mesh(
            name=f"obstacle{i}",
            geometry=hppfcl.Box(*box_size),
            placement=placement,
            color=(0, 0, 1, 0.8),
        )
        obstacles.append(mesh.geom_obj())

    return obstacles


def maze_edges(grid_size):
    env = MazeGoal(grid_size)
    env = MazeObserver(env)
    coordinate_frame = "local"
    env = RobotLinksObserver(env, coordinate_frame)
    return env


def maze_edges_distance_curriculum(grid_size):
    env = MazeGoalDistanceCurriculum(grid_size)
    env = MazeObserver(env)
    coordinate_frame = "local"
    env = RobotLinksObserver(env, coordinate_frame)
    return env


def maze_edges_obstacles_curriculum(grid_size):
    env = MazeGoalObstaclesCurriculum(grid_size)
    env = MazeObserver(env)
    coordinate_frame = "local"
    env = RobotLinksObserver(env, coordinate_frame)
    return env


def maze_raytracing(n_samples, n_rays):
    env = MazeGoal(grid_size=3)
    visibility_radius = 0.7
    memory_distance = 0.06
    env = RayTracingObserver(env, n_samples, n_rays, visibility_radius, memory_distance)
    coordinate_frame = "local"
    env = RobotLinksObserver(env, coordinate_frame)
    return env
