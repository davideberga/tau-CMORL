import torch
import numpy as np
from  .env_utils import soft_step_hard, uniform_tensor, rand_choice_tensor


class DynamicsSimulator:
    def __init__(
        self,
        wait_for_charging: float,
        area_h: float,
        area_w: float,
        squared_area: bool,
        beam_angles: torch.Tensor,
        device: str,
        close_thres: float,
    ):
        # Configuration parameters
        self.hold_t = None
        self.device = device
        self.dt = 0.2
        self.rover_max_velocity = 10.0
        self.rover_min_velocity = 0.0
        self.wait_for_charging = wait_for_charging
        self.close_thres = close_thres
        self.enough_close_to_charger = close_thres
        self.beam_angles = beam_angles.to(device)

        # Environment dimensions
        self.area_h = area_h
        self.area_w = area_w
        self.max_range_destination = area_h if squared_area else max(area_h, area_w)
        self.max_range_lidar = area_h / 2 if squared_area else min(area_h, area_w)
        self.epsilon = 0.2

        # Task config
        self.hold_t = wait_for_charging
        self.close_thres = close_thres
        self.enough_close_to_charger = close_thres
        self.battery_charge = 5
        self.dt = 0.2

    def generate_walls(self):
        walls_w = 3.0

        walls = []
        walls.append(np.array([[0.0, -10], [-walls_w, -10], [-walls_w, 20], [0, 20]]))
        walls.append(np.array([[0.0, 0.0], [10, 0], [10, -walls_w], [0, -walls_w]]))
        walls.append(
            np.array([[10.0 + walls_w, -10], [10, -10], [10, 20], [10 + walls_w, 20]])
        )
        walls.append(
            np.array([[0.0, 10], [10, 10], [10, 10.0 + walls_w], [0.0, 10 + walls_w]])
        )

        return walls

    def generate_objects(self):
        obs_w = 3.0
        objs_np = [np.array([[0.0, 0.0], [10, 0], [10, 10], [0, 10]])]  # map
        objs_np.append(
            np.array([[0.0, 0.0], [obs_w, 0], [obs_w, obs_w], [0, obs_w]])
        )  # first obstacle
        objs_np.append(
            objs_np[1] + np.array([[5 - obs_w / 2, 10 - obs_w]])
        )  # second obstacle (top-center)
        objs_np.append(
            objs_np[1] + np.array([[10 - obs_w, 0]])
        )  # third obstacle (bottom-right)
        objs_np.append(
            objs_np[1] / 2 + np.array([[5 - obs_w / 4, 5 - obs_w / 4]])
        )  # forth obstacle (center-center, shrinking)

        def to_torch(x, device):
            return torch.from_numpy(x).float().to(device)

        # Set walls for lidar

        objs_np.extend(self.generate_walls())

        objs = [to_torch(ele, self.device) for ele in objs_np]
        objs_t1 = [ele.unsqueeze(0).unsqueeze(0) for ele in objs]
        objs_t2 = [torch.roll(ele, shifts=-1, dims=2) for ele in objs_t1]

        return objs_np, objs, objs_t1, objs_t2

    def generate_objects_hard(self):
        obs_w = 3.0
        objs_np = [np.array([[0.0, 0.0], [10, 0], [10, 10], [0, 10]])]  # map
        objs_np.append(
            np.array([[0.0, 10.0], [obs_w, 10], [obs_w, 10 - obs_w], [0, 10 - obs_w]])
        )  # first obstacle
        objs_np.append(
            np.array(
                [
                    [10 - obs_w, 10.0],
                    [10, 10],
                    [10, 10 - obs_w],
                    [10 - obs_w, 10 - obs_w],
                ]
            )
        )
        objs_np.append(
            np.array(
                [
                    [5 - obs_w / 2, obs_w],
                    [5 + obs_w / 2, 0],
                    [5 + obs_w / 2, obs_w],
                    [5 - obs_w / 2, 0],
                ]
            )
        )
        objs_np.append(
            np.array(
                [
                    [5 - obs_w / 3, 4],
                    [5 + obs_w / 3, 5],
                    [5 + obs_w / 3, 5],
                    [5 - obs_w / 3, 5],
                ]
            )
        )
        objs_np.append(
            np.array(
                [
                    [8 - obs_w / 3, 6],
                    [8 + obs_w / 3, 6],
                    [8 + obs_w / 3, 4],
                    [8 - obs_w / 3, 4],
                ]
            )
        )
        objs_np.append(
            np.array(
                [
                    [2 - obs_w / 3, 6],
                    [2 + obs_w / 3, 6],
                    [2 + obs_w / 3, 4],
                    [2 - obs_w / 3, 4],
                ]
            )
        )

        def to_torch(x, device):
            return torch.from_numpy(x).float().to(device)

        # Set walls for lidar

        objs_np.extend(self.generate_walls())

        objs = [to_torch(ele, self.device) for ele in objs_np]
        objs_t1 = [ele.unsqueeze(0).unsqueeze(0) for ele in objs]
        objs_t2 = [torch.roll(ele, shifts=-1, dims=2) for ele in objs_t1]

        return objs_np, objs, objs_t1, objs_t2

    def generate_objects_walls_only(self):
        obs_w = 3.0
        objs_np = [np.array([[0.0, 0.0], [10, 0], [10, 10], [0, 10]])]  # map

        def to_torch(x, device):
            return torch.from_numpy(x).float().to(device)

        # Set walls for lidar
        objs_np.extend(self.generate_walls())

        objs = [to_torch(ele, self.device) for ele in objs_np]
        objs_t1 = [ele.unsqueeze(0).unsqueeze(0) for ele in objs]
        objs_t2 = [torch.roll(ele, shifts=-1, dims=2) for ele in objs_t1]

        return objs_np, objs, objs_t1, objs_t2

    def static_chargers(self):
        chargers = [[1, 7]]  # [5, 1], [8, 9]]
        return torch.tensor(chargers).to(self.device)

    def static_chargers_hard(self):
        chargers = [[2, 2], [8, 2], [5, 8]]
        return torch.tensor(chargers).to(self.device)

    def closest_charger(self, from_position: torch.Tensor, chargers: torch.Tensor):
        # from_position: (N, 2)
        # chargers: (M, 2)
        from_position = from_position.unsqueeze(1)  # (N,1,2)
        chargers = chargers.unsqueeze(0)  # (1,M,2)
        d2 = torch.sum((chargers - from_position) ** 2, dim=2)  # (N,M)
        idx = torch.argmin(d2, dim=1)  # (N,)
        return chargers.squeeze(0)[idx].to(self.device)  # (N,2)

    def transform_objects(self, objs):
        result = []
        for obj in objs:
            min_xy = obj.min(dim=0).values
            max_xy = obj.max(dim=0).values
            center = (min_xy + max_xy) / 2
            size = max_xy - min_xy
            result.append(torch.cat([center, size]))
        return torch.stack(result)

    def ray_rect_intersection(
        self, ray_origins, ray_directions, world_objects, max_range
    ):
        """
        Exact ray–axis-aligned rectangle intersection (non-differentiable).
        """
        centers = world_objects[:, :2]
        widths = world_objects[:, 2].unsqueeze(1)
        heights = world_objects[:, 3].unsqueeze(1)
        min_xy = centers - torch.cat([widths / 2, heights / 2], dim=1)
        max_xy = centers + torch.cat([widths / 2, heights / 2], dim=1)

        B, num_beams, _ = ray_origins.shape
        N = world_objects.shape[0]

        ray_origins_exp = ray_origins.unsqueeze(2).expand(B, num_beams, N, 2)
        ray_directions_exp = ray_directions.unsqueeze(2).expand(B, num_beams, N, 2)

        min_xy_exp = min_xy.unsqueeze(0).unsqueeze(0).expand(B, num_beams, N, 2)
        max_xy_exp = max_xy.unsqueeze(0).unsqueeze(0).expand(B, num_beams, N, 2)

        inv_dir = 1.0 / (ray_directions_exp + 1e-9)  # Prevent division by zero

        t1 = (min_xy_exp - ray_origins_exp) * inv_dir
        t2 = (max_xy_exp - ray_origins_exp) * inv_dir

        tmin = torch.maximum(
            torch.minimum(t1[..., 0], t2[..., 0]), torch.minimum(t1[..., 1], t2[..., 1])
        )
        tmax = torch.minimum(
            torch.maximum(t1[..., 0], t2[..., 0]), torch.maximum(t1[..., 1], t2[..., 1])
        )

        valid = (tmax >= tmin) & (tmax > 0)
        t_intersections = torch.where(
            valid, torch.clamp(tmin, min=0.0, max=max_range), max_range
        )

        return t_intersections  # shape: (B, num_beams, N)

    def simulate_lidar_exact(self, robot_pose, world_objects):
        # Ensure robot_pose is 2D (B, 3)
        if robot_pose.dim() == 1:
            robot_pose = robot_pose.unsqueeze(0)  # Convert (3,) to (1, 3)

        B = robot_pose.shape[0]
        num_beams = self.beam_angles.shape[0]

        global_angles = robot_pose[:, 2].unsqueeze(1) + self.beam_angles.unsqueeze(0)
        ray_dirs = torch.stack(
            [torch.cos(global_angles), torch.sin(global_angles)], dim=-1
        )
        ray_origins = robot_pose[:, :2].unsqueeze(1).expand(B, num_beams, 2)

        intersections = self.ray_rect_intersection(
            ray_origins, ray_dirs, world_objects, self.max_range_lidar
        )
        min_intersections = intersections.min(dim=-1).values  # shape: (B, num_beams)

        centers = world_objects[:, :2]
        widths = world_objects[:, 2]
        heights = world_objects[:, 3]
        min_xy = centers - torch.stack([widths / 2, heights / 2], dim=-1)
        max_xy = centers + torch.stack([widths / 2, heights / 2], dim=-1)

        robot_pos = robot_pose[:, :2].unsqueeze(1)  # shape: (B, 1, 2)
        inside = (
            (robot_pos >= min_xy.unsqueeze(0)) & (robot_pos <= max_xy.unsqueeze(0))
        ).all(dim=-1)
        inside_any = inside.any(dim=-1)  # shape: (B,)

        scan = torch.where(
            inside_any.unsqueeze(1),
            torch.full_like(min_intersections, 1e-3),
            min_intersections,
        )

        return scan.squeeze(0) if scan.shape[0] == 1 else scan

    def estimate_destination(self, from_pose: torch.Tensor, to_position: torch.Tensor):
        # from_pose: (N,3) -> (x, y, theta)
        # to_position: (N,2)
        dx = to_position[:, 0] - from_pose[:, 0]
        dy = to_position[:, 1] - from_pose[:, 1]
        dist = torch.sqrt(dx**2 + dy**2)
        dist = torch.clamp(dist / self.max_range_destination, max=1.0)
        angle = torch.atan2(dy, dx)  # - from_pose[:, 2]
        return dist, angle

    def sample_valid_positions(self, n, objs, min_clearance=0.5, max_tries=1000):
        """
        Sample positions inside the map and at least min_clearance away from obstacles.
        Returns tensor of shape (n, 2).
        """
        positions = []
        tries = 0

        # Get map bounds (assuming objs[0] is the map)
        map_xmin, map_xmax = torch.min(objs[0][:, 0]), torch.max(objs[0][:, 0])
        map_ymin, map_ymax = torch.min(objs[0][:, 1]), torch.max(objs[0][:, 1])

        while len(positions) < n and tries < max_tries:
            # Sample uniformly in the map bounding box
            x = torch.rand((n, 1)).to(self.device) * (map_xmax - map_xmin) + map_xmin
            y = torch.rand((n, 1)).to(self.device) * (map_ymax - map_ymin) + map_ymin
            pos = torch.cat([x, y], dim=1)

            # Validate: inside map & outside all obstacles
            valids = []
            for obj in objs[1:]:  # obstacles only
                obs_xmin, obs_xmax = (
                    torch.min(obj[:, 0]) - min_clearance,
                    torch.max(obj[:, 0]) + min_clearance,
                )
                obs_ymin, obs_ymax = (
                    torch.min(obj[:, 1]) - min_clearance,
                    torch.max(obj[:, 1]) + min_clearance,
                )

                not_in_obs = torch.logical_not(
                    (pos[:, 0] >= obs_xmin)
                    & (pos[:, 0] <= obs_xmax)
                    & (pos[:, 1] >= obs_ymin)
                    & (pos[:, 1] <= obs_ymax)
                )
                valids.append(not_in_obs)

            if valids:
                valid_mask = torch.stack(valids, dim=-1).all(dim=-1)
            else:
                valid_mask = torch.ones(len(pos), dtype=torch.bool)

            # Keep only valid positions
            valid_pos = pos[valid_mask]
            positions.extend(valid_pos[: (n - len(positions))].tolist())
            tries += 1

        return torch.tensor(positions[:n])

    def initialize_x(self, n, objs, chargers, test=False):
        if not test:
            self.epsilon = 0

        x_list = []
        x_theta = []
        total_n = 0
        while total_n < n:
            x_init, thetas = self.initialize_x_cycle(n, objs, test)
            valids = []
            for obj_i, obj in enumerate(objs):
                obs_cpu = obj.detach().cpu()
                xmin, xmax, ymin, ymax = (
                    torch.min(obs_cpu[:, 0]),
                    torch.max(obs_cpu[:, 0]),
                    torch.min(obs_cpu[:, 1]),
                    torch.max(obs_cpu[:, 1]),
                )

                for x, y in [
                    (x_init[:, 0], x_init[:, 1]),
                    (x_init[:, 2], x_init[:, 3]),
                    (x_init[:, 4], x_init[:, 5]),
                ]:
                    if obj_i == 0:  # in map
                        val = torch.logical_and(
                            (x - xmin) * (xmax - x) >= 0,
                            (y - ymin) * (ymax - y) >= 0,
                        )
                    else:  # avoid obstacles
                        val = torch.logical_not(
                            torch.logical_and(
                                (x - (xmin - self.epsilon))
                                * ((xmax + self.epsilon) - x)
                                >= 0,
                                (y - (ymin - self.epsilon))
                                * ((ymax + self.epsilon) - y)
                                >= 0,
                            )
                        )
                    valids.append(val)

            valids = torch.stack(valids, dim=-1)
            valids_indices = torch.where(torch.all(valids, dim=-1) == True)[0]
            x_val = x_init[valids_indices]
            total_n += x_val.shape[0]
            x_list.append(x_val)
            x_theta.append(thetas[valids_indices])

        x_list = torch.cat(x_list, dim=0)[:n]
        x_theta = torch.cat(x_theta, dim=0)[:n]

        tensor_objs_cx_cy_w_h = (
            self.transform_objects(objs).clone().detach().float().to(self.device)
        )

        # Reduce dimension of objects, distance from objects
        tensor_objs_cx_cy_w_h[:, 2] += self.epsilon * 2
        tensor_objs_cx_cy_w_h[:, 3] += self.epsilon * 2

        # Remove map from obstacles
        obstacles = tensor_objs_cx_cy_w_h[1:]

        robot_pose = torch.cat((x_list[:, :2], x_theta), dim=1).float().to(self.device)
        target_position = x_list[:, 2:4].float().to(self.device)
        charger_position = x_list[:, 4:6].float().to(self.device)
        battery_time_hold = x_list[:, 6:].float().to(self.device)
        target_dist, target_angle = self.estimate_destination(
            robot_pose, target_position
        )
        # charger_dist, charger_angle = self.estimate_destination(
        #     robot_pose, self.closest_charger(robot_pose[..., 0:2], chargers)
        # )

        charger_dist, charger_angle = self.estimate_destination(
            robot_pose, charger_position
        )

        scan = self.simulate_lidar_exact(robot_pose, obstacles) / self.max_range_lidar

        target_angle = target_angle.view(-1, 1)
        target_dist = target_dist.view(-1, 1)
        charger_angle = charger_angle.view(-1, 1)
        charger_dist = charger_dist.view(-1, 1)  # if single value per robot
        scan = scan.view(-1, 1)

        new_state = (
            torch.cat(
                [
                    scan,
                    target_angle,
                    target_dist,
                    charger_angle,
                    charger_dist,
                    battery_time_hold,
                ],
                dim=1 if n > 1 else 0,
            )
            .float()
            .to(self.device)
        )

        return (
            new_state,
            tensor_objs_cx_cy_w_h,
            robot_pose,
            target_position,
            charger_position,
        )

    def initialize_x_cycle(self, n, objs, test=False):
        charger_positions = self.sample_valid_positions(n, objs, min_clearance=0.25)
        charger_x, charger_y = charger_positions[:, 0:1], charger_positions[:, 1:2]

        closeness = 6 if test else 0.8

        MAX_BATTERY_N = 25
        battery_t = rand_choice_tensor(
            [self.dt * nn for nn in range(MAX_BATTERY_N + 1)], (n, 1)
        )

        rover_theta = uniform_tensor(-np.pi, np.pi, (n, 1))
        rover_rho = uniform_tensor(0, 1, (n, 1)) * (battery_t * self.rover_max_velocity)
        rover_rho = torch.clamp(rover_rho, closeness, 14.14)

        rover_x = charger_x + rover_rho * torch.cos(rover_theta)
        rover_y = charger_y + rover_rho * torch.sin(rover_theta)

        dest_positions = self.sample_valid_positions(n, objs, min_clearance=0.25)
        dest_x, dest_y = dest_positions[:, 0:1], dest_positions[:, 1:2]

        delta_x = dest_x - rover_x
        delta_y = dest_y - rover_y
        rover_theta = torch.atan2(delta_y, delta_x)

        # place hold case
        MAX_BATTERY_N = 5
        ratio = 0.15 if not test else 0.0
        rand_mask = uniform_tensor(0, 1, (n, 1))
        rand = rand_mask > 1 - ratio
        ego_rho = uniform_tensor(0, closeness, (n, 1))
        rover_x[rand] = (charger_x + ego_rho * torch.cos(rover_theta))[rand]
        rover_y[rand] = (charger_y + ego_rho * torch.sin(rover_theta))[rand]
        # battery_t[rand] = np.random.random() * (2 - 0.2) + 0.2
        battery_t[rand] = (
            self.dt * MAX_BATTERY_N
        )  # np.random.random() * (2.5 - 1.5) + 1.5

        return torch.cat(
            [
                rover_x,
                rover_y,
                dest_x,
                dest_y,
                charger_x,
                charger_y,
                battery_t,
            ],  # , hold_t],
            dim=1,
        ), rover_theta

    def update_state(
        self,
        state: torch.Tensor,
        v: float,
        theta: float,
        robot_pose: torch.Tensor,
        world_objects: torch.Tensor,
        target: torch.Tensor,
        chargers: torch.Tensor,
        collision_enabled: bool = False,
    ):
        v_lin = (
            v * (self.rover_max_velocity - self.rover_min_velocity)
            + self.rover_min_velocity
        )

        # 🚗 Compute new position
        new_x = robot_pose[0] + v_lin * torch.cos(theta) * self.dt
        new_y = robot_pose[1] + v_lin * torch.sin(theta) * self.dt
        new_pose = torch.tensor([new_x, new_y, theta], device=self.device)

        # 🔎 LIDAR and distances
        scan = self.simulate_lidar_exact(new_pose, world_objects) / self.max_range_lidar
        norm_dest, ang_dest = self.estimate_destination(
            new_pose.unsqueeze(0), target.unsqueeze(0)
        )
        c_norm, c_angle = self.estimate_destination(
            new_pose.unsqueeze(0), chargers.squeeze().unsqueeze(0)
        )

        battery_charge = 5
        near_charger = soft_step_hard(0.05 * (self.enough_close_to_charger - c_norm))
        battery_time = (state[11] - self.dt) * (
            1 - near_charger
        ) + battery_charge * near_charger

        # 📦 Final state vector
        new_state = torch.cat(
            [
                scan,
                torch.tensor(
                    [
                        ang_dest,
                        norm_dest,
                        c_angle,
                        c_norm,
                        battery_time,
                        # charger_time,  # include if needed in state
                    ],
                    device=self.device,
                ),
            ]
        )

        return new_state, new_pose

    # def initialize_x_cycle(self, n, obj,  test=False):
    #     charger_x = uniform_tensor(0, 10, (n, 1))
    #     charger_y = uniform_tensor(0, 10, (n, 1))

    #     closeness = 10 if test else 0.8
    #     MAX_BATTERY_N = 25
    #     battery_t = rand_choice_tensor([self.dt * nn for nn in range(MAX_BATTERY_N + 1)], (n, 1))
    #     rover_theta = uniform_tensor(-np.pi, np.pi, (n, 1))
    #     rover_rho = uniform_tensor(0, 1, (n, 1)) * (battery_t * self.rover_max_velocity)
    #     rover_rho = torch.clamp(rover_rho, closeness, 14.14)

    #     rover_x = charger_x + rover_rho * torch.cos(rover_theta)
    #     rover_y = charger_y + rover_rho * torch.sin(rover_theta)

    #     dest_x = uniform_tensor(0, 10, (n, 1))
    #     dest_y = uniform_tensor(0, 10, (n, 1))

    #     delta_x = dest_x - rover_x
    #     delta_y = dest_y - rover_y
    #     rover_theta = torch.atan2(delta_y, delta_x)

    #     # place hold case
    #     ratio = 0.25 if not test else 0.0
    #     rand_mask = uniform_tensor(0, 1, (n, 1))
    #     rand = rand_mask > 1 - ratio
    #     ego_rho = uniform_tensor(0, closeness, (n, 1))
    #     rover_x[rand] = (charger_x + ego_rho * torch.cos(rover_theta))[rand]
    #     rover_y[rand] = (charger_y + ego_rho * torch.sin(rover_theta))[rand]
    #     # battery_t[rand] = np.random.random() * 4
    #     battery_t[rand] = self.dt * MAX_BATTERY_N  # np.random.random() * (2.5 - 1.5) + 1.5

    #     hold_t = 0 * dest_x + self.dt * self.hold_t
    #     hold_t[rand] = rand_choice_tensor([self.dt * nn for nn in range(self.hold_t + 1)], (n, 1))[rand]

    #     return torch.cat(
    #         [rover_x, rover_y, dest_x, dest_y, charger_x, charger_y, battery_t, hold_t],
    #         dim=1,
    #     ), rover_theta

    # def update_state_batch(self, state, v, theta, robot_pose, world_objects, target, chargers, collision_enabled=False):
    #     """
    #     Fully vectorized update of the robot state for a batch of moves.
    #     """
    #     # --- Rescale velocity ---
    #     if not collision_enabled:
    #         v = v * (self.rover_max_velocity - self.rover_min_velocity) + self.rover_min_velocity

    #     # --- Update robot pose linearly ---
    #     # Predict angle displacement
    #     # new_x = robot_pose[:, 0] + (v * torch.cos(robot_pose[:, 2] + theta) * self.dt)
    #     # new_y = robot_pose[:, 1] + (v * torch.sin(robot_pose[:, 2] + theta) * self.dt)
    #     # new_heading = robot_pose[:, 2] + theta

    #     new_x = robot_pose[:, 0] + (v * torch.cos(theta) * self.dt)
    #     new_y = robot_pose[:, 1] + (v * torch.sin(theta) * self.dt)
    #     new_heading = theta

    #     new_pose = torch.stack([new_x, new_y, new_heading], dim=1)

    #     if collision_enabled:
    #         x_exp = new_pose[:, 0].unsqueeze(1)
    #         y_exp = new_pose[:, 1].unsqueeze(1)
    #         obs_cx = world_objects[:, 0].unsqueeze(0)
    #         obs_cy = world_objects[:, 1].unsqueeze(0)
    #         obs_w = world_objects[:, 2].unsqueeze(0)
    #         obs_h = world_objects[:, 3].unsqueeze(0)

    #         collision_mask = (torch.abs(x_exp - obs_cx) <= obs_w / 2) & (torch.abs(y_exp - obs_cy) <= obs_h / 2)
    #         collision_any = collision_mask.any(dim=1)

    #         if collision_any.shape[0] == 1 and collision_any[0]:
    #             return None, None

    #         # If a collision is detected, revert the pose to the previous one.
    #         new_pose[collision_any] = robot_pose[collision_any]

    #     new_scan = self.simulate_lidar_scan(new_pose, world_objects)
    #     t_norm, t_angle = self.estimate_destination(new_pose, target)
    #     c_norm, c_angle = self.estimate_destination(new_pose, chargers)

    #     # ADAPTED from the paper code
    #     battery_charge = 5
    #     near_charger = soft_step_hard(0.05 * (self.enough_close_to_charger - c_norm))
    #     # near_charger = (torch.tanh(500 * (0.05 * (self.enough_close_to_charger - nearest_dists))) + 1) / 2
    #     es_battery_time = (state[:, 11].unsqueeze(1) - self.dt) * (1 - near_charger) + battery_charge * near_charger
    #     es_charger_time = state[:, 12].unsqueeze(1) - self.dt * near_charger

    #     new_state = torch.cat(
    #         [
    #             new_scan,
    #             t_angle,
    #             t_norm,
    #             c_angle,
    #             c_norm,
    #             es_battery_time,
    #             es_charger_time,
    #         ],
    #         dim=1,
    #     ).to(self.device)

    #     return new_state, new_pose
