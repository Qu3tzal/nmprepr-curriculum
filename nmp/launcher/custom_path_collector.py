from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.samplers.rollout_functions import (
    multitask_rollout,
)
from rlkit.samplers.data_collector import (
    GoalConditionedPathCollector,
)
import numpy as np


class CurriculumGoalConditionedPathCollector(GoalConditionedPathCollector):
    def __init__(self,
        env,
        policy,
        max_num_epoch_paths_saved=None,
        render=False,
        render_kwargs=None,
        observation_key="observation",
        desired_goal_key="desired_goal",
        representation_goal_key="representation_goal",
    ):
        super(CurriculumGoalConditionedPathCollector, self).__init__(
            env,
            policy,
            max_num_epoch_paths_saved,
            render,
            render_kwargs,
            observation_key,
            desired_goal_key,
            representation_goal_key,
        )

        self.CURRICULUM_STATS_WINDOW = 20
        self.curriculum_stats_last_paths = {
            "success": [],
        }
        self.current_curriculum_difficulty = 0.0

    def collect_new_paths(
        self, max_path_length, num_steps, discard_incomplete_paths
    ):
        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length, num_steps - num_steps_collected,
            )
            path = multitask_rollout(
                self._env,
                self._policy,
                max_path_length=max_path_length_this_loop,
                render=self._render,
                render_kwargs=self._render_kwargs,
                observation_key=self._observation_key,
                desired_goal_key=self._desired_goal_key,
                representation_goal_key=self._representation_goal_key,
                return_dict_obs=True,
                reset_kwargs={"curriculum_difficulty": self.current_curriculum_difficulty}
            )
            path_len = len(path["actions"])
            if (
                path_len != max_path_length
                and not path["terminals"][-1]
                and discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len

            # Register the statistics used to manage the curriculum.
            self.curriculum_stats_last_paths["success"].append(1.0 if path["env_infos"]["success"][-1] else 0.0)
            self.curriculum_stats_last_paths["success"] =\
                self.curriculum_stats_last_paths["success"][:self.CURRICULUM_STATS_WINDOW]

            # Update the curriculum difficulty.
            success_rate = np.mean(np.asarray(self.curriculum_stats_last_paths["success"]))
            if success_rate > 0.75:
                self.current_curriculum_difficulty += min(0.1, 1.0)

            # Record the curriculum difficulty.
            path["curriculum_difficulty"] = self.current_curriculum_difficulty

            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths

    def get_diagnostics(self):
        stats = super(CurriculumGoalConditionedPathCollector, self).get_diagnostics()
        curriculum_difficulties = [path["curriculum_difficulty"] for path in self._epoch_paths]
        stats.update(
            create_stats_ordered_dict(
                "curriculum difficulty", curriculum_difficulties, always_show_all_stats=True,
            )
        )

        return stats
