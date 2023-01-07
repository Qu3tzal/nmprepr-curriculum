import gym
import mpenv.envs
import matplotlib.pyplot as plt

def main():
    env_name = "Maze-Simple-ObstaclesCurriculum-v0"
    print("Creating the environments...")
    easy_curriculum_env = gym.make(env_name, curriculum_difficulty=0.0)
    medium_curriculum_env = gym.make(env_name, curriculum_difficulty=0.3)
    hard_curriculum_env = gym.make(env_name, curriculum_difficulty=1.0)

    # Sample different instances.
    k = 3
    print("Sampling {} instances...".format(k))
    for i in range(k):
        easy_curriculum_env.reset()
        easy_curriculum_env.render()
        plt.savefig("easy_obstacles_curriculum_env-{}.png".format(i))

    # Sample different instances.
    k = 3
    print("Sampling {} instances...".format(k))
    for i in range(k):
        medium_curriculum_env.reset()
        medium_curriculum_env.render()
        plt.savefig("medium_obstacles_curriculum_env-{}.png".format(i))

    # Sample different instances.
    k = 3
    print("Sampling {} instances...".format(k))
    for i in range(k):
        hard_curriculum_env.reset()
        hard_curriculum_env.render()
        plt.savefig("hard_obstacles_curriculum_env-{}.png".format(i))


if __name__ == "__main__":
    main()
