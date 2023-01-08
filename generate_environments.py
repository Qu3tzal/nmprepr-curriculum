import gym
import mpenv.envs
import matplotlib.pyplot as plt


def generate_curriculum_environments(env_name):
    print("Creating the environments for {}...".format(env_name))
    easy_curriculum_env = gym.make(env_name, curriculum_difficulty=0.0)
    medium_curriculum_env = gym.make(env_name, curriculum_difficulty=0.5)
    hard_curriculum_env = gym.make(env_name, curriculum_difficulty=1.0)

    # Sample different instances.
    k = 3
    print("Sampling {} instances...".format(k))
    for i in range(k):
        easy_curriculum_env.reset()
        easy_curriculum_env.render()
        plt.savefig("{}-easy-{}.png".format(env_name, i))

    # Sample different instances.
    k = 3
    print("Sampling {} instances...".format(k))
    for i in range(k):
        medium_curriculum_env.reset()
        medium_curriculum_env.render()
        plt.savefig("{}-medium-{}.png".format(env_name, i))

    # Sample different instances.
    k = 3
    print("Sampling {} instances...".format(k))
    for i in range(k):
        hard_curriculum_env.reset()
        hard_curriculum_env.render()
        plt.savefig("{}-hard-{}.png".format(env_name, i))


def generate_simple_environments(env_name):
    print("Creating the environments...")
    env = gym.make(env_name)

    # Sample different instances.
    k = 3
    print("Sampling {} instances...".format(k))
    for i in range(k):
        env.reset()
        env.render()
        plt.savefig("{}-{}.png".format(env_name, i))

def main():
    generate_simple_environments("Maze-Medium-v0")
    generate_curriculum_environments("Maze-Medium-DistanceCurriculum-v0")
    generate_curriculum_environments("Maze-Medium-ObstaclesCurriculum-v0")


if __name__ == "__main__":
    main()
