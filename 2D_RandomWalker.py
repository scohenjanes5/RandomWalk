import random
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm


class RandomWalker(object):
    def __init__(self):
        self.width = 10
        self.height = 10
        self.n_walkers = 3
        self.probability_threshold = 0.5
        self.time_stamps = 10000
        self.walker_tracker = list()

    def probability_decisioner(self):
        return random.random() < self.probability_threshold

    def create_2D_space(self):
        return [None, None]

    def create_x_start(self):
        return random.randint(0, self.width)

    def create_y_start(self):
        return random.randint(0, self.height)

    def get_direction(self):
        return random.randint(0, 3)

    def go_left(self, initial_space):
        initial_space[0] = initial_space[0] - 1
        return initial_space

    def go_right(self, initial_space):
        initial_space[0] = initial_space[0] + 1
        return initial_space

    def go_up(self, initial_space):
        initial_space[1] = initial_space[1] + 1
        return initial_space

    def go_down(self, initial_space):
        initial_space[1] = initial_space[1] - 1
        return initial_space

    def define_goal(self):
        return random.randint(0, self.width)

    def plot_results(self):

        colors = cm.rainbow(np.linspace(0, 1, len(self.walker_tracker)))

        for result, c in zip(self.walker_tracker, colors):
            plt.scatter(result[0], result[1], color=c, label="Walker")

        plt.legend(loc="upper left")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("2D-Random Walk")
        plt.show()

    def run_simulation(self, initial_space):

        x_coordinates_tracker = list()
        y_coordinates_tracker = list()

        x_coordinates_tracker.append(initial_space[0])
        y_coordinates_tracker.append(initial_space[1])

        for time_stamp in range(0, self.time_stamps):
            direction = self.get_direction()

            if direction == 0:
                initial_space = self.go_left(initial_space)
                x_coordinates_tracker.append(initial_space[0])
                y_last_tracker = y_coordinates_tracker[-1]
                y_coordinates_tracker.append(y_last_tracker)

            elif direction == 1:
                initial_space = self.go_right(initial_space)
                x_coordinates_tracker.append(initial_space[0])
                y_last_tracker = y_coordinates_tracker[-1]
                y_coordinates_tracker.append(y_last_tracker)

            elif direction == 2:
                initial_space = self.go_down(initial_space)
                y_coordinates_tracker.append(initial_space[1])
                x_last_tracker = x_coordinates_tracker[-1]
                x_coordinates_tracker.append(x_last_tracker)

            elif direction == 3:
                initial_space = self.go_up(initial_space)
                y_coordinates_tracker.append(initial_space[1])
                x_last_tracker = x_coordinates_tracker[-1]
                x_coordinates_tracker.append(x_last_tracker)

        return (x_coordinates_tracker, y_coordinates_tracker)

    def main(self):

        initial_space = self.create_2D_space()
        x_start = self.create_x_start()
        y_start = self.create_y_start()

        for walker in range(0, self.n_walkers):
            initial_space[0] = x_start
            initial_space[1] = y_start

            self.walker_tracker.append(self.run_simulation(initial_space))

        self.plot_results()


if __name__ == "__main__":

    walker = RandomWalker()
    walker.main()
