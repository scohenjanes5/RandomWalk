import numpy as np
import random
from matplotlib import pyplot as plt
from collections import Counter


class RandomWalker(object):
    def __init__(self):
        self.L = 1001
        self.n_particles = 10
        self.amount_time_stamps = 1000
        self.probability_threshold = 0.5

    def create_walker(self):  # Creation of 10000 particles in the middle of the list

        walker = [0 for x in range(self.L)]
        walker[int((len(walker) + 1) / 2)] = 1

        return walker

    def probability_decisioner(self):
        return random.random() < self.probability_threshold

    def walking(
        self, walker
    ):  # A random path for n timestamps is created for every particle

        direction_list = list()
        Position_Tracker = list()

        for timestep in range(0, self.amount_time_stamps):
            direction = self.probability_decisioner()
            direction_list.append(direction)

        for direc in direction_list:

            if direc is True:  # Go Left
                index_to_move = walker.index(1)
                walker[index_to_move] = 0
                walker[index_to_move - 1] = 1
                Position_Tracker.append(walker.index(1))

            elif direc is False:  # Go Right
                index_to_move = walker.index(1)
                walker[index_to_move] = 0
                walker[index_to_move + 1] = 1
                Position_Tracker.append(walker.index(1))

            else:
                raise Exception("Unsupported Direction")

        return Position_Tracker

    def data_processer(self, time_list):  # Sorts the different timestamps

        l = list()
        counter_list = Counter(time_list)

        for key in sorted(counter_list.keys()):
            l.append([key, counter_list[key]])

        return l

    def condition(self, l, p, t_0):  # Computes the Results for plotting

        positions = [x[0] for x in l]
        amounts = [x[1] for x in l]
        l = list(zip(positions, amounts))

        num = 0
        den = 0

        for i in l:
            num += ((i[0] - t_0) ** p) * i[1]
            den += i[1]

        r = num / float(den) * 10

        return r

    def prepare_pool(self, Positions_Set):

        New_Position_Set = []

        for i in Positions_Set:  # Preparation of the list with positions
            # Every Walker starts in the same position

            if i[0] == 500:
                tmp = [x + 1 for x in i]
                New_Position_Set.append(tmp)

            elif i[0] == 502:
                tmp = [x - 1 for x in i]
                New_Position_Set.append(tmp)

        return np.asarray(New_Position_Set)

    def process_results(self, Positions_Set):

        Results = list()

        for i in range(
            0, self.amount_time_stamps
        ):  # Computes for every position how many times it has been visited in 1 particular timestamp
            tmp_list = Positions_Set[:, i]
            sorted_tmp_list = self.data_processer(tmp_list)
            res_1 = self.condition(sorted_tmp_list, 2, 501)
            res_2 = self.condition(sorted_tmp_list, 1, 501) ** 2
            final_res = res_1 - res_2
            Results.append(final_res)

        return Results

    def run_simulation(self):

        Positions_Set = list()
        Results = list()

        for i in range(0, self.n_particles):
            original_walker = self.create_walker()
            P = self.walking(original_walker)
            Positions_Set.append(P)

        Positions_Set = self.prepare_pool(Positions_Set)

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(8, 10)
        )  # Adjust figsize as needed

        # Plotting Random Walk Trajectories
        for single_pos in Positions_Set:
            ax1.plot(single_pos)
        ax1.set_title("Random Walk Trajectories")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Position of the particles")

        Results = self.process_results(Positions_Set)

        # Plotting Variance
        ax2.plot(Results, "g", label="Vector B, D=5")
        ax2.set_title(
            r"$\langle x^2(t) \rangle$ - $\langle x(t) \rangle^2$ as a function of t"
        )
        ax2.set_xlabel("t")
        ax2.set_ylabel(r"$\langle x^2(t) \rangle$ - $\langle x(t) \rangle^2$")
        ax2.legend(loc="upper left")

        plt.tight_layout()  # Adjust subplot parameters for a tight layout
        plt.show()

        # self.plot_results(Results) #No longer needed

    def main(self):
        self.run_simulation()


if __name__ == "__main__":
    random_walker = RandomWalker()
    random_walker.main()
