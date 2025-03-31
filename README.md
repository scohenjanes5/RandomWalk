# RandomWalk

An implementation of a random walk on a regular 1D lattice can be found in the RandomWalker.py program.

At each step in time (amount_time_stamps) the location of n particles (n_particles)
jumps to another site according to some probability distribution (probability_threshold).
The output of this process creates the following figure:

![alt tag](https://user-images.githubusercontent.com/14283557/33336135-4ddd14f2-d46f-11e7-8145-2919ab567402.png)


Furthermore, the program also shows the behaviour of the variance which is computed as follows:
$\sigma^2 = \langle x^2(t) \rangle - \langle x(t) \rangle^2$
The output of this computation can be seen in the following figure:

![alt tag](https://user-images.githubusercontent.com/14283557/33336178-6c7799c8-d46f-11e7-8d7b-5bc7aaa57cdf.png)

The program 2D_RandomWalker pushes the simulation one step further by computing the random walk phenomenon in a 
2 dimensional space. In this case the amount of walkers can decide at each time step to walk either left/right or up/down.
The output of the simulation has very interesting geometric properties. In fact, it corresponds to a discrete fractal which is plotted by the program and looks like the following figure:

![alt tag](https://user-images.githubusercontent.com/14283557/33526514-f6336328-d842-11e7-9d5b-ec97cef7c870.png)
