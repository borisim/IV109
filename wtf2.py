import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

import matplotlib.animation as animation

list = [k for k in range(10)]
y = range(0, len(list))
class PauseAnimation:
    def __init__(self):
        fig, (self.ax1, self.ax2) = plt.subplots(1,2)
        axamp = plt.axes([0.25, .03, 0.50, 0.02])
        # Slider
        samp = Slider(axamp, 'Amp', 0, 1, valinit=.5)

        # ax1.set_title('Click to pause/resume the animation')
        # x = np.linspace(-0.1, 0.1, 1000)

        # Start with a normal distribution
        # self.p, = ax.plot(x, self.n0)
        # self.p1, = ax1.barh(y, list)
        # self.p2, = ax2.barh(y, list)

        self.animation = animation.FuncAnimation(fig, self.update, frames=200, interval=50)
        self.paused = False

        # fig.canvas.mpl_connect('button_press_event', self.toggle_pause)

    def toggle_pause(self, *args, **kwargs):
        if self.paused:
            self.animation.resume()
        else:
            self.animation.pause()
        self.paused = not self.paused

    def update(self, i):
        self.ax1.clear()
        self.ax2.clear()
        list[0] += 1

        self.ax1.barh(y, list)
        self.ax2.barh(y, list)
        self.ax1.invert_xaxis()



pa = PauseAnimation()
plt.show()