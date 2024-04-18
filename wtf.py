import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# Function to update the plot for each frame of the animation
def update(frame):
    plt.cla()  # Clear the previous plot
    plt.plot(np.sin(frame * 0.1), label='sin(x)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Sine Wave (Frame {})'.format(frame))
    plt.legend()

# Create a figure and axes
fig, ax = plt.subplots()

# Create the animation
# 100 frames, each lasting for 100 milliseconds (total animation duration: 10 seconds)
ani = FuncAnimation(fig, update, frames=100, interval=100)

# Call the stop method to stop the animation
ani.event_source.stop()

# Close the plot window
# plt.close()

plt.show()
