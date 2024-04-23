import matplotlib.pyplot as plt
from matplotlib import animation
import pandas as pd

file_path = "data/men.xlsx"
df = pd.read_excel(file_path)
df = df.iloc[:, :2]
df.rename(columns={'Unnamed: 0': 'Age'}, inplace=True)
df.rename(columns={'Věkové složení mužů k 1. 1. 2022': 'Male'}, inplace=True)
df = df.iloc[6:]

file_path = "data/women.xlsx"
d = pd.read_excel(file_path)
d = d.iloc[:, :2]
d.rename(columns={'Unnamed: 0': 'Age'}, inplace=True)
d.rename(columns={'Věkové složení žen k 1. 1. 2022': 'Female'}, inplace=True)
d = d.iloc[6:107]

merged_df = pd.merge(df, d, on='Age', how='inner')
merged_df = merged_df.iloc[:101]
df = merged_df

def custom_grouping(index):
    return index // 2

# Group the DataFrame using the custom grouping function and sum each group
df = merged_df.groupby(custom_grouping).sum()


y = range(0, len(df))
x_male = df['Male']
x_female = df['Female']

list = [k for k in range(10)]
y = range(0, len(list))

def update(i):
    ax1.clear()
    ax2.clear()
    list[0] += 1

    ax1.barh(y, list)
    ax2.barh(y, list)
    ax1.invert_xaxis()
    # ax[1].barh(y, list)

# fig, ax = plt.subplots(figsize=(10, 6))
# fig, ax = plt.subplots(ncols=2, sharey=True, figsize=(9, 6))
fig, (ax1, ax2) = plt.subplots(1,2)
# ax[0].barh(y, list)
# ax[1].barh(y, list)
an = animation.FuncAnimation(fig, update, repeat=True, interval=300)


plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# import mpl_toolkits.axes_grid1
# import matplotlib.widgets

# class Player(FuncAnimation):
#     def __init__(self, fig, func, frames=None, init_func=None, fargs=None,
#                  save_count=None, mini=0, maxi=100, pos=(0.125, 0.92), **kwargs):
#         self.i = 0
#         self.min=mini
#         self.max=maxi
#         self.runs = True
#         self.forwards = True
#         self.fig = fig
#         self.func = func
#         self.setup(pos)
#         FuncAnimation.__init__(self,self.fig, self.update, frames=self.play(), 
#                                            init_func=init_func, fargs=fargs,
#                                            save_count=save_count, **kwargs )    

#     def play(self):
#         while self.runs:
#             self.i = self.i+self.forwards-(not self.forwards)
#             if self.i > self.min and self.i < self.max:
#                 yield self.i
#             else:
#                 self.stop()
#                 yield self.i


#     def setup(self, pos):
#         playerax = self.fig.add_axes([pos[0],pos[1], 0.64, 0.04])
#         divider = mpl_toolkits.axes_grid1.make_axes_locatable(playerax)
#         sliderax = divider.append_axes("right", size="500%", pad=0.07)
        
#         self.slider = matplotlib.widgets.Slider(sliderax, '', 
#                                                 self.min, self.max, valinit=self.i)
#         self.slider.on_changed(self.set_pos)

#     def set_pos(self,i):
#         self.i = int(self.slider.val)
#         self.func(self.i)

#     def update(self,i):
#         self.slider.set_val(i)



# fig, ax = plt.subplots()
# x = np.linspace(0,6*np.pi, num=100)
# y = np.sin(x)

# ax.plot(x,y)
# point, = ax.plot([],[], marker="o", color="crimson", ms=15)

# def update(i):
#     point.set_data(x[i],y[i])

# ani = Player(fig, update, maxi=len(y)-1)

# plt.show()