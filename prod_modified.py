import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from matplotlib import ticker
import pandas as pd
import random

# Animation controls
interval = 100 # ms, time between animation frames

# data preprocessing
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

df = merged_df.groupby(custom_grouping).sum()

# plot figure and axes
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16,8))
ax1.set_position([0.425, 0.1, 0.25, 0.7])
ax2.set_position([0.725, 0.1, 0.25, 0.7])

# global variables
y = range(0, len(df))
x_male = df['Male']
x_female = df['Female']
year = 2024
started = False
paused = False
population = x_male.sum() + x_female.sum()
dep = ((sum(x_male[0:7]) + sum(x_female[0:7]) + sum(x_male[32:]) + sum(x_female[32:]))
       / (sum(x_male[7:32]) + sum(x_female[7:32]))) * 100
text = plt.figtext(.6,.9,"Population: {:,}, Year: {}, Dependency: {}".format(int(population), year, int(dep)), ha='center')

BIGGEST_POP_YEAR = max(max(x_male), max(x_female))

def pause(val):
    global paused
    global started

    if not started:
        started = True
        return

    if paused:
        ani.resume()
    else:
        ani.pause()
    paused = not paused
    fig.canvas.draw_idle()


def reset(val):
    global paused
    if not paused:
        paused = True
        ani.pause()

    global df
    global x_male
    global x_female
    global population
    global year
    global text
    global dep

    x_male = df['Male']
    x_female = df['Female']
    year = 2024
    population = x_male.sum() + x_female.sum()
    dep = ((sum(x_male[0:7]) + sum(x_female[0:7]) + sum(x_male[32:]) + sum(x_female[32:]))
       / (sum(x_male[7:32]) + sum(x_female[7:32]))) * 100

    draw()
    fig.canvas.draw_idle()

def mig(val):
    global x_male
    global x_female
    global migration_avg_age
    global migamount
    global split

    boys = migamount * (split / 100)
    girls = migamount * ((100 - split) / 100)

    nbins = 20

    hist = generate_migrants(boys, nbins)
    for i in range(nbins):
        x_male[migration_avg_age//2 + i - nbins // 2] += hist[i]

    hist = generate_migrants(girls, nbins)
    for i in range(nbins):
        x_female[migration_avg_age//2 + i - nbins // 2] += hist[i]

def draw():
    global text
    global population
    global year
    global y
    global x_male
    global x_female
    global dep

    ax1.clear()
    ax2.clear()

    text.remove()
    text = plt.figtext(.7,.9,"Population: {:,}, Year: {}, Dependency: {}".format(int(population), year, int(dep)), ha='center')

    ax1.barh(y, x_female, align='center', color='lightpink')
    ax1.barh(y, x_male, align='center', color='blue')
    ax1.set(title='Males')
    ax2.barh(y, x_male, align='center', color='lightblue')
    ax2.barh(y, x_female, align='center', color='red')
    ax2.set(title='Females')
    ax1.invert_xaxis()

    axis_lim = BIGGEST_POP_YEAR * 1.2

    ax1.set_xlim([axis_lim, 0])
    ax2.set_xlim([0, axis_lim])

    positions = [0, 10, 20, 30, 40, 50]
    labels = [0, 20, 40, 60, 80, "100+"]
    ax1.yaxis.set_major_locator(ticker.FixedLocator(positions))
    ax1.yaxis.set_major_formatter(ticker.FixedFormatter(labels))
    ax2.yaxis.set_major_locator(ticker.FixedLocator(positions))
    ax2.yaxis.set_major_formatter(ticker.FixedFormatter(labels))





# pause button
axpause = plt.axes([0.075, 0.05, 0.1, 0.075])
bpause = Button(axpause, '{}'.format('run/pause'),color="grey")
bpause.on_clicked(pause)

# reset button
axreset = plt.axes([0.19, 0.05, 0.1, 0.075])
breset = Button(axreset, 'reset',color="grey")
breset.on_clicked(reset)

# migrants button
axmig = plt.axes([0.3, 0.05, 0.1, 0.075])
bmig = Button(axmig, 'wave',color="grey")
bmig.on_clicked(mig)

# slider 1
sax1 = plt.axes([0.075, .7, 0.3, 0.02])
s1 = Slider(sax1, 'death rate', 10, 60, valinit=60)

# slider 2
fr = 1.83
sax2 = plt.axes([0.075, .6, 0.3, 0.02])
s2 = Slider(sax2, 'fertility rate', 0, 5, valinit=fr)

# slider 3
split = 50
sax3 = plt.axes([0.075, .5, 0.3, 0.02])
s3 = Slider(sax3, 'gender ratio', 0, 100, valinit=split)

# slider 4
migration_avg_age = 30
sax4 = plt.axes([0.075, .4, 0.3, 0.02])
s4 = Slider(sax4, 'average age', 20, 50, valinit=migration_avg_age)

# slider 5
migamount = 0
sax5 = plt.axes([0.075, .3, 0.3, 0.02])
s5 = Slider(sax5, 'migration wave', 0, 500000, valinit=migamount)

# slider 6
net_migration = 0
sax6 = plt.axes([0.075, .2, 0.3, 0.02])
s6 = Slider(sax6, 'net migration', 0, 500000, valinit=net_migration)


def update_s1(val):
    global probs
    x0 = val
    probs = np.linspace(0, 100, 51)
    probs = logistic_function(probs, int(x0))
    probs /= 100
    probs = 1 - probs
    probs = pd.DataFrame(probs, columns=['probies']).to_numpy()

    fig.canvas.draw_idle()

def update_s2(val):
    global fr
    fr = val
    fig.canvas.draw_idle()

def update_s3(val):
    global split
    split = val
    fig.canvas.draw_idle()

def update_s4(val):
    global migration_avg_age
    migration_avg_age = val
    fig.canvas.draw_idle()

def update_s5(val):
    global migamount
    migamount = val
    fig.canvas.draw_idle()

def update_s6(val):
    global net_migration
    net_migration = val
    fig.canvas.draw_idle()


def logistic_function(x, x0):
    return 1 / (0.005 + np.exp(-0.1 * (x - x0)))

x0 = 60
probs = np.linspace(0, 100, 51)
probs = logistic_function(probs, x0)
probs /= 100
probs = 1 - probs
probs = pd.DataFrame(probs, columns=['probies']).to_numpy()

def generate_migrants(amount, num_bins):
    data = np.random.normal(0, 30, int(amount))
    hist, _ = np.histogram(data, bins=range(-num_bins, num_bins + 1, 2))


    return hist

def update_plot(num):
    global x_male
    global x_female
    global fr
    global year
    global population
    global migration_avg_age
    global dep

    if not started:
        return

    year += 1
    population = x_male.sum() + x_female.sum()
    dep = ((sum(x_male[0:7]) + sum(x_female[0:7]) + sum(x_male[32:]) + sum(x_female[32:]))
       / (sum(x_male[7:32]) + sum(x_female[7:32]))) * 100

    mothers = sum(x_female[7:20])
    kids = (mothers * fr) / 12

    x_male = x_male.to_frame().mul(probs, axis = 0).shift(periods=1).fillna(kids/2, limit=1)['Male']
    x_female = x_female.to_frame().mul(probs, axis = 0).shift(periods=1).fillna(kids/2, limit=1)['Female']

    global net_migration
    boys = net_migration * (split / 100)
    girls = net_migration * ((100 - split) / 100)

    nbins = 20

    hist = generate_migrants(boys, nbins)
    for i in range(nbins):
        x_male[migration_avg_age//2 + i - nbins // 2] += hist[i]

    hist = generate_migrants(girls, nbins)
    for i in range(nbins):
        x_female[migration_avg_age//2 + i - nbins // 2] += hist[i]

    age = 0
    wtf = 0
    for i, value in enumerate(x_male):
        age += (i + 1) * value
        wtf += value


    for i, value in enumerate(x_female):
        age += (i + 1) * value
        wtf += value

    draw()


# call update function on slider value change
s1.on_changed(update_s1)
s2.on_changed(update_s2)
s3.on_changed(update_s3)
s4.on_changed(update_s4)
s5.on_changed(update_s5)
s6.on_changed(update_s6)

draw()

ani = animation.FuncAnimation(fig, update_plot, interval=interval)

plt.show()

# youth/old ratio
# slider for continuous amount of imigrants

# one slider for migration age
# one slider for migration gender
# slider and button for instant amount of imigrants