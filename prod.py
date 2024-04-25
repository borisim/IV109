import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
import pandas as pd
import random

# Animation controls
interval = 200 # ms, time between animation frames

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
text = plt.figtext(.6,.9,"Population: {:,}, Year: {}".format(int(population), year), ha='center')


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

    x_male = df['Male']
    x_female = df['Female']
    year = 2024
    population = x_male.sum() + x_female.sum()

    draw()
    fig.canvas.draw_idle()

 
def draw():
    global text
    global population
    global year
    global y
    global x_male
    global x_female

    ax1.clear()
    ax2.clear()

    text.remove()
    text = plt.figtext(.7,.9,"Population: {:,}, Year: {}".format(int(population), year), ha='center')

    ax1.barh(y, x_male, align='center', color='blue')
    ax1.set(title='Males')
    ax2.barh(y, x_female, align='center', color='red')
    ax2.set(title='Females')
    ax1.invert_xaxis()
 

# pause button
axpause = plt.axes([0.075, 0.1, 0.1, 0.075])
bpause = Button(axpause, '{}'.format('run/pause'),color="grey")
bpause.on_clicked(pause)

# reset button
axreset = plt.axes([0.19, 0.1, 0.1, 0.075])
breset = Button(axreset, 'reset',color="grey")
breset.on_clicked(reset)

# slider 1
sax1 = plt.axes([0.075, .9, 0.3, 0.02])
s1 = Slider(sax1, 'death rate', 10, 50, valinit=50)

# slider 2
fr = 1.83
sax2 = plt.axes([0.075, .7, 0.3, 0.02])
s2 = Slider(sax2, 'fertility rate', 0, 10, valinit=fr)

# slider 1
split = 51
sax3 = plt.axes([0.075, .5, 0.3, 0.02])
s3 = Slider(sax3, 'gender ratio', 0, 100, valinit=split)

# slider 1
migration = 0
sax4 = plt.axes([0.075, .3, 0.3, 0.02])
s4 = Slider(sax4, 'net migration', -30, 30, valinit=migration)


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
    global migration
    migration = val
    fig.canvas.draw_idle()

def logistic_function(x, x0):
    return 1 / (0.005 + np.exp(-0.1 * (x - x0)))

x0 = 50
probs = np.linspace(0, 100, 51)
probs = logistic_function(probs, x0)
probs /= 100
probs = 1 - probs
probs = pd.DataFrame(probs, columns=['probies']).to_numpy()


def update_plot(num):
    global x_male
    global x_female
    global fr
    global year
    global population

    if not started:
        return
    
    year += 1
    population = x_male.sum() + x_female.sum()
    
    mothers = sum(x_female[7:25])
    kids = (mothers * fr) / 17
    kids = kids + (migration * population / 1000)
    boys = kids * (split / 100)
    girls = kids * ((100 - split) / 100)


    x_male = x_male.to_frame().mul(probs, axis = 0).shift(periods=1).fillna(boys, limit=1)['Male']
    x_female = x_female.to_frame().mul(probs, axis = 0).shift(periods=1).fillna(girls, limit=1)['Female']

    age = 0
    wtf = 0
    for i, value in enumerate(x_male):
        age += (i + 1) * value 
        wtf += value
        

    for i, value in enumerate(x_female):
        age += (i + 1) * value 
        wtf += value
    
    print(age / wtf * 2)

    draw()


# call update function on slider value change
s1.on_changed(update_s1)
s2.on_changed(update_s2)
s3.on_changed(update_s3)
s4.on_changed(update_s4)

draw()

ani = animation.FuncAnimation(fig, update_plot, interval=interval)

plt.show()