#import libraries 
import numpy as np
import pandas as pd 
import random
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.widgets import Slider

import time




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


plt.ion()

y = range(0, len(df))
x_male = df['Male']
x_female = df['Female']

population = x_male.sum() + x_female.sum()
population = '{:,.0f} M'.format(population/1e6)

#define plot parameters
fig, axes = plt.subplots(ncols=2, sharey=True, figsize=(9, 6))
plt.show(block=False)
plt.show(block=False)

#specify background color and plot title
font = {'family': 'monospace',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }
fig.patch.set_facecolor('xkcd:grey')
plt.figtext(.5,.9,"Population: {}".format(population), fontdict=font, ha='center')
    
#define male and female bars
axes[0].barh(y, x_female, align='center', color='lightpink')
axes[0].barh(y, x_male, align='center', color='blue')
axes[0].set(title='Males')
axes[1].barh(y, x_male, align='center', color='lightblue')
axes[1].barh(y, x_female, align='center', color='red')
axes[1].set(title='Females')
plt.show(block=False)
bg = fig.canvas.copy_from_bbox(fig.bbox)


fig.canvas.blit(fig.bbox)

def logistic_function(x, x0):
    return 1 / (0.005 + np.exp(-0.1 * (x - x0)))

x0 = 50
probs = np.linspace(0, 100, 51)
probs = logistic_function(probs, x0)
probs /= 100
probs = 1 - probs
probs = pd.DataFrame(probs, columns=['probies']).to_numpy()

#adjust grid parameters and specify labels for y-axis



sum = 0
nits = 0

for j in range(30000):
    

    # plt.show()
    axes = fig.subplots(1, 2)
    fig.patch.set_facecolor('xkcd:grey')
    plt.figtext(.5,.9,"Population: {}".format(population), fontdict=font, ha='center')

   
    start = time.time()
    # print(j)
    # axes[1].grid()
    axes[0].set(yticks=np.arange(0, 100, step=10))
    axes[0].invert_xaxis()
    # axes[0].grid()

    # fig.canvas.restore_region(bg)


    x_male = x_male.to_frame().mul(probs, axis = 0).shift(periods=1).fillna(random.randint(70000, 100000), limit=1)['Male']
    x_female = x_female.to_frame().mul(probs, axis = 0).shift(periods=1).fillna(random.randint(70000, 100000), limit=1)['Female']


    axes[0].barh(y, x_female, align='center', color='lightpink')
    axes[0].barh(y, x_male, align='center', color='blue')
    axes[0].set(title='Males')
    axes[1].barh(y, x_male, align='center', color='lightblue')
    axes[1].barh(y, x_female, align='center', color='red')
    axes[1].set(title='Females')
    
    

    # fig.canvas.blit(fig.bbox)
    fig.canvas.flush_events()
    fig.clear()

    nits += 1
    sum += time.time() - start
    print(sum / nits)


