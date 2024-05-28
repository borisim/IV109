import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from matplotlib import ticker
import pandas as pd

# Animation controls
interval = 100  # ms, time between animation frames

# data preprocessing
file_path = "data/men.xlsx"
df = pd.read_excel(file_path)
df = df.iloc[:, :2]
df.rename(columns={"Unnamed: 0": "Age"}, inplace=True)
df.rename(columns={"Věkové složení mužů k 1. 1. 2022": "Male"}, inplace=True)
df = df.iloc[6:]

file_path = "data/women.xlsx"
d = pd.read_excel(file_path)
d = d.iloc[:, :2]
d.rename(columns={"Unnamed: 0": "Age"}, inplace=True)
d.rename(columns={"Věkové složení žen k 1. 1. 2022": "Female"}, inplace=True)
d = d.iloc[6:107]

merged_df = pd.merge(df, d, on="Age", how="inner")
merged_df = merged_df.iloc[:101]
df = merged_df


# def custom_grouping(index):
#     return index // 2


# df = merged_df.groupby(custom_grouping).sum()

# plot figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
ax1.set_position([0.45, 0.1, 0.25, 0.7])
ax2.set_position([0.7, 0.1, 0.25, 0.7])

# global variables
y = range(0, len(df))
x_male = df["Male"]
x_female = df["Female"]
year = None
started = False
paused = False
population = None
dep = None
text = plt.figtext(0.7, 0.8, "", ha="center")

BIGGEST_POP_YEAR = max(max(x_male), max(x_female))

# A better solution would be a normal distribution, but this is good enough for now
FERTILITY_START_AGE = 18
FERTILITY_END_AGE = 35

GENDER_BIRTH_RATE_RATIO = 1.04  # multiplies amount of boys born


def logistic_function(x, x0):
    return 1 / (0.005 + np.exp(-0.1 * (x - x0)))


male_x0 = 60
male_death_probs = np.linspace(0, 100, len(df))
male_death_probs = logistic_function(male_death_probs, male_x0)
male_death_probs /= 100
male_survival_probs = 1 - male_death_probs
male_survival_probs = pd.DataFrame(male_survival_probs, columns=["probies"]).to_numpy()

female_x0 = 65
female_death_probs = np.linspace(0, 100, len(df))
female_death_probs = logistic_function(female_death_probs, female_x0)
female_death_probs /= 100
female_survival_probs = 1 - female_death_probs
female_survival_probs = pd.DataFrame(
    female_survival_probs, columns=["probies"]
).to_numpy()


def dependency_ratio(x_male, x_female):
    return (
        (
            sum(x_male[0:15])
            + sum(x_female[0:15])
            + sum(x_male[65:])
            + sum(x_female[65:])
        )
        / (sum(x_male[15:65]) + sum(x_female[15:65]))
    ) * 100


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


def reset_vals():
    global df
    global x_male
    global x_female
    global population
    global year
    global text
    global dep

    x_male = df["Male"]
    x_female = df["Female"]
    year = 2022
    population = x_male.sum() + x_female.sum()
    dep = dependency_ratio(x_male, x_female)


def reset(val):
    global paused
    if not paused:
        paused = True
        ani.pause()

    reset_vals()

    draw()
    fig.canvas.draw_idle()


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
    text = plt.figtext(
        0.7,
        0.85,
        "Population: {:,}, Year: {}, Dependency Ratio: {}".format(
            int(population), year, int(dep)
        ),
        ha="center",
    )

    ax1.barh(y, x_female, align="center", color="lightpink")
    ax1.barh(y, x_male, align="center", color="blue")
    ax1.set(title="Males")
    ax2.barh(y, x_male, align="center", color="lightblue")
    ax2.barh(y, x_female, align="center", color="red")
    ax2.set(title="Females")
    ax1.invert_xaxis()

    axis_lim = BIGGEST_POP_YEAR * 1.1

    ax1.set_xlim([axis_lim, 0])
    ax2.set_xlim([0, axis_lim])

    positions = [0, 20, 40, 60, 80, 100]
    labels = [0, 20, 40, 60, 80, "100+"]
    ax1.yaxis.set_major_locator(ticker.FixedLocator(positions))
    ax1.yaxis.set_major_formatter(ticker.FixedFormatter(labels))

    ax2.axes.get_yaxis().set_visible(False)
    ax2.spines["left"].set_visible(False)
    # ax2.yaxis.set_major_locator(ticker.FixedLocator(positions))
    # ax2.yaxis.set_major_formatter(ticker.FixedFormatter(labels))


def mig(val):
    global x_male
    global x_female
    global migration_avg_age
    global migamount
    global split
    global started

    boys = migamount * (split / 100)
    girls = migamount * ((100 - split) / 100)

    nbins = 40

    hist = generate_migrants(boys, nbins)
    for i in range(nbins):
        x_male[migration_avg_age + i - nbins // 2] += hist[i]

    hist = generate_migrants(girls, nbins)
    for i in range(nbins):
        x_female[migration_avg_age + i - nbins // 2] += hist[i]

    if paused:
        draw()


# pause button
axpause = plt.axes([0.05, 0.05, 0.1, 0.075])
bpause = Button(axpause, "{}".format("run/pause"), color="grey")
bpause.on_clicked(pause)

# reset button
axreset = plt.axes([0.165, 0.05, 0.1, 0.075])
breset = Button(axreset, "reset", color="grey")
breset.on_clicked(reset)

# migrants button
axmig = plt.axes([0.275, 0.05, 0.1, 0.075])
bmig = Button(axmig, "wave", color="grey")
bmig.on_clicked(mig)

# slider 1
sax1 = plt.axes([0.075, 0.7, 0.3, 0.02])
s1 = Slider(sax1, "death rate", 10, 60, valinit=60)

# slider 2
fr = 1.674
sax2 = plt.axes([0.075, 0.6, 0.3, 0.02])
s2 = Slider(sax2, "fertility rate", 0, 5, valinit=fr)

# slider 3
split = 50
sax3 = plt.axes([0.075, 0.5, 0.3, 0.02])
s3 = Slider(sax3, "gender ratio", 0, 100, valinit=split)

# slider 4
migration_avg_age = 35
sax4 = plt.axes([0.075, 0.4, 0.3, 0.02])
s4 = Slider(sax4, "average age", 20, 50, valinit=migration_avg_age)

# slider 5
migamount = 0
sax5 = plt.axes([0.075, 0.3, 0.3, 0.02])
s5 = Slider(sax5, "migration wave", 0, 500000, valinit=migamount)

# slider 6
net_migration = 0
sax6 = plt.axes([0.075, 0.2, 0.3, 0.02])
s6 = Slider(sax6, "net migration", 0, 250000, valinit=net_migration)

death_probs = 2


def update_s1(val):
    global death_probs
    x0 = val
    death_probs = np.linspace(0, 100, len(df))
    death_probs = logistic_function(death_probs, int(x0))
    death_probs /= 100
    death_probs = 1 - death_probs
    death_probs = pd.DataFrame(death_probs, columns=["probies"]).to_numpy()

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


def generate_migrants(amount, num_bins):
    data = np.random.normal(0, 20, int(amount))
    hist, _ = np.histogram(data, bins=range(-num_bins // 2, num_bins // 2 + 1))

    if sum(hist) == 0:
        return hist

    return hist * (amount / sum(hist))


def update_simulation():
    global x_male
    global x_female
    global fr
    global year
    global population
    global migration_avg_age
    global dep

    year += 1
    population = x_male.sum() + x_female.sum()
    dep = dependency_ratio(x_male, x_female)

    start_i = FERTILITY_START_AGE
    end_i = FERTILITY_END_AGE + 1
    mothers = sum(x_female[start_i:end_i])
    kids = (mothers * fr) / (end_i - start_i)

    x_male = (
        x_male.to_frame()
        .mul(male_survival_probs, axis=0)
        .shift(periods=1)
        .fillna(kids * (GENDER_BIRTH_RATE_RATIO) / 2, limit=1)["Male"]
    )
    x_female = (
        x_female.to_frame()
        .mul(female_survival_probs, axis=0)
        .shift(periods=1)
        .fillna(kids * (1 / GENDER_BIRTH_RATE_RATIO) / 2, limit=1)["Female"]
    )

    global net_migration
    boys = net_migration * (split / 100)
    girls = net_migration * ((100 - split) / 100)

    nbins = 40

    hist = generate_migrants(boys, nbins)
    for i in range(nbins):
        x_male[migration_avg_age + i - nbins // 2] += hist[i]

    hist = generate_migrants(girls, nbins)
    for i in range(nbins):
        x_female[migration_avg_age + i - nbins // 2] += hist[i]

    age = 0
    wtf = 0
    for i, value in enumerate(x_male):
        age += (i + 1) * value
        wtf += value

    for i, value in enumerate(x_female):
        age += (i + 1) * value
        wtf += value


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

    update_simulation()

    draw()


# call update function on slider value change
s1.on_changed(update_s1)
s2.on_changed(update_s2)
s3.on_changed(update_s3)
s4.on_changed(update_s4)
s5.on_changed(update_s5)
s6.on_changed(update_s6)

reset_vals()
draw()

ani = animation.FuncAnimation(fig, update_plot, interval=interval)


plt.show()

# youth/old ratio
# slider for continuous amount of imigrants

# one slider for migration age
# one slider for migration gender
# slider and button for instant amount of imigrants
