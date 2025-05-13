import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1: Load and process the data globally
df = pd.read_csv('medical_examination.csv')

# 2: Add 'overweight' column
bmi = df['weight'] / ((df['height'] / 100) ** 2)
df['overweight'] = bmi.apply(lambda x: 0 if x < 25 else 1)

# 3: Normalize cholesterol and gluc
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# 4: Categorical plot function
def draw_cat_plot():
    # 5: Create melted DataFrame
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6: Group and count
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'])['value'].count().reset_index(name='total')

    # 7: Create the catplot
    g = sns.catplot(x="variable", y="total", hue="value", col="cardio",
                    data=df_cat, kind="bar")
    g.set_axis_labels("variable", "total")

    # 8: Get the figure
    fig = g.fig

    # 9: Save and return
    fig.savefig('catplot.png')
    return fig

# 10: Heatmap function
def draw_heat_map():
    # 11: Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'].between(df['height'].quantile(0.025), df['height'].quantile(0.975))) &
        (df['weight'].between(df['weight'].quantile(0.025), df['weight'].quantile(0.975)))
    ]

    # 12: Calculate correlation matrix
    corr = df_heat.corr()

    # 13: Generate mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14: Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # 15: Draw heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', cmap='coolwarm', vmin=-1, vmax=1, center=0, ax=ax)

    # 16: Save and return
    fig.savefig('heatmap.png')
    return fig