"""
Global Country Information Dataset 2023 Project
Author: Cliff Asmussen
Purpose: Data visualization project analyzing global indicators using the World Data 2023 dataset.
Visualizations:
    1. Birth Rate vs Gross Tertiary Education Enrollment (Global + G20)
    2. Armed Forces Size vs CO₂ Emissions
    3. % of Population in Armed Forces (G20 only)
"""

# === 1. Import Libraries === #
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as colors
import matplotlib.cm as cm

# === 2. Load Dataset === #
df = pd.read_csv(r"C:\Users\cliff\OneDrive\Documents\Global Country Information Project\world-data-2023.csv")

# === 3. Data Cleaning and Preparation === #

# Standardize column names
df.rename(columns=lambda x: x.strip().lower().replace(" ", "_"), inplace=True)
df.rename(columns={'agricultural_land(_%)':'agricultural_land(%)'}, inplace=True)

# Map countries to continents
continent_map = {
    'Africa': ['Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cabo Verde', 'Cameroon', 
               'Central African Republic', 'Chad', 'Comoros', 'Congo', 'Djibouti', 'DR Congo', 'Egypt', 
               'Equatorial Guinea', 'Eritrea', 'Eswatini', 'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 'Guinea', 
               'Guinea-Bissau', 'Ivory Coast', 'Kenya', 'Lesotho', 'Liberia', 'Libya', 'Madagascar', 'Malawi', 
               'Mali', 'Mauritania', 'Mauritius', 'Morocco', 'Mozambique', 'Namibia', 'Niger', 'Nigeria', 
               'Rwanda', 'São Tomé and Príncipe', 'Senegal', 'Seychelles', 'Sierra Leone', 'Somalia', 
               'South Africa', 'South Sudan', 'Sudan', 'Tanzania', 'Togo', 'Tunisia', 'Uganda', 'Zambia', 'Zimbabwe'],
    'Asia': ['Afghanistan', 'Armenia', 'Azerbaijan', 'Bahrain', 'Bangladesh', 'Bhutan', 'Brunei', 'Cambodia', 
             'China', 'Cyprus', 'Georgia', 'India', 'Indonesia', 'Iran', 'Iraq', 'Israel', 'Japan', 'Jordan', 
             'Kazakhstan', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Lebanon', 'Malaysia', 'Maldives', 'Mongolia', 
             'Myanmar', 'Nepal', 'North Korea', 'Oman', 'Pakistan', 'Palestine', 'Philippines', 'Qatar', 
             'Saudi Arabia', 'Singapore', 'South Korea', 'Sri Lanka', 'Syria', 'Taiwan', 'Tajikistan', 
             'Thailand', 'Timor-Leste', 'Turkey', 'Turkmenistan', 'United Arab Emirates', 'Uzbekistan', 
             'Vietnam', 'Yemen'],
    'Europe': ['Albania', 'Andorra', 'Austria', 'Belarus', 'Belgium', 'Bosnia and Herzegovina', 'Bulgaria', 
               'Croatia', 'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 
               'Hungary', 'Iceland', 'Ireland', 'Italy', 'Kosovo', 'Latvia', 'Liechtenstein', 'Lithuania', 
               'Luxembourg', 'Malta', 'Moldova', 'Monaco', 'Montenegro', 'Netherlands', 'North Macedonia', 
               'Norway', 'Poland', 'Portugal', 'Romania', 'Russia', 'San Marino', 'Serbia', 'Slovakia', 
               'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'Ukraine', 'United Kingdom', 'Vatican City'],
    'North America': ['Antigua and Barbuda', 'Bahamas', 'Barbados', 'Belize', 'Canada', 'Costa Rica', 'Cuba', 
                      'Dominica', 'Dominican Republic', 'El Salvador', 'Grenada', 'Guatemala', 'Haiti', 
                      'Honduras', 'Jamaica', 'Mexico', 'Nicaragua', 'Panama', 'Saint Kitts and Nevis', 
                      'Saint Lucia', 'Saint Vincent and the Grenadines', 'Trinidad and Tobago', 'United States'],
    'South America': ['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Guyana', 'Paraguay', 
                      'Peru', 'Suriname', 'Uruguay', 'Venezuela'],
    'Oceania': ['Australia', 'Fiji', 'Kiribati', 'Marshall Islands', 'Micronesia', 'Nauru', 'New Zealand', 
                'Palau', 'Papua New Guinea', 'Samoa', 'Solomon Islands', 'Tonga', 'Tuvalu', 'Vanuatu'],
    'Antarctica': ['Antarctica']
}
country_to_continent = {country: cont for cont, countries in continent_map.items() for country in countries}
df['continent'] = df['country'].map(country_to_continent).fillna('Unknown')

# G20 countries
g20_countries = [
    "Argentina", "Australia", "Brazil", "Canada", "China", "France", "Germany", "India",
    "Indonesia", "Italy", "Japan", "Mexico", "Russia", "Saudi Arabia", "South Africa",
    "South Korea", "Turkey", "United Kingdom", "United States"
]

# Select relevant columns
columns = [
    'country', 'gdp', 'gross_primary_education_enrollment_(%)',
    'gross_tertiary_education_enrollment_(%)', 'fertility_rate', 'birth_rate',
    'population', 'armed_forces_size', 'continent', 'co2-emissions'
]
df_selected = df[columns].copy()

# Clean numeric columns
def clean_numeric_column(series):
    return (
        series.astype(str)
        .str.replace(',', '', regex=False)
        .str.extract(r'([\d\.]+)', expand=False)
        .astype(float)
    )

for col in ['gdp', 'gross_primary_education_enrollment_(%)', 'gross_tertiary_education_enrollment_(%)',
            'birth_rate', 'population', 'armed_forces_size', 'co2-emissions']:
    df_selected[col] = clean_numeric_column(df_selected[col])

# Drop missing values
df_cleaned = df_selected.dropna()

# Create derived columns
df_cleaned['gdp_per_capita'] = df_cleaned['gdp'] / df_cleaned['population']
df_cleaned['armed_forces_percent'] = (df_cleaned['armed_forces_size'] / df_cleaned['population']) * 100
df_cleaned['avg_education_enrollment'] = (
    df_cleaned['gross_primary_education_enrollment_(%)'] +
    df_cleaned['gross_tertiary_education_enrollment_(%)']
) / 2

# === 4. Visual 1: Birth Rate vs Gross Tertiary Education Enrollment === #

# All countries
plot_data = df_cleaned[df_cleaned['continent'] != 'Unknown']
plt.figure(figsize=(10, 12))
ax = plt.gca()
ax.set_facecolor('#2a2a2a')

sns.scatterplot(
    x='birth_rate',
    y='gross_tertiary_education_enrollment_(%)',
    hue='continent',
    size='population',
    sizes=(20, 1000),
    data=plot_data,
    alpha=0.6
)

handles, labels = ax.get_legend_handles_labels()
continent_labels = plot_data['continent'].dropna().unique()
continent_handles = [handles[labels.index(label)] for label in continent_labels]
legend = ax.legend(continent_handles, continent_labels, title='Continent')
plt.setp(legend.get_texts(), color='black')
plt.setp(legend.get_title(), color='black')

plt.xlabel('Birth Rate (%)', color='white')
plt.ylabel('Gross Tertiary Education Enrollment (%)', color='white')
plt.title('2023 Birth Rate vs Tertiary Education Enrollment', color='white', fontsize=22)
plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
plt.tick_params(colors='white')
plt.gcf().patch.set_facecolor('#2a2a2a')
plt.show()

# G20 only
df_g20 = df_cleaned[df_cleaned['country'].isin(g20_countries)]
plt.figure(figsize=(9, 12))
ax = plt.gca()
ax.set_facecolor('#2a2a2a')

sns.scatterplot(
    x='birth_rate',
    y='gross_tertiary_education_enrollment_(%)',
    size='population',
    sizes=(20, 1000),
    color='#3776ab',
    data=df_g20,
    legend=False,
    alpha=0.6
)

for _, row in df_g20.iterrows():
    plt.text(row['birth_rate'] + 0.5, row['gross_tertiary_education_enrollment_(%)'] + 0.5,
             row['country'], fontsize=11, color='white')

plt.xlabel('Birth Rate (%)', color='white')
plt.ylabel('Gross Tertiary Education Enrollment (%)', color='white')
plt.title('2023 Birth Rate vs Tertiary Education Enrollment (G20)', color='white', fontsize=22)
plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
plt.tick_params(colors='white')
plt.gcf().patch.set_facecolor('#2a2a2a')
plt.show()

# === 5. Visual 2: Armed Forces Size vs CO₂ Emissions === #

plot_data = df_cleaned[
    (df_cleaned['continent'] != 'Unknown') &
    (df_cleaned['co2-emissions'] > 0) &
    (df_cleaned['armed_forces_size'] > 0)
]

plt.figure(figsize=(10, 12))
ax = plt.gca()
ax.set_facecolor('#2a2a2a')

sns.scatterplot(
    x='armed_forces_size',
    y='co2-emissions',
    hue='gdp_per_capita',
    size='population',
    sizes=(20, 1000),
    data=plot_data,
    alpha=0.6,
    palette='viridis',
    legend=False
)

norm = colors.Normalize(vmin=plot_data['gdp_per_capita'].min(), vmax=plot_data['gdp_per_capita'].max())
sm = cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('GDP per Capita', color='white')
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Armed Forces Size (log)', color='white')
plt.ylabel('CO₂ Emissions (Tonnes, log)', color='white')
plt.title('2023 Armed Forces Size vs CO₂ Emissions', color='white', fontsize=20)
plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
plt.tick_params(colors='white')
plt.gcf().patch.set_facecolor('#2a2a2a')
plt.show()

# === 6. Visual 3: G20 – % of Population in Armed Forces === #

g20_data = df_cleaned[
    (df_cleaned['country'].isin(g20_countries)) &
    (df_cleaned['armed_forces_percent'] > 0)
]

g20_sorted = g20_data.sort_values(by='armed_forces_percent', ascending=True)

plt.figure(figsize=(14, 10))
ax = plt.gca()
ax.set_facecolor('#2a2a2a')

plt.barh(g20_sorted['country'], g20_sorted['armed_forces_percent'])

plt.xlabel('% of Population in Armed Forces', color='white')
plt.title('G20 Countries by % of Population in Armed Forces (2023)', color='white', fontsize=18)
plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
plt.tick_params(colors='white')
plt.gcf().patch.set_facecolor('#2a2a2a')
ax.set_facecolor('#2a2a2a')

for i, value in enumerate(g20_sorted['armed_forces_percent']):
    plt.text(value + 0.01, i, f"{value:.2f}%", color='white', va='center')

plt.tight_layout()
plt.show()
