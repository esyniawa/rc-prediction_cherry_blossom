import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Read the data
temp_df = pd.read_csv('./data/Japanese_City_Temps.csv')
temp_cities = set(temp_df.columns[1:])  # Exclude Date column

sakura_first_df = pd.read_csv('data/sakura_first_bloom_dates.csv')
sakura_first_cities = set(sakura_first_df['Site Name'])

sakura_full_df = pd.read_csv('data/sakura_full_bloom_dates.csv')
sakura_full_cities = set(sakura_full_df['Site Name'])

world_cities_df = pd.read_csv('data/worldcities.csv')
world_cities = set(world_cities_df['city_ascii'])

unique_cities = {
    'temp_df': len(temp_cities),
    'sakura_first_df': len(sakura_first_cities),
    'sakura_full_df': len(sakura_full_cities),
    'world_cities_df': len(world_cities)
}

print(unique_cities)
# Create pairs of datasets
datasets = {
    'Temperature': temp_cities,
    'Sakura First Bloom': sakura_first_cities,
    'Sakura Full Bloom': sakura_full_cities,
    'World Cities': world_cities
}

# Calculate overlaps
overlaps = []
for name1 in datasets:
    for name2 in datasets:
        if name1 < name2:  # Only calculate each pair once
            overlap_count = len(datasets[name1].intersection(datasets[name2]))
            overlaps.append({
                'Dataset 1': name1,
                'Dataset 2': name2,
                'Overlap Count': overlap_count
            })

# Create DataFrame with results
overlap_df = pd.DataFrame(overlaps)
print("City overlaps between datasets:")
print(overlap_df)

# The overlap counts
overlap_counts = {
    'Temperature vs Sakura First': len(temp_cities.intersection(sakura_first_cities)),
    'Temperature vs Sakura Full': len(temp_cities.intersection(sakura_full_cities)),
    'Temperature vs World Cities': len(temp_cities.intersection(world_cities)),
    'Sakura First vs Sakura Full': len(sakura_first_cities.intersection(sakura_full_cities)),
    'Sakura First vs World Cities': len(sakura_first_cities.intersection(world_cities)),
    'Sakura Full vs World Cities': len(sakura_full_cities.intersection(world_cities))
}
