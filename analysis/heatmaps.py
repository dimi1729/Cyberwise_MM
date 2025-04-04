import pandas as pd
import folium
from folium.plugins import HeatMap



def generate_map(column_name):
    df = pd.read_csv("data/locations.csv")
    df = df[(df[column_name+' Lat'] != 'Not found') & (df[column_name+' Long'] != 'Not found') & df[column_name+' Lat'].notna() & df[column_name+' Long'].notna()]
    m = folium.Map()

    heat_data = df[[column_name+' Lat', column_name+' Long']].astype(float).values.tolist()
    HeatMap(heat_data, radius = 8).add_to(m)

    m.save(column_name+".html")

generate_map("Source")
generate_map("Destination")