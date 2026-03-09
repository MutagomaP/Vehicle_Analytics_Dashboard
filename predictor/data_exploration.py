import pandas as pd
from predictor.rwanda_map import create_rwanda_map_with_districts

# Data Exploration

def dataset_exploration(df):
    table_html = df.head().to_html(
        classes="table table-bordered table-striped table-sm",
        float_format="%.2f",
        justify="center",
        index=False,
     )
    return table_html


# Data description
def data_exploration(df):
    table_html = df.head().to_html(
        classes="table table-bordered table-striped table-sm",
        float_format="%.2f",
        justify="center",
         )
    return table_html


# Rwanda map with district distribution
def rwanda_map_exploration(df):
    """Generate Rwanda map showing vehicle client distribution by district"""
    return create_rwanda_map_with_districts(df)