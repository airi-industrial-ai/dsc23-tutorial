import gdown
import pandas as pd

def load_gas_supply():
    gdown.download('https://drive.google.com/uc?id=1kGp65UabcFcKAHSM4K2MgkPAYgbyhQj5', 'gas_supply.csv')
    target = pd.read_csv(
        'gas_supply.csv',
        skiprows=5,
        header=None,
        date_parser=pd.to_datetime,
        index_col=0)
    target = target.sort_index().asfreq('W-Fri')
    target = target.rename(columns={1: 'supply'})
    target.index.name = 'date'
    target = target.iloc[:1500]
    return target
