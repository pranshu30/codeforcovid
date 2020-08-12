import pandas as pd
import numpy as np
import os
import pathlib



class CovidData:
    
    def __init__(self, smoothingdays=1,num_of_counties=100):
        pass
        self.smoothing_days = smoothingdays
        self.number_of_counties = num_of_counties
        self.local_path =  str(pathlib.Path(__file__).parent.resolve())
        
    
    def get_county_cases(self, max_number = 800):
        df_orig = pd.read_csv("https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv")
        state_to_code = pd.read_csv(os.path.join(self.local_path,'name_to_code_states.csv'))[['State','Code']].set_index('State').to_dict()['Code']
        df_orig = df_orig[~df_orig['fips'].isna()]
        df_orig['fips'] = df_orig['fips'].astype(int).apply(lambda x: str(x).zfill(5))
        
        df_orig['date'] = pd.to_datetime(df_orig['date'])
        df_orig = df_orig[df_orig['state'].isin(state_to_code.keys())]
        df_orig['state'] = df_orig['state'].apply(lambda x: state_to_code[x])
        df_orig['state-county'] = df_orig['state'] + '-' + df_orig['county']
        self.df_orig = df_orig
        self.df_orig = self.df_orig[~df_orig['state'].isin(['AK', 'HI'])]
        

        df = self.df_orig.drop_duplicates(['date', 'fips']).pivot(index = 'date', columns = 'fips', values = 'cases')
        
        df_nofilter = df.copy()
        df_diff = df.diff().fillna(0.0)

        self.df_cases = df_diff.copy()
        
        self.counties = df_orig.groupby('fips').sum().sort_values(by='cases', ascending=False).iloc[0:max_number].index.tolist()

    def get_google_mobility(self):
        google_url = "https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv?cachebust=cf3bfcad6ccbca1e"
        self.mobility_df=pd.read_csv(google_url)
        
    def get_county_coordinates(self):
        df_geo = pd.read_csv(os.path.join(self.local_path,'Geocodes_USA_with_Counties.csv'))
        df_geo = df_geo[~df_geo['state'].isin(['AK', 'HI'])]
        df_geo['county'] = df_geo['county'].apply(lambda x: x[:-1] if x[-1] == ' ' else x)
        df_geo['states-county'] = df_geo['state'] + '-' + df_geo['county']
        df_geo = df_geo.rename(index={'NY-New York City':'NY-New York'})
        df_geo['FIPS'] = df_geo['FIPS'].astype(str).apply(lambda x: x.zfill(5))
        self.fips_to_pretty = df_geo[['FIPS', 'Hover']].set_index('FIPS').to_dict()['Hover']
        production_counties = ['13103','26119', '38027']
        DC_counties = ['47137','56027', '41033', '39109']
        df_geo = df_geo[['states-county', 'latitude','longitude', 'county', 'population', 'FIPS']].dropna(axis=0).groupby('FIPS').max()[['latitude', 'longitude', 'population', 'states-county']].copy()
        self.df_geo = df_geo[df_geo.index.isin(self.counties)].copy()
        self.production_geo = df_geo.loc[production_counties,:].copy()
        self.dc_geo = df_geo.loc[DC_counties,:].copy()
        print(self.df_cases.shape)
        self.df_cases = self.df_cases.loc[:,self.df_cases.columns.isin(self.df_geo.index.tolist())]
        print(self.df_cases.shape)



if __name__ == '__main__':
    data = CovidData()
    data.get_county_cases()
    print(data.df_cases.head())
    data.get_county_coordinates()
    print(data.df_geo.head())
    print(data.production_geo.head())
    print(data.dc_geo.head())