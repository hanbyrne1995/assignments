# Final project classes

# Importing libraries
import numpy as np
import xarray as xr
from math import radians, isnan
from scipy.stats import linregress
import pandas as pd
from collections import defaultdict

class ModelInput:
    def __init__(self, modelID):
        '''
        Takes the input of a modelID (in this case URL), reads in the data and makes sure the coordinates are lat and lon; fixes dates so that they are standard format
        :param modelID: A unique ID for the model in the form of an openDAP url or in the form of the dataset itself (to handle the cases where models have been concatenated)
        '''
        self.modelID = modelID
        self.ds = self.ExecAllSteps()
        
    
    # starting by assuming I will be putting the URLs in directly; ultimately move to some kind of dictionary
    def LoadData(self, modelID):
        '''
        Loads the data from the openDAP server
        '''
        try:
            
            # checking if we are dealing with a url or dealing with the dataset itself
            if isinstance(modelID, str):
                self.ds = xr.open_dataset(self.modelID)
            
            else:
                self.ds = modelID
                
            # print('Successful data loading')
            
        except Exception as e:
            print(f'Error loading data for model {self.modelID}: {e}')
        
    # doing the data checks on the latitudes and longitudes
    def CleanCoords(self):
        '''
        Converts any coordinates that are not lat and lon into the appropriate dimension names
        '''
        try:
            if 'lat' not in self.ds.dims:
                if 'y' in self.ds.dims:
                    self.ds = self.ds.rename({'y': 'lat'})
                elif 'latitude' in self.ds.dims:
                    self.ds = self.ds.rename({'latitude': 'lat'})

            if 'lon' not in self.ds.dims:
                if 'x' in self.ds.dims:
                    self.ds = self.ds.rename({'x': 'lon'})
                elif 'longitude' in self.ds.dims:
                    self.ds = self.ds.rename({'longitude': 'lon'})
            
            # print('Successful coordinate cleaning')
            
        except Exception as e:
            
            print(f'Error checking and correcting coordinates: {e}')
    
    def FixDate(self):
        '''
        Converts all date formats to DateTime64 so no need to check start and end date types
        '''
        try:
            convertedTime = pd.to_datetime(self.ds.time.values.astype(str))
            self.ds['time'] = ('time', convertedTime)
        
        except Exception as e:
            print(f'Error in converting the time: {e}')
            
    def ExecAllSteps(self):
        self.LoadData(self.modelID)
        self.CleanCoords()
        self.FixDate()
        return self.ds

class Gradient:
    def __init__(self, modelInput):
        '''
        Calculates the area-weighted average in the two regions of interest; definitions per Seager et al 2022
        - gradient is calculated as western Pacific box average minus eastern Pacific box average
        - latitude and longitude extents defined per this paper
        
        :param modelInput: Instance of ModelInput class
        '''
        self.ds = modelInput
        self.modelName = self.SaveAttrs()
        self.boxE = None # initialising the East and West boxes to be filled
        self.boxW = None
        self.weightsE = None
        self.weightsW = None
        self.meansstE = None
        self.meansstW = None
        self.gradient = self.ExecuteAllSteps()
        
    def SaveAttrs(self):
        '''
        Saves the attributes of the dataset as the modelname that we can use to identify the model going forward
        '''
        
        try:
            selected_attrs = ['parent_source_id', 'variant_label'] 
            attributes = {attr: self.ds.attrs[attr] for attr in selected_attrs if attr in self.ds.attrs}
            self.modelName = '_'.join(attributes.values())
            
            # print('Successful attribute finding')
            
        except Exception as e:
            
            print(f'Error in loading attributes to dictionary: {e}')
        
        return self.modelName
        
    def SliceRegions(self):
        '''
        Slice the data into the regions defined in Seager et al 2022:
            East: lat: -3 to 3; lon: 190 to 270
            West: lat: -3 to 3; lon: 140 to 170
        '''
        try:
            # region_East
            lonminE, lonmaxE = 190, 270
            latminE, latmaxE = -3,3

            # region_West
            lonminW, lonmaxW = 140, 170
            latminW, latmaxW = -3, 3

            # slicing the data into these regions
            self.boxE = self.ds.ts.sel(lon = slice(lonminE, lonmaxE), lat = slice(latminE, latmaxE))
            self.boxW = self.ds.ts.sel(lon = slice(lonminW, lonmaxW), lat = slice(latminW, latmaxW))

            # print('Successful region slicing')

        except Exception as e:
            
            print(f'Error slicing regions: {e}')

    def CalculateGradient(self):
        '''
        Calculates the difference in weighted average sst variable (West - East)
        '''
        try:
            # note that this isn't regridded but there isn't areacello available so will estimate using latitude
            self.weightsE = np.cos(np.radians(self.boxE.lat))
            self.weightsW = np.cos(np.radians(self.boxW.lat))

            # calculating the mean monthly weighted SST for each box
            self.meansstE = self.boxE.weighted(self.weightsE).mean(('lat', 'lon'))
            self.meansstW = self.boxW.weighted(self.weightsW).mean(('lat', 'lon'))

            # calculate the temperature difference (W - E)
            self.gradient = self.meansstW - self.meansstE
            
            # print('Successful gradient calculation')
        
        except Exception as e:
            
            print(f'Error calculating gradient: {e}')
        
    def ExecuteAllSteps(self):
        self.SliceRegions()
        self.CalculateGradient()
        return self.gradient

class Trend:
    def __init__(self, gradient, minTrend):
        '''
        Calculates the trends for different start and end points for based on the input gradient time series
        
        Inputs:
        ::param gradient: instance of Gradient class (a gradient time series)
            
        Outputs:
            A dictionary of trends for the time periods indicated below
        '''
        self.gradient = gradient.gradient
        self.minTrend = minTrend
        self.modelName = gradient.modelName
        self.trendsDf = None
        self.trends = self.ExecuteAllSteps()
    
    
    def CalculateTrends(self):
        '''
        Calculates the moving interval trends (different beginning and end points)
        '''
        try:
            # initialising timing constants
            monStart = 1 # January
            monEnd = 12   # December
            yearStart = 1870 # Note that this can be changed 
            yearEnd = 2024 
            dayStart = 1
            dayEnd = 31
            interval = 1 # years
            trendLength = 120 # months
            self.trends = {}

            # calculating a time index to use in the polyfit
            indexTime = np.arange(len(self.gradient.time))

            # for loops to calculate the trends for each start and end year combination
            for start_year in range(yearStart, yearEnd+1 - self.minTrend, interval):
                for end_year in range(yearStart + self.minTrend, yearEnd+1, interval):

                    # fill the triangle where end date is before start date with NaNs
                    if start_year >= end_year:
                        self.trends[start_year, end_year] = np.nan
                    
                    # fill the triangle in for lengths that are shorter than the minTrend set
                    elif end_year - start_year < self.minTrend:
                        self.trends[start_year, end_year] = np.nan
                
                    else:
                        # create start and end dates for this period to subset the data
                        # check what the format of the time data is (checking the first element)
                        
                        start_date = np.datetime64(f'{start_year}-{monStart:02d}-{dayStart:02d}')
                        end_date = np.datetime64(f'{end_year}-{monEnd:02d}-{dayEnd:02d}')            

                        # create the subsetted dataset and subsetted time index
                        gradientSubset = self.gradient.sel(time = (self.gradient.time >= start_date) & (self.gradient.time <= end_date))
                        indexTimeSubset = indexTime[:len(gradientSubset.time)]

                        # calculate the slope and intercept; multiply slope by number of months for units of K per length of trend period
                        if len(indexTimeSubset) > 12*interval:
                            slope, intercept = np.polyfit(indexTimeSubset, gradientSubset, 1)
                            self.trends[start_year, end_year] = slope*trendLength
                        
                        else:
                            self.trends[start_year, end_year] = np.nan
                            
            # print('Successful trend calculation')
                            
        except Exception as e:
            
            print(f'Error calculating trends: {e}')
            
    def CreateDataFrame(self):
        '''
        Creates a dataframe of the trends that can be stored
        '''
        try:
            self.trendsDf = pd.DataFrame(list(self.trends.items()), columns = ['Year', 'Trend'])
            self.trendsDf[['start_year', 'end_year']] = pd.DataFrame(self.trendsDf['Year'].tolist(), index = self.trendsDf.index)
            self.trendsDf.drop('Year', axis = 1, inplace = True)
            self.trendsDf = self.trendsDf.pivot(index = 'end_year', columns = 'start_year', values = 'Trend')
            self.trendsDf = self.trendsDf.sort_index(ascending = False)
    
        
        except Exception as e:
            print(f'Error in creating dataframe: {e}')
        
        
    def ExecuteAllSteps(self):
        self.CalculateTrends()
        self.CreateDataFrame()
        return self.trends