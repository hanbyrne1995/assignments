# Final project functions

# importing libraries
import numpy as np
from collections import defaultdict
import xarray as xr
import cftime
import os
import random

from FP_classes import ModelInput

def ClassifyHistModelsLite(urlList):
    '''
    Function that classifies historical models according to whether they span the full period or not
    
    Inputs:
        urlList: list of historical model urls (per the openDAP format)
        
    Outputs:
        a dictionary where the keys are either 'Full' or the modelID (i.e., modelName and variant label)
    '''
    
    # initialising a dictionary
    histModels = defaultdict(list)
    
    char = '_'
    char2 = '.'
    fullDates = '185001-201412'
    
    for url in urlList:
        # finding Amon
        indAmon = url.index(char)

        # find the modelName
        indModelNameStart = url.index(char, indAmon+1) + 1
        indModelNameEnd = url.index(char, indModelNameStart+1)

        modelName = url[indModelNameStart:indModelNameEnd]

        # find the run variant
        indVariantStart = url.index(char, indModelNameEnd+1) + 1
        indVariantEnd = url.index(char, indVariantStart+1)

        modelVariant = url[indVariantStart:indVariantEnd]

        # finding the date range
        indDateStart = url.index(char, indVariantEnd+2) + 1
        indDateEnd = url.index(char2, indDateStart)

        dateRange = url[indDateStart:indDateEnd]
        
        # now checking whether full or not
        if dateRange == fullDates:
            key = 'Full'
        else:
            key = modelName + '_' + modelVariant
            
        histModels[key].append(url)
        
    histModels = dict(histModels)
    return histModels

def CreateScenarioDictionary(modelListScenario):
    '''
    Creates a dictionary of scenarios that are the right length of time for this study. The keys are the source_ids of the models.
    
    Inputs:
        modelListScenario: the filtered list of URLs from the scenario models that we are interested in
        
    Outputs:
        a dictionary where the keys are the scenarioIDs for the models and the values are the URLs of models that are the right length for this study; Note that scenarioIDs are a combination of the model and
        parent variant (i.e., the historical model that seeded the model)
    '''
    
    # initialise a defaultdict to store the URLs
    scenarioModels = defaultdict(list)

    # dates for checking that the scenario fits into the right time
    nModels = len(modelListScenario)
    count = 0
    start_year = 2015
    monStart = 1
    dayStart = 31 # because these are sometimes on the 16th of the month
    end_year = 2022
    monEnd = 12
    dayEnd = 1  # because these are sometimes on the 16th of the month as well

    for model in modelListScenario:
        count +=1
        # check that the scenario actually spans the time that we need before saving it
        ds = xr.open_dataset(model)
        sourceID = ds.attrs['source_id']
        parentVariant = ds.attrs['parent_variant_label']
        scenarioID = sourceID + '_' + parentVariant

        # run two versions of checking depending on the format that the date time information is in
        if isinstance(ds.time.values[0], np.datetime64):
                start_date = np.datetime64(f'{start_year}-{monStart:02d}-{dayStart:02d}')
                end_date = np.datetime64(f'{end_year}-{monEnd:02d}-{dayEnd:02d}')            

        elif isinstance(ds.time.values[0], cftime.DatetimeNoLeap):
            start_date = cftime.DatetimeNoLeap(start_year, monStart, dayStart)
            end_date = cftime.DatetimeNoLeap(end_year, monEnd, dayEnd)

        # save the URL
        if (ds.time[0] <= start_date) & (ds.time[-1] >= end_date):
            # append the value to the list using the source_id as the key
            scenarioModels[scenarioID].append(model)
        
        # keeping track of progress
        print(f'Scenario dictionary complete: {count} / {nModels}')

    # save the default dict as a dict
    scenarioModels = dict(scenarioModels)
    
    return scenarioModels

def MakeChangeDir(path):
    '''
    Function that checks to see if a directory given by a path exists; if it doesn't exist, then it creates that directory and changes into it; if it does exist it changes to that directory
    
    Inputs:
        path: path to directory that's in question
    '''
    
    if os.path.isdir(path):
        os.chdir(path)
    else:
        os.mkdir(path)
        os.chdir(path)

def ConcatModels(modelDict):
    
    '''
    For the models that weren't complete from the classify models stage (i.e., didn't cover the full period), concatenate all of the models saved
    
    Inputs:
        modelDict: a dictionary pertaining to a specific model (i.e., from the output of ClassifyModels, this should be modelsDict['Name of model'])
        
    Outputs:
        a full xarray dataset that has been concatenated to the full length
    
    '''
    # open one of the datasets
    ds = xr.open_dataset(modelDict[0])
    
    if len(modelDict) > 1:
    
        for i in range(1, len(modelDict)):
            ds2 = xr.open_dataset(modelDict[i])
            ds = xr.concat([ds, ds2], dim = 'time')
            
    else:
        
        ds = ds
    
    # sorting the data by time
    ds = ds.sortby('time')
    
    # run a check to make sure that the dataset encompasses the full period
    start_year = 1850
    monStart = 1
    dayStart = 31 # because these are sometimes on the 16th of the month
    end_year = 2014
    monEnd = 12
    dayEnd = 1  # because these are sometimes on the 16th of the month as well
    
    # run two versions of checking depending on the format that the date time information is in
    if isinstance(ds.time.values[0], np.datetime64):
            start_date = np.datetime64(f'{start_year}-{monStart:02d}-{dayStart:02d}')
            end_date = np.datetime64(f'{end_year}-{monEnd:02d}-{dayEnd:02d}')            

    elif isinstance(ds.time.values[0], cftime.DatetimeNoLeap):
        start_date = cftime.DatetimeNoLeap(start_year, monStart, dayStart)
        end_date = cftime.DatetimeNoLeap(end_year, monEnd, dayEnd)

    # return the full model
    if (ds.time[0] <= start_date) & (ds.time[-1] >= end_date):
        return ds

    # run an error message if the datasets are incomplete
    else:
        raise ValueError('Concatenated models do not span full period')
    
    return ds
    

def ExtendPeriod(key, modelInput, scenarioModels):
    '''
    Function that takes in the modelInput output and scenarioModels and combines to make one ds that has the full period from the start of the historical model to the end of the scenario model.
    
    Inputs:
        modelInput: an instance of the ModelInput class
        scenarioModels: dictionary of the scenario models labelled with their source_ids
    
    Outputs:
        modelFullPeriod: a model that spans the full period of the historical and scenario
        match: a tuple containing an identifier for the scenario model that the historical model was concatenated with and the note of random versus non-random
            NOTE: this is the variant label and not the parent variant label
    '''
    # the way that this runs depends on whether there's a scenario that matches the historical run in terms of parent
    
    
    if key in list(scenarioModels.keys()):
            
        # execute the code for the situation in which we can directly concatenate the arrays
        dsScenario = ModelInput(scenarioModels[key][0]).ds
        modelFullPeriod = xr.concat([modelInput.ds, dsScenario], dim = 'time')
        
    else:

        # execute the code for the situation in which you have to randomise the assigment
        modelHistID = modelInput.ds.attrs['source_id']
        runHist = modelInput.ds.attrs['variant_label']

        # now randomly select one of the models from the same source_id
        # create a list of source_IDs (as in model names) so that we can choose an index from that list and randomise
        
        # first have to flatten any of them that might have subdictionaries
        scenarioModelsFlat = {}

        for key, value in scenarioModels.items():
            if len(value) > 1:
                counter = 0
                for subValue in value:
                    subValueList = []
                    subValueList.append(subValue)
                    scenarioModelsFlat[key + '_' + str(counter)] = subValueList
                    counter += 1
            else:
                scenarioModelsFlat[key] = value
    
        scenarioModelSource = []

        for i in list(scenarioModelsFlat.keys()):
            index = i.index('_')
            modelSource = i[:index]
            scenarioModelSource.append(modelSource)

        # create a mask for those model sources that match
        histMask = [modelID == modelHistID for modelID in scenarioModelSource]

        # create a list of integers to be the indices
        indices = list(range(len(list(scenarioModelsFlat.keys()))))

        # filter for only the indices that have True in the mask
        indicesMatch = [index for index, flag in zip(indices, histMask) if flag]

        # select a random index for the source
        scenarioRandom = list(scenarioModelsFlat)[random.choice(indicesMatch)]
        dsScenario = ModelInput(scenarioModelsFlat[scenarioRandom][0]).ds
        modelFullPeriod = xr.concat([modelInput.ds, dsScenario], dim = 'time')
    
    return modelFullPeriod

def RemoveClimatology(modelFullPeriod):
    '''
    Function that removes the climatology from the ts variable in the dataset
    
    Input: modelFullPeriod in this case is the full length concatenated ds that comes from combing hist and scen data
    
    Output: returns a ds with the climatology removed
    '''

    gb = modelFullPeriod.groupby('time.month')
    dsAnom = gb - gb.mean(dim = 'time')
    dsAnom.attrs = modelFullPeriod.attrs.copy()
    return dsAnom

