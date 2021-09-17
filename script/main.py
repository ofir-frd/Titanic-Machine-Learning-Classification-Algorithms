

"""""""""""
Titatic Project


DF description:
    
Variable	Definition	            Key
survival 	Survival 	            0 = No, 1 = Yes
pclass 	    Ticket class 	        1 = 1st, 2 = 2nd, 3 = 3rd
sex 	    Sex 	                0 = female, 1= male
Age 	    Age in years 	
sibsp 	    # of siblings / spouses aboard the Titanic 	
parch 	    # of parents / children aboard the Titanic 	
ticket 	    Ticket number 	
fare 	    Passenger fare 	
cabin 	    Cabin number 	
embarked 	Port of Embarkation 	C/0 = Cherbourg, Q/1 = Queenstown, S/2 = Southampton

"""""""""""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn



### Clean and organize raw data
def titaticDFPreProcessing(titaticDF):

    
    ### Remove feature that are irrelevant for survivability.
    ### Consider Port of Embarkation removal due to insignificance.
    
    titaticDF = titaticDF.drop('Name', axis=1)
    titaticDF = titaticDF.drop('Ticket', axis=1)
    titaticDF = titaticDF.drop('Cabin', axis=1)      # consider use in the future (78% missing data)
    
    ### Transform all features to numbers
    
    titaticDF["Sex"] = titaticDF["Sex"].apply(lambda x: 0 if x == 'female' else 1) # female/male to 0/1
    titaticDF["Embarked"] = titaticDF["Embarked"].apply(lambda x: 0 if x == 'C' else ( 1 if x == 'Q' else 2)) # C to 0, Q to 1, S to 2





def main():
    
    ### Initial statues of the data
    
    print(titaticDF.head())
    
    titaticDFFeatureList= []
    
    [titaticDFFeatureList.append(feature) for feature in titaticDF.columns]
          
    for feature in titaticDFFeatureList:
        print("{} had {} % missing values".format(feature,np.round(titaticDF[feature].isnull().sum()/len(titaticDF)*100,2)))
    
     titaticDF = titaticDFPreProcessing(titaticDF)  # Initiate DF pre-processing
    
    
if __name__ == "__main__":
    main()
