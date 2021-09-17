

"""""""""""
Titatic Project


DF description:
    
Variable	Definition	          Key
survival 	Survival 	            0 = No, 1 = Yes
pclass 	    Ticket class 	      1 = 1st, 2 = 2nd, 3 = 3rd
sex 	    Sex 	
Age 	    Age in years 	
sibsp 	    # of siblings / spouses aboard the Titanic 	
parch 	    # of parents / children aboard the Titanic 	
ticket 	    Ticket number 	
fare 	    Passenger fare 	
cabin 	    Cabin number 	
embarked 	Port of Embarkation 	C = Cherbourg, Q = Queenstown, S = Southampton



"""""""""""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn


def main():
    
    ### Initial statues of the data
    
    print(titaticDF.head())
    
    titaticDFFeatureList= []
    
    [titaticDFFeatureList.append(feature) for feature in titaticDF.columns if titaticDF[feature].isnull().sum()>1]
          
    for feature in titaticDFFeatureList:
        print("{} had {} % missing values".format(feature,np.round(titaticDF[feature].isnull().sum()/len(titaticDF)*100,2)))
    

    
    
if __name__ == "__main__":
    main()
