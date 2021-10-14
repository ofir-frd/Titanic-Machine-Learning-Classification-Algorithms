

"""""""""""
Titatic Project


DF description:
    
Variable	Definition	            Key
survival 	Survival 	            0 = No, 1 = Yes
pclass 	    Ticket class 	        1 = 1st, 2 = 2nd, 3 = 3rd
sex 	    Sex 	                0 = female, 1= male
Age 	    Age in years 	
Sibsp 	    # of siblings / spouses aboard the Titanic 	
Parch 	    # of parents / children aboard the Titanic 	
Ticket 	    Ticket number 	
Fare 	    Passenger fare 	
Cabin 	    Cabin number 	
Embarked 	Port of Embarkation 	C/0 = Cherbourg, Q/1 = Queenstown, S/2 = Southampton

"""""""""""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import re

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

### Apply decision tree regressor to predict the survivability of passangers
### Returns accuracyValue and crossValScore
def run_decision_tree_classifier(titaticTestDF, titaticTrainDF, titaticTestResults, maxDepth, randomState):

    x = titaticTrainDF.drop('Survived', axis=1)

    y = titaticTrainDF['Survived']

    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, train_size = 0.8,
                                                        random_state = randomState, shuffle= True)

    dtObject = DecisionTreeClassifier(max_depth = maxDepth)

    dtObject.fit(x_train, y_train)

    PredictedTrain = dtObject.predict(x_test)
    
    accuracyValueTrain = accuracy_score(y_test, PredictedTrain)
    
    crossValScoreTrain = cross_val_score(dtObject, x, y, cv = 10)
    
    PredictedTest = dtObject.predict(titaticTestDF)
    
    accuracyValueTest = accuracy_score(titaticTestResultsClean, PredictedTest)
    
    crossValScoreTest = cross_val_score(dtObject, x, y, cv = 10)

    return accuracyValueTrain, crossValScoreTrain, accuracyValueTest, crossValScoreTest
    
    
### Post-processing data plots
def plot_processed_data(titaticTestDF, titaticTrainDF):
    
    ### Presents the relative division of classes in the train and test databases
    
    figure, axis = plt.subplots(1, 2)

    axis[0].pie(titaticTestDF["Pclass"].value_counts(), labels= titaticTestDF["Pclass"].value_counts().index, autopct='%1.0f%%')
    axis[0].set_title('Test BD:\n passangers per class')

    axis[1].pie(titaticTrainDF["Pclass"].value_counts(), labels= titaticTrainDF["Pclass"].value_counts().index, autopct='%1.0f%%')
    axis[1].set_title('Train BD:\n passangers per class')
    plt.show()
    
    
    ### The quantity and presentage for dead and alive passangers per class
    
    firstClassMortality = titaticTrainDF.loc[titaticTrainDF['Pclass'] == 1, :]['Survived'].value_counts()
    secondClassMortality = titaticTrainDF.loc[titaticTrainDF['Pclass'] == 2, :]['Survived'].value_counts()
    thirdClassMortality = titaticTrainDF.loc[titaticTrainDF['Pclass'] == 3, :]['Survived'].value_counts()
    
    classMortality = [firstClassMortality[0],secondClassMortality[0],thirdClassMortality[0]],[firstClassMortality[1],secondClassMortality[1],thirdClassMortality[1]]
    
    fig, axis = plt.subplots(figsize=(5,10))
    
    plt.barh(['1st Class','2nd Class','3rd Class'],classMortality[0], label="Died")
    plt.barh(['1st Class','2nd Class','3rd Class'],classMortality[1], label="Survived", left=classMortality[0])
    
    axis.set_title('Survived pasangers per Class')
    axis.set_xlabel("Number of passangers")
    axis.legend(ncol=2, loc="lower right", frameon=True)
    
    for i in range(0,len(classMortality[0])): 
        percentageValueSurvived = (classMortality[1][i] / (classMortality[0][i] + classMortality[1][i])) * 100
        percentageValueDied = 100 - percentageValueSurvived
        
        percentageValueSurvived = percentageValueSurvived.round(1)
        percentageValueDied = percentageValueDied.round(1)
        
        axis.annotate(str(percentageValueSurvived) + '%', ((classMortality[0][i] + classMortality[1][i])-(classMortality[1][i]/1.3),i))
        axis.annotate(str(percentageValueDied) + '%', (classMortality[0][i]/(3.5-(i*0.5)),i))
        
        
    plt.show()
    
    
    ### The distribution by age of dead and alive passangers

    fig03, axis = plt.subplots(1, 2)
    
    axis[0].hist(titaticTrainDF.loc[titaticTrainDF['Survived'] == 0, :]['Age'], color = 'royalblue')
    axis[1].hist(titaticTrainDF.loc[titaticTrainDF['Survived'] == 1, :]['Age'], color = 'darkorange')
    
    axis[0].set_title('Died passengers by age')
    axis[0].set_xlabel("Number of passengers")
    
    axis[1].set_title('Survived passengers by age')
    axis[1].set_xlabel("Number of passengers")
    
    plt.show()
    
    
    ### The distribution by number of siblings/spouses of dead and alive passangers
    
    fig04, axis = plt.subplots(1, 2)

    axis[0].hist(titaticTrainDF.loc[titaticTrainDF['Survived'] == 0, :]['SibSp'], color = 'royalblue')
    axis[1].hist(titaticTrainDF.loc[titaticTrainDF['Survived'] == 1, :]['SibSp'], color = 'darkorange')
    
    axis[0].set_title('Died passengers')
    axis[0].set_xlabel("Num. of siblings/spouses")
    
    axis[1].set_title('Survived passengers')
    axis[1].set_xlabel("Num. of siblings/spouses")
    
    plt.show()
       
    
    ### The distribution by number of parents/children of dead and alive passangers
    
    fig05, axis = plt.subplots(1, 2)

    axis[0].hist(titaticTrainDF.loc[titaticTrainDF['Survived'] == 0, :]['Parch'], color = 'royalblue')
    axis[1].hist(titaticTrainDF.loc[titaticTrainDF['Survived'] == 1, :]['Parch'], color = 'darkorange')
    
    axis[0].set_title('Died passengers')
    axis[0].set_xlabel("Num. of parents/children")
    
    axis[1].set_title('Survived passengers')
    axis[1].set_xlabel("Num. of parents/children")
    
    plt.show()
    
    
    ### The quantity and presentage for dead and alive passangers per boarding location
    
    firstEmbarked = titaticTrainDF.loc[titaticTrainDF['Embarked'] == 0, :]['Survived'].value_counts()
    secondEmbarked = titaticTrainDF.loc[titaticTrainDF['Embarked'] == 1, :]['Survived'].value_counts()
    thirdEmbarked = titaticTrainDF.loc[titaticTrainDF['Embarked'] == 2, :]['Survived'].value_counts()
    
    embarkedMortality = [firstEmbarked[0],secondEmbarked[0],thirdEmbarked[0]],[firstEmbarked[1],secondEmbarked[1],thirdEmbarked[1]]
    
    fig06, axis = plt.subplots(figsize=(5,10))
    
    plt.barh(['Cherbourg','Queenstown','Southampton'],embarkedMortality[0], label="Died")
    plt.barh(['Cherbourg','Queenstown','Southampton'],embarkedMortality[1], label="Survived", left=embarkedMortality[0])

    axis.set_title('Survived/Died pasangers per embarked location')
    axis.set_xlabel("Number of passangers")
    axis.legend(ncol=2, loc="lower right", frameon=True)
    axis.set
    
    for i in range(0,len(embarkedMortality[0])): 
        percentageValueSurvived = (embarkedMortality[1][i] / (embarkedMortality[0][i] + embarkedMortality[1][i])) * 100
        percentageValueDied = 100 - percentageValueSurvived
        
        percentageValueSurvived = percentageValueSurvived.round(1)
        percentageValueDied = percentageValueDied.round(1)
        
        axis.annotate(str(percentageValueSurvived) + '%', ((embarkedMortality[0][i] + embarkedMortality[1][i])-(embarkedMortality[1][i]/1.3),i))
        axis.annotate(str(percentageValueDied) + '%', (embarkedMortality[0][i]/(3.5-(i*0.5)),i+0.1))
        
    plt.show()


### Removes Passanger ID feature
def remove_passanger_id (titaticDF)
    return titaticDF.drop('PassengerId', axis=1) 


### Removes row from correlated DF that were taken out during the pre processing
### removal is conducted by passanger ID values
def correlate_results_by_id(titaticTestDF, titaticTestResults):
    
    titaticTestResultsClean = titaticTestResults
    
    for indexValue in titaticTestResults.index:
        if titaticTestDF['PassengerId'].get(titaticTestResults.index[indexValue], default = False) is False:
            titaticTestResultsClean = titaticTestResultsClean.drop(index=indexValue)
            
    return titaticTestResultsClean
   
    
### Clean and organize raw data
def titaticdf_pre_processing(titaticDF):

    ### Remove feature that are irrelevant for survivability.
    ### Consider Port of Embarkation removal due to insignificance.
    
    titaticDF = titaticDF.drop('Name', axis=1)
    titaticDF = titaticDF.drop('Ticket', axis=1)
    titaticDF = titaticDF.drop('Fare', axis=1)
    titaticDF = titaticDF.drop('Cabin', axis=1)      # consider use in the future (78% missing data)
    
    ### Transform all features to numbers
    
    titaticDF["Sex"] = titaticDF["Sex"].apply(lambda x: 0 if x == 'female' else 1) # female/male to 0/1
    titaticDF["Embarked"] = titaticDF["Embarked"].apply(lambda x: 0 if x == 'C' else ( 1 if x == 'Q' else 2)) # C to 0, Q to 1, S to 2

    ### Managing missing data:
    
   # titaticDF["Fare"] = titaticDF["Fare"].fillna(titaticDF["Fare"].mean())                # fill the one missing Fare data point with mean value    
        
    ### !! 1st trial to solve this challange will include removal of 20% of the data due to missing age parameter !! ###
    titaticDF.dropna(how = 'any', axis = 0, inplace = True)

    print("Data Pre-processing completed...")
    
    return titaticDF


### Prints DF head and missing values status for each feauture
def print_df_status(titaticDF):

    print(titaticDF.head())
        
    titaticDFFeatureList= []
        
    [titaticDFFeatureList.append(feature) for feature in titaticDF.columns]
              
    for feature in titaticDFFeatureList:
        print("{} had {} % missing values".format(feature,np.round(titaticDF[feature].isnull().sum()/len(titaticDF)*100,2)))
        
    print ('\n')


def main():
    
    titaticTestDF = pd.read_csv('test.csv')                   # Database Import
    titaticTrainDF = pd.read_csv('train.csv')                 # Database Import
    titaticTestResults = pd.read_csv('gender_submission.csv') # Database Import
    
    ### Initial statues of the data

    print_df_status(titaticTestDF)
    print_df_status(titaticTrainDF)
    print_df_status(titaticTestResults)
    
    titaticTestDF = titaticdf_pre_processing(titaticTestDF)  # Initiate DF pre-processing
    titaticTrainDF = titaticdf_pre_processing(titaticTrainDF)  # Initiate DF pre-processing
    titaticTestResults = correlate_results_by_id(titaticTestDF, titaticTestResults)
        
    ### Once titaticTestResults are collocated with the test DF
    ### PassangerID feature is no more needed
    titaticTestDF = remove_passanger_id(titaticTestDF)
    titaticTrainDF = remove_passanger_id(titaticTrainDF)
    titaticTestResults = remove_passanger_id(titaticTestResults)
    
    print_df_status(titaticTestDF)
    print_df_status(titaticTrainDF)
    print_df_status(titaticTestResults)
        
    plot_processed_data(titaticTestDF, titaticTrainDF)
    
    print(run_decision_tree_classifier(titaticTestDF, titaticTrainDF, titaticTestResults, 10, 42))
    
if __name__ == "__main__":
    main()
