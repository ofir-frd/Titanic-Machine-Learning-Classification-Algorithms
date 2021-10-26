# Titanic
Predict survival on the Titanic

A Competition from Kaggle:

https://www.kaggle.com/c/titanic



    #########################
    #######  Analysis #######
    #########################
    
    
Initial Overview with seaborn pairplot:

![pairplot](https://user-images.githubusercontent.com/85901822/138879417-65561237-bb87-4f4e-92cb-8f48d0d9fc90.png)
    
    
    
passengers class distribution:

![Passangers class distribution](https://user-images.githubusercontent.com/85901822/133884139-d800d625-70a1-45df-b189-b56865739207.png)



The quantity and percentage for dead and alive passengers per class:

![LifeandDeathByClass](https://user-images.githubusercontent.com/85901822/135080688-139b97d1-93dd-43df-87ef-8eb4aefab14d.png)



The distribution by the age of dead and alive passengers:

![LifeandDeathByAge](https://user-images.githubusercontent.com/85901822/135081332-6cbf61ad-10f9-4c47-9722-3c67d2d2fd6f.png)



The distribution by the number of siblings/spouses of dead and alive passengers:

![LifeandDeathBySibs](https://user-images.githubusercontent.com/85901822/135587178-5b7844ed-e2c9-4e99-8bb5-461cda23b901.png)



The distribution by the number of parents/children of dead and alive passengers:

![LifeandDeathByChilds](https://user-images.githubusercontent.com/85901822/135590033-e501e778-b8ec-4a4f-b559-ef27603a0544.png)



The quantity and percentage for dead and alive passengers per boarding location:

![LifeandDeathByEmbarked](https://user-images.githubusercontent.com/85901822/135593653-849dec90-273a-4cf6-b41d-77dcc183e90a.png)


    #########################
    ###  Machine Learning ###
    #########################


    ##### Decision Tree #####


Sample of **desition tree classification** with a max depth of 4.

The accuracy score of this algorithm on the initial DB is 82.51%, and on the test DB is **96.69%**.

![Figure 2021-10-14 183414](https://user-images.githubusercontent.com/85901822/137436008-e4e6cbf7-9a79-4db2-b602-618608be311f.png)



Examination of the decision tree classification algorithm efficiency based on the tree depth range of 1 to 10.

accuracy score:

![image](https://user-images.githubusercontent.com/85901822/137876447-ff853047-c587-4b45-b7d1-08db3bb94cdc.png)

![Accuracy Values per tree depth](https://user-images.githubusercontent.com/85901822/137874555-f53ea9dc-7f0d-40f0-9ce9-8119e5445f63.png)

![desicion tree box plot](https://user-images.githubusercontent.com/85901822/137873646-5f469a14-b665-42df-bb81-45a54644ef54.png)


    #### Random Forest ####


Accuracy scores of random forest classification algorithm as a function of tree depth (3 to 5) and amount of trees (100 to 1000 in steps of 100):

![RandomForest Train](https://user-images.githubusercontent.com/85901822/138872195-d8441273-67b6-4d74-932d-f3100169f46c.png)

![RandomForest Test](https://user-images.githubusercontent.com/85901822/138872211-9129bdf9-d93e-461a-8599-e117385fe16f.png)

