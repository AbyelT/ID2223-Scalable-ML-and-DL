
import os
import modal

# NOT DAILY

LOCAL=False

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4","joblib","seaborn","sklearn","dataframe-image"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("abyel-hopsworks-secret"))
   def f():
       g()

def g():
    import hopsworks
    import pandas as pd

    # Authenticate to Hopsworks, get the project & dataset
    project = hopsworks.login()
    fs = project.get_feature_store()
    titanic_df = pd.read_csv("https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv")

    # Drop features without little or no predictive power
    titanic_df.drop(["Name", "Ticket", "Fare", "Cabin", "SibSp", "Parch"], axis=1, inplace=True)

    # Data preparation steps 

    ## Feature imputation: replace missing data with mean of age
    mean_age = titanic_df['Age'].mean()
    titanic_df['Age'].fillna(mean_age, inplace=True)
    
    ## Binning of age
    bins = [0, 15, 30, 45, 60, 75, 90]
    #labels = [0,1,2,3,4,5] # ['kid', 'young', 'adult', 'senior', 'old', 'superold']
    titanic_df['Age'] = pd.cut(titanic_df['Age'], bins=bins, labels=False, include_lowest=True)
    titanic_df['Age'] = titanic_df['Age'].astype('int64') #.map( {'male': 0, 'female': 1} )

    ## Cast datatype of a column into string
    titanic_df['Sex'] = titanic_df['Sex'].map( {'male': 0, 'female': 1} )

    ## convert category to numbers
    titanic_df['Embarked'] = titanic_df['Embarked'].fillna(titanic_df['Embarked'].mode()[0])
    titanic_df['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2], inplace=True)

    # print(titanic_df.isna().sum())
    # print(titanic_df.head(10))

    # Upload dataset as a feature group in Hopsworks feature store
    titanic_fg = fs.get_or_create_feature_group(
        name="titanic_modal",
        version=1,
        primary_key=["PassengerId", "Age", "Sex", "Pclass", "Embarked"], 
        description="Titanic dataset")
    titanic_fg.insert(titanic_df, write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()