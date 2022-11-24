import os
import modal
    
BACKFILL=False
LOCAL=False

# DAILY

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4","joblib","seaborn","sklearn","dataframe-image"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("abyel-hopsworks-secret"))
   def f():
       g()


def generate_passenger(survived, passenger_id):
    """
    Returns a single iris flower as a single row in a DataFrame
    """
    import pandas as pd
    import random

    random_pclass = random.randint(1, 3)
    random_age = random.randint(1, 100)
    random_sex = random.randint(0, 1)
    random_sibsp = random.randint(1, 3)
    random_parch = random.randint(0, 2)

    passenger_df = pd.DataFrame({ "passengerid": [passenger_id],
                                "age": [random_age],
                                "sex": [random_sex],
                                "pclass": [random_pclass],
                                "sibsp": [random_sibsp],
                                "parch": [random_parch],
                      })
    passenger_df['survived'] = 0  
    
    return passenger_df


def get_random_titanic_passenger(passenger_id):
    """
    Returns a DataFrame containing one random titanpic passenger
    """
    import pandas as pd
    import random

    # randomly pick one of these 2 and write it to the featurestore
    coin_flip = random.randint(0, 1)

    if coin_flip >= 1:
        passenger_df = generate_passenger(1, passenger_id)
        print("This passenger Survived!")
    else:
        passenger_df = generate_passenger(0, passenger_id)
        print("This passenger Perished...")

    return passenger_df


def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    titanic_fg = fs.get_feature_group(name="titanic_modal", version=1)    

    # if BACKFILL == True:
    #     titanic_df = pd.read_csv("https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv")
    # else:
    #     titanic_df = pd.read_csv("./../titanic.csv") # get_random_iris_flower()

    titanic_df = titanic_fg.read()
    passenger_id = titanic_df['passengerid'].max() + 1
    passenger_df = get_random_titanic_passenger(passenger_id)

    print(passenger_df.head(5))
    
    # titanic_df = titanic_fs.get_or_create_feature_group(
    #     name="titanic_modal",
    #     version=1,
    #     primary_key=["passengerid"], 
    #     description="Titanic passenger dataset")
    
    titanic_fg.insert(passenger_df, write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()