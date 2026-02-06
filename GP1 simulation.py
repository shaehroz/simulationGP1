import numpy as np
import pandas as pd
import ast #used to convert the driver location into (x,y) coordinates
from dataclasses import dataclass #used to create classes easier and neater

##Start by loading the data sets into python that were provided to us.
##For the commands below just replace the directory with your own, within the ""
riders = pd.read_excel(r"C:\Users\shaeh\Desktop\edinburgh-notes\Simulation\Group Project 1\riders.xlsx")

drivers = pd.read_excel(r"C:\Users\shaeh\Desktop\edinburgh-notes\Simulation\Group Project 1\drivers.xlsx")

# Prints the first 5 rows of the spreadsheet, just to make sure it's loaded correctly.
print(riders.head())
print(drivers.head())
# Tells what type each column is in the spreadsheets
print(riders.dtypes)
print(drivers.dtypes)

def convert_to_xy(v): #function to convert from a string to (x,y) coordinates (tuple)
    if isinstance(v, tuple):
        return v
    return ast.literal_eval(v) #ast converts a string to a tuple. i.e. try s = "(3.2, 7.8)" // ast.literal_eval(s) <- this would output (3.2, 7.8) as coordinates

#Now we will use apply which is a pandas function that carries out whatever is within the brackets to the whole row/column
#Drivers conversion
drivers["initial_location"] = drivers["initial_location"].apply(convert_to_xy)
drivers["current_location"] = drivers["current_location"].apply(convert_to_xy)

#Convert the tuples to list. Pandas will see we have a list of coordinates so will automatically convert it into two columns.
#For current location we will call them x,y and for intial location we will call them initial x, initial y

drivers[["x", "y"]] = pd.DataFrame(drivers["current_location"].tolist(),index=drivers.index)
drivers[["initial_x", "initial_y"]] = pd.DataFrame(drivers["initial_location"].tolist(),index=drivers.index)

#Carry out similar procedure for riders
#Riders conversion
riders["pickup_location"]  = riders["pickup_location"].apply(convert_to_xy)
riders["dropoff_location"] = riders["dropoff_location"].apply(convert_to_xy)

riders[["pickup_x", "pickup_y"]] = pd.DataFrame(riders["pickup_location"].tolist(),index=riders.index)
riders[["dropoff_x", "dropoff_y"]] = pd.DataFrame(riders["dropoff_location"].tolist(),index=riders.index)

@dataclass
class Driver:
    id: int #driver id
    x: float #horizontal location on grid
    y: float #vertical location on grid
    busy: bool = False #if driver is busy or not, auto assigned not busy
    earnings: float = 0.0 #initial earnings, autoset to 0#
    miles_total: float = 0.0
    miles_paid: float = 0.0
    busy_time: float = 0.0
@dataclass
class Rider:
    id: int
    request_time: float
    pickup_x: float
    pickup_y: float
    dropoff_x: float
    dropoff_y: float
    status: str = "waiting" #waiting/matched/cancelled/completed
    cancel_time: float | None = None # "float | None = None" just means that this variable can be a float OR None
    driver_id: int | None = None

#Carry out instantiation, using the class above to actually create an object
#The class describes what a driver looks like, the object will contain actual info

driver_objs = [
    Driver(
        id=int(row["id"]),
        x=float(row["x"]),
        y=float(row["y"])
    )
    for _, row in drivers.iterrows() #in laymans: go through every row in the dataframe
]
