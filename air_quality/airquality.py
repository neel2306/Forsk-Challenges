import requests
import json 
user_location = str(input("Please enter your city name :"))
user_location_api = "http://api.openweathermap.org/geo/1.0/direct?q="+user_location+"&limit=1&appid=82905910d99fffd1ceab2be5d569b3ed"
geo = requests.get(user_location_api)

#latitude.
lat = str(geo.json()[0]['lat'])

#longitude.
lon = str(geo.json()[0]['lon'])

air_quality_api = "http://api.openweathermap.org/data/2.5/air_pollution?lat="+lat+"&lon="+lon+"&appid=82905910d99fffd1ceab2be5d569b3ed"

air_qual = requests.get(air_quality_api).json()['list'][0]
flag = air_qual['main']['aqi']
print("Air Quality Index --> Possible values: 1, 2, 3, 4, 5.\n \nWhere 1 = Good, 2 = Fair, 3 = Moderate, 4 = Poor, 5 = Very Poor.")
print("\nAir Quality Index of", user_location, "is", flag)
