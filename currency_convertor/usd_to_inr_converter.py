import requests as rq
import json as js
api = 'https://free.currconv.com/api/v7/convert?q=USD_INR&compact=ultra&apiKey=fa4d523fa7aa50fedc10'
reponse = rq.get(api)
print("USD to INR is ->",reponse.json()['USD_INR'])
