import requests
import json
import csv
import logging


API_KEY = 'ihU0CD4DEQgpa+OW+EsqHQ==n6COksm5iDFI1UhX'
API_URL = 'https://api.api-ninjas.com/v1/cars?'

logging.basicConfig(filename="get_vehicle_logs.log", level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def get_api_vehicles():
    start_year = 1985
    end_year = 2022
    limit = 50
    
    with open('vehicles.csv', 'a', newline='') as csvfile:
        fieldnames = ['city_mpg', 'class', 'combination_mpg', 'cylinders', 'displacement', 'drive', 'fuel_type', 'highway_mpg', 'make', 'model', 'transmission', 'year']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        while start_year <= end_year:
            print(f"year: {start_year}")
            url = API_URL + f"limit={limit}&fuel_type=gas&year={start_year}"
            
            response = requests.get(url, headers={'X-Api-Key': API_KEY})
            if response.status_code == requests.codes.ok:
                data = response.json()
                for vehicle in data:
                    writer.writerow(vehicle)
                start_year += 1
            else:
                print("Error", response.status_code, response.text)
                break
        

if __name__ == "__main__":
    get_api_vehicles()