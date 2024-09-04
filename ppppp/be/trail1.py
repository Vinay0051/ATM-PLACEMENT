import csv
import requests

class PlaceSearch:
    def __init__(self, api_key):
        self.api_key = api_key

    def search_nearby_places(self, location, place_type):
        url = f'https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={location}&radius=1000&type={place_type}&key={self.api_key}'
        response = requests.get(url)
        data = response.json()

        if data['status'] == 'OK':
            with open('info.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Latitude', 'Longitude'])
                for result in data['results']:
                    latitude = result['geometry']['location']['lat']
                    longitude = result['geometry']['location']['lng']
                    writer.writerow([latitude, longitude])

            print("Place details (latitude and longitude) saved to 'info.csv'")
        else:
            print('Error occurred:', data['status'])
