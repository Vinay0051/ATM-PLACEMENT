import requests
import urllib

class PlacesAPI:
    def __init__(self, api_key):
        self.api_key = api_key

    def get_photo_url(self, lat, lng):
        url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={lat},{lng}&radius=100&key={self.api_key}"
        response = requests.get(url)
        data = response.json()

        photo_reference = data['results'][0]['photos'][0]['photo_reference']
        photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_reference}&key={self.api_key}"

        return photo_url

    def save_photo(self, photo_url, file_path):
        urllib.request.urlretrieve(photo_url, file_path)
        print("Image saved successfully!")
