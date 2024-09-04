import csv
import requests

class PlaceSearch:
    def __init__(self, api_key):
        self.api_key = api_key

    def get_photo_url(self, photo_reference):
        photo_url = ''
        if photo_reference:
            photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_reference}&key={self.api_key}"
        return photo_url

    def search_nearby_places(self, location, place_type):
        url = f'https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={location}&radius=1000&type={place_type}&key={self.api_key}'
        response = requests.get(url)
        data = response.json()

        if data['status'] == 'OK':
            counter = 0  # Counter variable to keep track of the number of entries

            with open('info.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Name', 'Address', 'Latitude', 'Longitude', 'Rating', 'Photo Reference', 'Photo URL'])

                for result in data['results']:
                    if counter >= 6:
                        break  # Break out of the loop if the maximum number of entries (6) is reached

                    name = result['name']
                    address = result['vicinity']
                    latitude = result['geometry']['location']['lat']
                    longitude = result['geometry']['location']['lng']
                    rating = result.get('rating', 'N/A')
                    photo_reference = result.get('photos', [{}])[0].get('photo_reference', '')
                    photo_url = self.get_photo_url(photo_reference)

                    writer.writerow([name, address, latitude, longitude, rating, photo_reference, photo_url])
                    counter += 1  # Increment the counter after writing an entry

            print("Place details saved to 'info.csv'")
        else:
            print('Error occurred:', data['status'])

# Create an instance of PlaceSearch and provide your API key
#place_search = PlaceSearch("YOUR_API_KEY")

# Call the search_nearby_places method with location and place_type
#place_search.search_nearby_places("latitude,longitude", "restaurant")
