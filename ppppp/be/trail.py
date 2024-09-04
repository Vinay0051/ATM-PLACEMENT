import csv
import requests

# Set up the API key
API_KEY ='AIzaSyCsJp9BP-TpdqL2xrIOwuK-oRdX_rZn2gE'

# Specify the location coordinates (latitude and longitude)
location = '12.84698,80.22672'

# Set the type of place you're interested in (e.g., 'ATM')
place_type = 'atm'

# Function to retrieve photo URL from photo reference
def get_photo_url(photo_reference):
    photo_url = ''
    if photo_reference:
        photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_reference}&key={API_KEY}"
    return photo_url

# Send a request to the Places API
url = f'https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={location}&radius=1000&type={place_type}&key={API_KEY}'
response = requests.get(url)
data = response.json()

# Process the response
if data['status'] == 'OK':
    with open('info.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Name', 'Address', 'Latitude', 'Longitude', 'Rating', 'Photo Reference', 'Photo URL'])
        for result in data['results']:
            name = result['name']
            address = result['vicinity']
            latitude = result['geometry']['location']['lat']
            longitude = result['geometry']['location']['lng']
            rating = result.get('rating', 'N/A')
            photo_reference = result.get('photos', [{}])[0].get('photo_reference', '')
            photo_url = get_photo_url(photo_reference)

            writer.writerow([name, address, latitude, longitude, rating, photo_reference, photo_url])

    print("Place details saved to 'info.csv'")
else:
    print('Error occurred:', data['status'])
