import certifi
import ssl
import geopy
from flask import Flask, render_template, request
from geopy.geocoders import Nominatim
from dijistra1 import ClusteringGraph
from trail1 import PlaceSearch
from photo_retrival import PlacesAPI
import json
import dijistra1

api_key = "AIzaSyCsJp9BP-TpdqL2xrIOwuK-oRdX_rZn2gE"
ctx = ssl.create_default_context(cafile=certifi.where())
geopy.geocoders.options.default_ssl_context = ctx

geolocator = Nominatim(user_agent="geoapiExercises")
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('test.html')


@app.route('/validate', methods=['POST', 'GET'])
def validate():
    if request.method == 'POST':
        name = request.form.get('name')
        passw = request.form.get('pass')
        print(name, passw)
        return render_template('dd.html')


@app.route('/receive', methods=["POST", "GET"])
def receive():
    global json_data
    if request.method == "POST":
        a = request.form.get('query')
        print(a)
        location = geolocator.geocode(a)
        latitude = location.latitude
        longitude = location.longitude
        latlon = [latitude, longitude]
        print(latlon)
        d = {}
        d['coordinates'] = latlon
        with open('summa.json', 'w') as f:
            json.dump(d, f)
        with open('summa.json') as json_file:
            data = json.load(json_file)
        json_data = json.dumps(data)

        trail1_location = str(latlon[0]) + "," + str(latlon[1])
        place_type = "atm"
        placesearch = PlaceSearch(api_key)
        placesearch.search_nearby_places(trail1_location, place_type)
        data_path = r"E:\B.Tech sem 4\project\IV SEM PROJ\ppppp\info.csv"
        graph = ClusteringGraph(data_path)
        graph.load_data()
        graph.perform_clustering(n_clusters=3)
        graph.plot_clusters()
        graph.compute_distances()
        graph.set_source_vertex(latitude=latlon[0], longitude=latlon[1])  # Replace with actual latitude and longitude
        graph.compute_shortest_paths()
        graph.plot_dijkstra_graph()
        # graph.find_closest_centroid()
        closest_centroid = graph.find_closest_centroid()
        placesapi = PlacesAPI(api_key)
        photo_url = placesapi.get_photo_url(closest_centroid[0], closest_centroid[1])
        file_path = "photo.png"
        placesapi.save_photo(photo_url, file_path)
        graph.print_closest_centroid()
        graph.compute_centroids()
        graph.print_centroids()
    return render_template('ff.html', json_data=json_data)


@app.route('/efficient_location')
def efficient_location():
    with open('summa1.json') as json_file1:
        data = json.load(json_file1)
    json_data = json.dumps(data)
    return render_template('op.html', json_data=json_data)

@app.route('/graph')
def graph():
    return render_template('result2.html')


if __name__ == "__main__":
    app.run(debug=True)
