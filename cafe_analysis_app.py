import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import googlemaps
from tensorflow.keras.models import load_model

# Load sentiment analysis model
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
sentiment_model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

# Load image classification model
image_model = load_model('restaurant_rating_model.h5')  # Ensure the model is in the same directory

# Initialize Google Maps API
gmaps = googlemaps.Client(key='AIzaSyB1jm6Nl44nMnIZqX9qzx0up_lUro5QyfA')  # Replace with your Google Maps API key

import os
import requests

def download_model(model_url, model_path):
    if not os.path.exists(model_path):
        print(f"Downloading model from {model_url}...")
        response = requests.get(model_url, stream=True)
        with open(model_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Model downloaded to {model_path}.")
    else:
        print("Model already exists locally.")

# URLs for models
model_h5_url = "https://drive.google.com/file/d/1hGJXbPd9bg5-GAxjJbOWiYSZeYc1jRHP/view?usp=sharing"

# Paths to save the models locally
model_h5_path = "restaurant_rating_model.h5"

# Download models if not present
download_model(model_h5_url, model_h5_path)

# Streamlit app setup
st.title("Cafe & Tourist Spot Recommender")
st.write("Enter a location to find nearby cafes and attractions that match your preferences.")

# Take location input from user
place_name = st.text_input("Enter the location (city or area):")

if place_name:
    try:
        geocode_result = gmaps.geocode(place_name)

        if not geocode_result:
            st.error(f"Location '{place_name}' not found. Please try again.")
        else:
            st.write(f"**Geocoded Result:** {geocode_result[0]['formatted_address']}")
            location = geocode_result[0]['geometry']['location']
            current_location = (location['lat'], location['lng'])
            st.write(f"**Coordinates for {place_name}:** {current_location}")

            # Find nearby cafes
            places_result = gmaps.places_nearby(location=current_location, radius=2000, type='cafe')
            cafes = places_result['results'][:10]
            cafe_info = []

            for cafe in cafes:
                place_id = cafe['place_id']
                details = gmaps.place(place_id=place_id, fields=['name', 'rating', 'user_ratings_total', 'review'])

                if 'reviews' in details['result']:
                    reviews = details['result']['reviews']
                    review_texts = [review['text'] for review in reviews]

                    # Get photo URL if available
                    if 'photos' in cafe:
                        photo_reference = cafe['photos'][0]['photo_reference']
                        photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_reference}&key=AIzaSyB1jm6Nl44nMnIZqX9qzx0up_lUro5QyfA"
                    else:
                        photo_url = None

                    cafe_info.append({
                        'name': details['result']['name'],
                        'rating': details['result']['rating'],
                        'user_ratings_total': details['result']['user_ratings_total'],
                        'reviews': review_texts,
                        'photo_url': photo_url
                    })

            df_cafes = pd.DataFrame(cafe_info)
            st.write("### Cafe Ratings Overview")
            st.write(df_cafes[['name', 'rating', 'user_ratings_total']])


            # Sentiment analysis on reviews
            def sentiment_score(review):
                tokens = tokenizer.encode(review[:512], return_tensors='pt')
                result = sentiment_model(tokens)
                return int(torch.argmax(result.logits)) + 1


            for index, row in df_cafes.iterrows():
                df_cafes.at[index, 'average_sentiment'] = sum(
                    [sentiment_score(review) for review in row['reviews']]) / len(row['reviews'])


            # Predict ratings from images
            def predict_rating_from_image(image_url):
                response = requests.get(image_url)
                img = Image.open(BytesIO(response.content))

                if img.mode == 'RGBA':
                    img = img.convert('RGB')

                img = img.resize((224, 224))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                predictions = image_model.predict(img_array)
                return np.argmax(predictions)


            for index, row in df_cafes.iterrows():
                if row['photo_url']:
                    predicted_rating = predict_rating_from_image(row['photo_url'])
                    df_cafes.at[index, 'predicted_rating'] = predicted_rating + 1  # Adjust for 1-5 scale
                else:
                    df_cafes.at[index, 'predicted_rating'] = None

            df_cafes['final_score'] = (df_cafes['average_sentiment'] + df_cafes['predicted_rating']) / 2
            df_cafes_sorted = df_cafes.sort_values(by='final_score', ascending=False)

            st.write("### Cafes sorted by combined analysis:")
            st.write(df_cafes_sorted[['name', 'rating', 'user_ratings_total', 'average_sentiment', 'predicted_rating',
                                      'final_score']])

            for index, row in df_cafes_sorted.iterrows():
                st.write(f"**Cafe:** {row['name']}")
                st.write(
                    f"Rating: {row['rating']} | User Ratings: {row['user_ratings_total']} | Average Sentiment: {row['average_sentiment']} | Predicted Rating: {row['predicted_rating']} | Final Score: {row['final_score']}")
                if row['photo_url']:
                    response = requests.get(row['photo_url'])
                    img = Image.open(BytesIO(response.content))
                    st.image(img, caption=row['name'])
                else:
                    st.write("No image available for this cafe.")

            # Input available time and mood from user
            time_available = st.number_input("Enter your available time in hours:", min_value=1, max_value=12, step=1)
            mood = st.selectbox("Select your preferred spot type:",
                                ["Tourist Attraction", "Game Zone", "Museum", "Park"])

            # Fetch nearby places based on mood and radius
            type_mapping = {
                "Tourist Attraction": 'tourist_attraction',
                "Game Zone": 'amusement_park',
                "Museum": 'museum',
                "Park": 'park'
            }
            selected_type = type_mapping[mood]


            def fetch_nearby_places(place_type, radius=2000):
                return gmaps.places_nearby(location=current_location, radius=radius, type=place_type)['results']


            selected_spots = fetch_nearby_places(selected_type)
            spots_info = []

            for spot in selected_spots:
                spot_name = spot['name']
                spot_location = (spot['geometry']['location']['lat'], spot['geometry']['location']['lng'])

                dist_matrix = gmaps.distance_matrix(origins=current_location, destinations=spot_location,
                                                    mode="driving")
                distance_km = dist_matrix['rows'][0]['elements'][0]['distance']['value'] / 1000  # Convert meters to km
                travel_time = distance_km / 40  # Approximate travel time at 40 km/h

                if travel_time <= time_available:
                    # Get reviews and calculate sentiment score for each spot
                    place_id = spot['place_id']
                    details = gmaps.place(place_id=place_id, fields=['review', 'photo', 'rating'])
                    if 'reviews' in details['result']:
                        reviews = [review['text'] for review in details['result']['reviews']]
                        average_sentiment = sum([sentiment_score(review) for review in reviews]) / len(reviews)
                    else:
                        average_sentiment = None

                    if 'photos' in spot:
                        photo_reference = spot['photos'][0]['photo_reference']
                        photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_reference}&key=AIzaSyB1jm6Nl44nMnIZqX9qzx0up_lUro5QyfA"
                    else:
                        photo_url = None

                    spots_info.append({
                        'name': spot_name,
                        'distance_km': distance_km,
                        'estimated_travel_time': travel_time,
                        'average_sentiment': average_sentiment,
                        'rating': details['result'].get('rating'),
                        'photo_url': photo_url
                    })

            # Display spots filtered by mood and time
            if spots_info:
                st.write(f"### Recommended {mood} Spots within your available time:")
                spots_df = pd.DataFrame(spots_info).sort_values(by=['average_sentiment', 'estimated_travel_time'],
                                                                ascending=[False, True])

                for _, row in spots_df.iterrows():
                    st.write(f"**Spot:** {row['name']}")
                    st.write(
                        f"Distance: {row['distance_km']} km")
                    st.write(f"Average Sentiment: {row['average_sentiment']} | Google Rating: {row['rating']}")
                    if row['photo_url']:
                        response = requests.get(row['photo_url'])
                        img = Image.open(BytesIO(response.content))
                        st.image(img, caption=row['name'])
                    else:
                        st.write("No image available for this spot.")
            else:
                st.write("No recommended spots within the given time and preference.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
