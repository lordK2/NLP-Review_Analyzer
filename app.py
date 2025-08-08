# app.py - The backend for my sentiment analyzer project

# First, gotta import all the tools we need
from flask import Flask, request, jsonify, render_template # Flask is for the web server part
import requests # This lets us get stuff from web pages/APIs
import joblib # For loading my saved model

# Initialize the Flask application. This is like the main engine.
app = Flask(__name__)

# --- IMPORTANT: PASTE YOUR TMDB API KEY HERE ---
# This is the secret key I got from the TMDb website.
TMDB_API_KEY = "391f9c14603803d3fe13085a81d3a962" 

# --- Load the AI model and the vectorizer ---
# These are the files I made with the Jupyter notebook.
# I need to load them when the server starts so they're ready to use.
print("Server is starting up... trying to load my saved model files.")
try:
    # Load the vectorizer first. This turns words into numbers.
    print("Loading vectorizer.pkl...")
    my_vectorizer = joblib.load('vectorizer.pkl')
    print("Vectorizer seems to be loaded okay.")

    # Now load the actual sentiment model.
    print("Loading sentiment_model.pkl...")
    my_model = joblib.load('sentiment_model.pkl')
    print("Sentiment model seems to be loaded okay.")

    print("\n--- Looks like the model and vectorizer are ready! ---")

except Exception as e:
    # If something goes wrong here, the app can't work.
    print("!!! BIG ERROR: Could not load the model files. !!!")
    print(f"The error was: {e}")
    my_model = None
    my_vectorizer = None

# --- Here are the functions for talking to the TMDb API ---

# This function takes a movie name and finds its ID number.
def find_movie_id_from_tmdb(title):
    print(f"Searching for movie ID for: '{title}'")
    # This is the URL for the TMDb search API.
    # I have to put my API key and the movie title in it.
    api_search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
    
    # Use requests to get the data from that URL.
    response_from_api = requests.get(api_search_url)
    # Convert the response to JSON so it's easy to work with.
    data_from_api = response_from_api.json()
    
    # The results are inside a list called 'results'.
    # I'll check if the list is not empty.
    if data_from_api.get('results'):
        # Just grab the first result, it's usually the right one.
        first_result = data_from_api['results'][0]
        movie_id = first_result['id']
        print(f"Found movie ID: {movie_id}")
        return movie_id
    else:
        # If the 'results' list is empty, the movie wasn't found.
        print(f"Could not find any movie with the title '{title}'.")
        return None

# This function gets all the reviews for a movie using its ID.
def get_reviews_for_movie(movie_id):
    print(f"Getting reviews for movie ID: {movie_id}")
    # This is the URL for the reviews API endpoint.
    api_reviews_url = f"https://api.themoviedb.org/3/movie/{movie_id}/reviews?api_key={TMDB_API_KEY}"
    
    # Get the data from the reviews URL.
    response_from_api = requests.get(api_reviews_url)
    data_from_api = response_from_api.json()
    
    # The reviews are in a list called 'results'.
    # I need to loop through them and just get the 'content' part of each review.
    list_of_reviews = []
    if data_from_api.get('results'):
        all_review_data = data_from_api['results']
        for review_item in all_review_data:
            review_text = review_item['content']
            list_of_reviews.append(review_text)
    
    print(f"Found {len(list_of_reviews)} reviews.")
    return list_of_reviews

# --- These are the routes for the website ---

# This is the main page of the app (like the homepage).
@app.route('/')
def show_main_page():
    # This just shows the index.html file to the user.
    # Flask is smart and looks for it in the 'templates' folder.
    print("A user visited the home page.")
    return render_template('index.html')

# This is the "API endpoint" that the JavaScript on the front-end will call.
# It does all the heavy lifting.
@app.route('/analyze', methods=['POST'])
def handle_analysis_request():
    print("\nReceived a new request to analyze a movie!")
    
    # First, check if the model even loaded correctly when the app started.
    if not my_model or not my_vectorizer:
        print("Error because model isn't loaded.")
        return jsonify({'error': 'Model not loaded. Cannot perform analysis.'}), 500
    
    # Also check if I remembered to paste my API key.
    if TMDB_API_KEY == "PASTE_YOUR_API_KEY_HERE":
        print("Error because API key is missing.")
        return jsonify({'error': 'TMDb API key is not set in app.py.'}), 500

    # Get the data that the JavaScript sent to us. It should be JSON.
    request_data = request.get_json()
    # Get the movie title from that data.
    movie_title_from_user = request_data.get('movie_title')

    # Make sure the user actually typed something.
    if not movie_title_from_user:
        print("Error because user didn't send a movie title.")
        return jsonify({'error': 'Movie title is required.'}), 400

    print(f"User wants to analyze the movie: '{movie_title_from_user}'")
    
    # --- Step 1: Find the movie's ID ---
    movie_id = find_movie_id_from_tmdb(movie_title_from_user)
    if not movie_id:
        # If we couldn't find the movie, send an error back to the user.
        error_message = f"Could not find a movie with the title '{movie_title_from_user}'."
        return jsonify({'error': error_message}), 404
    
    # --- Step 2: Get the reviews for that movie ---
    reviews_list = get_reviews_for_movie(movie_id)
    if not reviews_list:
        # If the movie exists but has no reviews, tell the user.
        return jsonify({'error': 'This movie does not have any reviews on TMDb.'}), 404

    print(f"Got {len(reviews_list)} reviews to analyze.")

    # --- Step 3: Use the AI model to predict sentiment ---
    # First, convert the list of review sentences into numbers using the vectorizer.
    reviews_vectorized = my_vectorizer.transform(reviews_list)
    # Now, use the model to predict if each review is positive (1) or negative (0).
    sentiment_predictions = my_model.predict(reviews_vectorized)
    
    # --- Step 4: Count the results and prepare them to send back ---
    print("Counting up the positive and negative results...")
    final_results_list = []
    positive_review_count = 0
    negative_review_count = 0

    # Loop through the original reviews and the predictions at the same time.
    for i in range(len(reviews_list)):
        review_text = reviews_list[i]
        prediction = sentiment_predictions[i]
        
        # Convert the prediction (0 or 1) into a word ('Positive' or 'Negative').
        if prediction == 1:
            sentiment_label = 'Positive'
            positive_review_count = positive_review_count + 1
        else:
            sentiment_label = 'Negative'
            negative_review_count = negative_review_count + 1
            
        # Create a dictionary for this single review and its result.
        result_item = {
            'text': review_text, 
            'sentiment': sentiment_label
        }
        # Add it to our list of final results.
        final_results_list.append(result_item)

    # --- Step 5: Send the final package of data back to the frontend ---
    # This will be a JSON object with all the info the JavaScript needs to build the results page.
    final_response_data = {
        'positive_count': positive_review_count,
        'negative_count': negative_review_count,
        'total_reviews': len(final_results_list),
        'reviews': final_results_list 
    }
    
    print("Analysis complete! Sending results back to the user's browser.")
    return jsonify(final_response_data)

# --- This part makes the server run when I type "python app.py" ---
if __name__ == '__main__':
    # debug=True is super helpful, it makes the server auto-restart when I save changes.
    app.run(debug=True)
