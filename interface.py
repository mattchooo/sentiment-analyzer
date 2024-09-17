import pickle
from transformers import RobertaTokenizer
import torch
import tkinter as tk
from tkinter import messagebox
from googleapiclient.discovery import build

API_KEY = "AIzaSyDglcYb_IfLi1P9pTS6DmMyWA0tuhWB0_c"
label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}

# Load RoBERTa
def load_model():
    with open("roberta_model.pkl", 'rb') as f:
        model = pickle.load(f)
    return model

# Tokenize the input for RoBERTa
def preprocess_input(text, tokenizer):
    return tokenizer(text, return_tensors='pt')

# Predict a value for the sentiment of the input
def predict_sentiment(model, input):
    output = model(**input)
    predicted_class = torch.argmax(output.logits).item()
    return predicted_class

# Map predicted sentiment value (0, 1, 2) to a string (negative, neutral, positive)
def postprocess_output(predicted_class):
    return label_mapping[predicted_class]

# Get trending YouTube videos
def get_trending_videos():
    request = youtube.videos().list(
        part="snippet",
        chart="mostPopular",
        maxResults=10,
        regionCode="US"
    )
    response = request.execute()
    trending_videos = []
    for item in response['items']:
       video_data = item['snippet']
       video_title = video_data['title']
       video_id = item['id']
       trending_videos.append({'title': video_title, 'video_id': video_id})
    return trending_videos

# Fetch the comments of a provided video
def get_video_comments(video_id):
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id
    )
    response = request.execute()
    comments = []
    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
        comments.append(comment)
    return comments

# Given the comments of a YouTube video, tally up the predicted sentiment of each and return in an array
def analyze_sentiment(comments):
    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    for comment in comments:
        input = preprocess_input(comment, roberta_tokenizer)
        predicted_class = predict_sentiment(model, input)
        predicted_sentiment = postprocess_output(predicted_class)
        sentiment_counts[predicted_sentiment] += 1
    total_comments = sum(sentiment_counts.values())
    percentages = {key: (value / total_comments) * 100 for key, value in sentiment_counts.items()}
    overall_sentiment = max(percentages, key=percentages.get)
    return {
        'overall': overall_sentiment,
        'percentages': percentages
    }

# Create the buttons on the GUI that can be clicked (trending YouTube videos)
def create_video_buttons(videos):
    for video in videos:
        video_title = video['title']
        video_id = video['video_id']
        button = tk.Button(
            window,
            text=video_title,
            command=lambda video_id=video_id: analyze_video(video_id)
        )
        button.pack(fill=tk.X)

# Output message box with returned array of tallied sentiments
def create_sentiment_message(sentiment_data):
    message = f"Overall sentiment in comments: {sentiment_data['overall']}\n\n"
    percentages = sentiment_data['percentages']
    for sentiment, percentage in percentages.items():
        message += f"{sentiment.capitalize()}: {percentage:.2f}%\n"
    return message

# Analyze the comments' sentiments of the video clicked
def analyze_video(video_id):
    comments = get_video_comments(video_id)
    if not comments:
        messagebox.showerror("Error", "Failed to retrieve comments for the video.")
        return
    sentiment_data = analyze_sentiment(comments)
    message = create_sentiment_message(sentiment_data)
    messagebox.showinfo("Sentiment Analysis", message)


# Main
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = load_model()

window = tk.Tk()
window.title("YouTube Trending Videos: Comments Sentiment Analysis")
window.geometry("500x300")

youtube = build("youtube", "v3", developerKey=API_KEY)
videos = get_trending_videos()
if not videos:
  messagebox.showerror("Error", "Failed to retrieve trending videos.")
  exit()

create_video_buttons(videos)

window.mainloop()