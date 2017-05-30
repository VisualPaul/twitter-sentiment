import tweepy
import api_keys
from Queue import Queue

queue = Queue(10)

class Listener(tweepy.StreamListener):
    def on_data(self, data):
        queue.put(data)
auth = tweepy.OAuthHandler(api_keys.TWITTER_API_KEY, api_keys.TWITTER_API_SECRET)
auth.set_access_token(api_keys.TWITTER_ACCESS_TOKEN, api_keys.TWITTER_TOKEN_SECRET)
stream = tweepy.Stream(auth, Listener())

stream.filter(track=['the', 'be', 'and', 'a'], languages=['en'], async=True)
