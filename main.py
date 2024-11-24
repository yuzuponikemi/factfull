import tweepy

# APIキーを設定
CONSUMER_KEY = '****'
CONSUMER_SECRET = '****'
ACCESS_TOKEN = '****'
ACCESS_SECRET = '****'

# 認証を行う
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
api = tweepy.API(auth)

# ツイートを投稿
api.update_status('ここにツイート内容を入力')
