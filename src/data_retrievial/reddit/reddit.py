import praw
import pandas as pd
import load_env
import os

reddit = praw.Reddit(
    client_id="sYWC73tdOG1rl-55VQsIvw",
    client_secret="LZFYc2VRLSht2YSpD-t7JXgzI9Heuw",
    user_agent="prova by /u/helig0"
)

def searchByTopic(subreddit, query, query_terms):
  """search topic in Subreddit"""
  posts_pol = reddit.subreddit(subreddit).top(limit=20)
  
  query = " ".join(query_terms)
  searchs_pol = posts_pol.search(query, sort='relevance', time_filter='all', limit=500)

  return list(searchs_pol)

query = 'Music'
#query_terms = ['AI','ai','artificial intelligence','deep learning']

search_results = searchByTopic('CharacterAI', query, query_terms)

posts = []
for submission in search_results:
    posts.append({
        'title': submission.title,
        'body':submission.selftext,
        'score': submission.score
        # 'url': submission.url
    })
df = pd.DataFrame(posts)
print(df)

if __name__ == "__main__":
    load_env.load_env()
    
    reddit = praw.Reddit(client_id = os.environ['REDDIT_CLIENT_ID'],
                            client_secret = os.environ['REDDIT_CLIENT_SECRET'],
                            user_agent = os.environ['REDDIT_USER_AGENT'])