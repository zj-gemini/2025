from typing import List, Dict, Set
import heapq


class Twitter:
    def __init__(self):
        # userId -> set of followees
        self.follows: Dict[int, Set[int]] = {}
        # userId -> list of (timestamp, tweetId)
        self.tweets: Dict[int, List[tuple]] = {}
        self.time = 0  # global timestamp

    def postTweet(self, userId: int, tweetId: int) -> None:
        if userId not in self.tweets:
            self.tweets[userId] = []
        self.tweets[userId].append((self.time, tweetId))
        self.time += 1

    def getNewsFeed(self, userId: int) -> List[int]:
        # Get self and followees
        followees = self.follows.get(userId, set()).copy()
        followees.add(userId)
        # Collect all tweets from self and followees
        tweets = []
        for uid in followees:
            if uid in self.tweets:
                for t in self.tweets[uid][-10:]:  # Only need last 10 per user
                    tweets.append(t)
        # Get 10 most recent
        tweets.sort(reverse=True)  # sort by timestamp descending
        return [tweetId for _, tweetId in tweets[:10]]

    def follow(self, followerId: int, followeeId: int) -> None:
        if followerId == followeeId:
            return
        if followerId not in self.follows:
            self.follows[followerId] = set()
        self.follows[followerId].add(followeeId)

    def unfollow(self, followerId: int, followeeId: int) -> None:
        if followerId in self.follows:
            self.follows[followerId].discard(followeeId)


# Test for the example case
def test():
    twitter = Twitter()
    twitter.postTweet(1, 5)
    print(twitter.getNewsFeed(1))  # [5]
    twitter.follow(1, 2)
    twitter.postTweet(2, 6)
    print(twitter.getNewsFeed(1))  # [6, 5]
    twitter.unfollow(1, 2)
    print(twitter.getNewsFeed(1))  # [5]


if __name__ == "__main__":
    test()
