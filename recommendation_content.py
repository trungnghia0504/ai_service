import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from surprise import SVD, Dataset, Reader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import logging
from bson.objectid import ObjectId
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentRecommender:
    def __init__(self):
        self.client = AsyncIOMotorClient(os.getenv("MONGODB_URI"))
        self.db = self.client[os.getenv("DATABASE_NAME")]
        self.users = self.db["users"]
        self.posts = self.db["posts"]
        self.interactions = self.db["interactions"]
        self.params_collection = self.db["recommendation_params"]

    async def get_params(self):
        """Fetch recommendation parameters from MongoDB"""
        params = await self.params_collection.find_one()
        if not params:
            # Default parameters if none exist
            params = {
                "user_top_n": 10,
                "interest_threshold": 0.5,
                "max_distance_km": 50,
                "user_cf_weight": 0.5,
                "user_cb_weight": 0.5,
                "commentWeight": 2,
                "likeWeight": 1,
                "viewWeight": 0.5,
                "post_top_n": 10,
                "similarity_threshold": 0.1,
                "post_cf_weight": 0.5,
                "post_cb_weight": 0.3,
                "post_social_weight": 0.2,
            }
            await self.params_collection.insert_one(params)
        return params

    async def create_indexes(self):
        """Create indexes to optimize queries"""
        await self.interactions.create_index([("fromUser", 1), ("postId", 1)])
        await self.posts.create_index([("tags", 1)])
        await self.posts.create_index([("likes", 1)])

    async def _collaborative_filtering(self, user_id):
        """Collaborative filtering based on post interactions"""
        params = await self.get_params()
        top_n = params["post_top_n"]
        comment_weight = params["commentWeight"]
        like_weight = params["likeWeight"]
        view_weight = params["viewWeight"]

        pipeline = [
            {"$match": {"postId": {"$exists": True}}},
            {"$group": {"_id": {"userId": "$fromUser", "postId": "$postId", "type": "$type"}, "count": {"$sum": 1}}},
            {"$project": {
                "userId": "$_id.userId", "postId": "$_id.postId", "type": "$_id.type",
                "weighted_count": {
                    "$switch": {
                        "branches": [
                            {"case": {"$eq": ["$type", "comment"]}, "then": {"$multiply": ["$count", comment_weight]}},
                            {"case": {"$eq": ["$type", "like"]}, "then": {"$multiply": ["$count", like_weight]}},
                            {"case": {"$eq": ["$type", "view"]}, "then": {"$multiply": ["$count", view_weight]}}
                        ],
                        "default": "$count"
                    }
                }
            }},
            {"$group": {"_id": {"userId": "$userId", "postId": "$postId"}, "count": {"$sum": "$weighted_count"}}},
            {"$sort": {"count": -1}},
        ]
        interactions = [doc async for doc in self.interactions.aggregate(pipeline)]
        if not interactions:
            logger.warning("No interactions found")
            return []

        data = [(str(i["_id"]["userId"]), str(i["_id"]["postId"]), i["count"]) for i in interactions]
        df = pd.DataFrame(data, columns=["userId", "postId", "count"])
        df["count"] = df["count"].clip(upper=10)  # Normalize rating scale
        reader = Reader(rating_scale=(1, 10))
        dataset = Dataset.load_from_df(df, reader)
        trainset = dataset.build_full_trainset()

        algo = SVD()
        algo.fit(trainset)

        all_posts = set(df["postId"].unique())
        user_interacted = set(str(i["postId"]) for i in await self.interactions.find({"fromUser": user_id}).to_list(1000) if "postId" in i)
        potential_posts = [pid for pid in all_posts if pid not in user_interacted]

        user_id_str = str(user_id)
        predictions = [algo.predict(user_id_str, pid) for pid in potential_posts]
        top_recs = sorted(predictions, key=lambda x: x.est, reverse=True)[:top_n]
        return [pred.iid for pred in top_recs]

    async def _content_based_filtering(self, user_id):
        """Content-based filtering based on user interests and post content"""
        params = await self.get_params()
        top_n = params["post_top_n"]
        similarity_threshold = params["similarity_threshold"]

        user = await self.users.find_one({"_id": user_id})
        if not user:
            logger.warning(f"No user found for user_id: {user_id}")
            return []

        posts = [doc async for doc in self.posts.find()]
        if not posts:
            logger.warning("No posts found")
            return []

        user_interests = " ".join(user["interests"])
        logger.info(f"User interests: {user_interests}")
        post_contents = [user_interests] + [
            " ".join((p.get("tags", []) + [p["content"]])) for p in posts
        ]
        logger.info(f"Post contents: {post_contents[1:]}")
        tfidf = TfidfVectorizer().fit_transform(post_contents)
        similarity = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
        logger.info(f"Similarity scores: {similarity.tolist()}")

        interactions = await self.interactions.find({"fromUser": user_id}).to_list(1000)
        user_interacted = set(str(i["postId"]) for i in interactions if "postId" in i)
        logger.info(f"User interacted posts: {user_interacted}")
        recs = [
            (str(p["_id"]), similarity[i])
            for i, p in enumerate(posts)
            if str(p["_id"]) not in user_interacted and similarity[i] > similarity_threshold
        ]
        recs.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"Recommendations before slicing: {recs}")

        return [r[0] for r in recs[:top_n]]

    async def _social_filtering(self, user_id):
        """Social filtering based on friends' likes"""
        params = await self.get_params()
        top_n = params["post_top_n"]

        user = await self.users.find_one({"_id": user_id})
        if not user:
            logger.warning(f"No user found for user_id: {user_id}")
            return []

        friends = [ObjectId(f) for f in user["friends"]]
        posts = [doc async for doc in self.posts.find({"likes": {"$in": friends}})]
        if not posts:
            logger.warning("No posts liked by friends found")
            return []

        user_interacted = set(str(i["postId"]) for i in await self.interactions.find({"fromUser": user_id}).to_list(None) if "postId" in i)
        
        recs = [
            (str(p["_id"]), len(set(str(l) for l in p["likes"]) & set(str(f) for f in friends)))
            for p in posts
            if str(p["_id"]) not in user_interacted
        ]
        recs.sort(key=lambda x: (x[1], x[0]), reverse=True)
        return [r[0] for r in recs[:top_n]]

    async def hybrid_recommendations(self, user_id):
        """Hybrid recommendation combining CF, CB, and social filtering"""
        params = await self.get_params()
        top_n = params["post_top_n"]
        cf_weight = params["post_cf_weight"]
        cb_weight = params["post_cb_weight"]
        social_weight = params["post_social_weight"]

        cf_task = self._collaborative_filtering(user_id)
        cb_task = self._content_based_filtering(user_id)
        social_task = self._social_filtering(user_id)
        cf_recs, cb_recs, social_recs = await asyncio.gather(cf_task, cb_task, social_task)

        combined = {}
        for i, rec in enumerate(cf_recs):
            combined[rec] = combined.get(rec, 0) + cf_weight * (top_n - i)
        for i, rec in enumerate(cb_recs):
            combined[rec] = combined.get(rec, 0) + cb_weight * (top_n - i)
        for i, rec in enumerate(social_recs):
            combined[rec] = combined.get(rec, 0) + social_weight * (top_n - i)

        return sorted(combined.keys(), key=lambda x: combined[x], reverse=True)[:top_n]