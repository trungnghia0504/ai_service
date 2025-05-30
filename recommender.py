import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from surprise import SVD, Dataset, Reader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from bson.objectid import ObjectId
import logging
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Recommender:
    def __init__(self):
        self.client = AsyncIOMotorClient(os.getenv("MONGODB_URI"))
        self.db = self.client[os.getenv("DATABASE_NAME")]
        self.users = self.db.users
        self.interactions = self.db.interactions
        self.params_collection = self.db.recommendation_params

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
        await self.users.create_index([("friends", 1)])
        await self.users.create_index([("addfriends", 1)])
        await self.users.create_index([("interests", 1)])
        await self.users.create_index([("location", "2dsphere")])

    def to_str_list(self, items):
        """Convert list to string list"""
        return [str(item) for item in items]

    async def _collaborative_filtering(self, user_id):
        params = await self.get_params()
        top_n = params["user_top_n"]
        comment_weight = params["commentWeight"]
        like_weight = params["likeWeight"]
        view_weight = params["viewWeight"]

        user = await self.users.find_one({"_id": user_id})
        if not user:
            logger.warning(f"No user found for user_id: {user_id}")
            return []

        # Pipeline to fetch interactions with weighted counts
        pipeline = [
            {"$match": {"toUser": {"$ne": user_id}}},
            {
                "$group": {
                    "_id": {"fromUser": "$fromUser", "toUser": "$toUser", "type": "$type"},
                    "count": {"$sum": 1}
                }
            },
            {
                "$project": {
                    "fromUser": "$_id.fromUser",
                    "toUser": "$_id.toUser",
                    "type": "$_id.type",
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
                }
            },
            {
                "$group": {
                    "_id": {"fromUser": "$fromUser", "toUser": "$toUser"},
                    "count": {"$sum": "$weighted_count"}
                }
            },
            {"$sort": {"count": -1}},
            {"$limit": 1000},
            {"$project": {"fromUser": "$_id.fromUser", "toUser": "$_id.toUser", "count": "$count", "_id": 0}}
        ]
        interactions = [doc async for doc in self.interactions.aggregate(pipeline)]
        logger.info(f"Number of interactions fetched: {len(interactions)}")
        if not interactions:
            logger.warning("No interactions found for collaborative filtering")
            return []

        # Prepare data for Surprise
        data = [(str(i["fromUser"]), str(i["toUser"]), i["count"]) for i in interactions]
        df = pd.DataFrame(data, columns=["fromUser", "toUser", "count"])
        if str(user_id) not in df["fromUser"].unique():
            logger.warning(f"User {user_id} has no interactions as fromUser, CF cannot predict")
            return []

        reader = Reader(rating_scale=(1, 10))
        dataset = Dataset.load_from_df(df, reader)
        trainset = dataset.build_full_trainset()

        # Train SVD model
        algo = SVD()
        algo.fit(trainset)

        # Get all user IDs and filter potential users
        all_users = set(df["toUser"].unique()) | set(df["fromUser"].unique())
        friends = self.to_str_list(user.get("friends", []) + user.get("addfriends", []))
        user_id_str = str(user_id)
        potential_users = [uid for uid in all_users if uid not in friends and uid != user_id_str]
        logger.info(f"Potential users: {len(potential_users)} users - {potential_users}")
        if not potential_users:
            logger.warning("No potential users found after filtering")
            return []

        # Predict and get top-N recommendations
        predictions = [algo.predict(user_id_str, uid) for uid in potential_users]
        top_recs = sorted(predictions, key=lambda x: x.est, reverse=True)[:top_n]
        logger.info(f"Top recommendations before filtering: {[pred.iid for pred in top_recs]}")

        # Ensure no self-recommendation
        recommendations = [pred.iid for pred in top_recs if pred.iid != user_id_str]
        logger.info(f"Final recommendations: {recommendations}")
        return recommendations

    async def _content_based_filtering(self, user_id):
        params = await self.get_params()
        interest_threshold = params["interest_threshold"]
        max_distance_km = params["max_distance_km"]
        top_n = params["user_top_n"]

        user = await self.users.find_one({"_id": user_id})
        if not user:
            return []
        
        friends_and_addfriends = [ObjectId(str(fid)) for fid in user.get("friends", []) + user.get("addfriends", [])]
        pipeline = [
            {
                "$geoNear": {
                    "near": user["location"],
                    "distanceField": "dist.calculated",
                    "maxDistance": max_distance_km * 1000,
                    "spherical": True,
                    "query": {"_id": {"$nin": friends_and_addfriends + [ObjectId(user_id)]}}
                }
            },
            {"$limit": 100}
        ]
        nearby_users = [doc async for doc in self.users.aggregate(pipeline)]
        if not nearby_users:
            return []

        user_interests = " ".join(user["interests"])
        interests = [user_interests] + [" ".join(u["interests"]) for u in nearby_users]
        tfidf = TfidfVectorizer().fit_transform(interests)
        similarity = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

        user_id_str = str(user_id)
        recs = [
            (
                str(u["_id"]),
                0.5 * similarity[i] + 0.5 * (1 / (1 + u["dist"]["calculated"] / 1000))
            )
            for i, u in enumerate(nearby_users)
            if str(u["_id"]) != user_id_str and (similarity[i] >= interest_threshold or u["dist"]["calculated"] <= max_distance_km * 1000)
        ]
        recs.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in recs[:top_n]]

    async def hybrid_recommendations(self, user_id):
        params = await self.get_params()
        top_n = params["user_top_n"]
        user_cf_weight = params["user_cf_weight"]
        user_cb_weight = params["user_cb_weight"]

        cf_task = self._collaborative_filtering(user_id)
        cb_task = self._content_based_filtering(user_id)
        cf_recs, cb_recs = await asyncio.gather(cf_task, cb_task)

        combined = {}
        user_id_str = str(user_id)
        for i, rec in enumerate(cf_recs):
            if rec != user_id_str:
                combined[rec] = combined.get(rec, 0) + user_cf_weight * (top_n - i)
        for i, rec in enumerate(cb_recs):
            if rec != user_id_str:
                combined[rec] = combined.get(rec, 0) + user_cb_weight * (top_n - i)

        recommendations = sorted(combined.keys(), key=lambda x: combined[x], reverse=True)[:top_n]
        logger.info(f"Hybrid returned {len(recommendations)} recommendations: {recommendations}")
        return recommendations