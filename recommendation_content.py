# recommendation_service/recommender.py
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

# Load biến môi trường từ file .env
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

    async def create_indexes(self):
        """Tạo chỉ mục tối ưu truy vấn"""
        await self.interactions.create_index([("fromUser", 1), ("postId", 1)])
        await self.posts.create_index([("tags", 1)])
        await self.posts.create_index([("likes", 1)])

    async def _collaborative_filtering(self, user_id, top_n=10):
        """Lọc cộng tác dựa trên tương tác với bài viết"""
        pipeline = [
            {"$match": {"postId": {"$exists": True}}},
            {"$group": {"_id": {"userId": "$fromUser", "postId": "$postId", "type": "$type"}, "count": {"$sum": 1}}},
            {"$project": {
                "userId": "$_id.userId", "postId": "$_id.postId", "type": "$_id.type",
                "weighted_count": {
                    "$switch": {
                        "branches": [
                            {"case": {"$eq": ["$type", "comment"]}, "then": {"$multiply": ["$count", 2]}},
                            {"case": {"$eq": ["$type", "like"]}, "then": "$count"},
                            {"case": {"$eq": ["$type", "view"]}, "then": {"$multiply": ["$count", 0.5]}}
                        ],
                        "default": "$count"
                    }
                }
            }},
            {"$group": {"_id": {"userId": "$userId", "postId": "$postId"}, "count": {"$sum": "$weighted_count"}}},
            {"$sort": {"count": -1}},
            # Loại bỏ $limit hoặc tăng nếu cần
        ]
        interactions = [doc async for doc in self.interactions.aggregate(pipeline)]
        if not interactions:
            logger.warning("No interactions found")
            return []

        data = [(str(i["_id"]["userId"]), str(i["_id"]["postId"]), i["count"]) for i in interactions]
        df = pd.DataFrame(data, columns=["userId", "postId", "count"])
        df["count"] = df["count"].clip(upper=10)  # Chuẩn hóa thang điểm
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

    async def _content_based_filtering(self, user_id, top_n=10):
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
            if str(p["_id"]) not in user_interacted and similarity[i] > 0.1  # Thêm ngưỡng
        ]
        recs.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"Recommendations before slicing: {recs}")

        # # Nếu không có gợi ý nào thỏa mãn ngưỡng, trả về bài viết phổ biến
        # if not recs:
        #     logger.info("No content-based recommendations found, falling back to popular posts")
        #     # Chuyển user_interacted thành danh sách ObjectId
        #     user_interacted_ids = [ObjectId(post_id) for post_id in user_interacted]
        #     popular_posts = [
        #         str(doc["_id"])
        #         async for doc in self.posts.find({"_id": {"$nin": user_interacted_ids}})
        #         .sort([("likes", -1), ("createdAt", -1)])
        #         .limit(top_n)
        #     ]
        #     logger.info(f"Popular posts: {popular_posts}")
        #     return popular_posts

        return [r[0] for r in recs[:top_n]]

    async def _social_filtering(self, user_id, top_n=10):
        """Lọc dựa trên lượt thích của bạn bè"""
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

    async def hybrid_recommendations(self, user_id, top_n=10):
        """Kết hợp hybrid"""
        cf_task = self._collaborative_filtering(user_id, top_n)
        cb_task = self._content_based_filtering(user_id, top_n)
        social_task = self._social_filtering(user_id, top_n)
        cf_recs, cb_recs, social_recs = await asyncio.gather(cf_task, cb_task, social_task)

        combined = {}
        for i, rec in enumerate(cf_recs):
            combined[rec] = combined.get(rec, 0) + 0.5 * (top_n - i)  # 50% SVD
        for i, rec in enumerate(cb_recs):
            combined[rec] = combined.get(rec, 0) + 0.3 * (top_n - i)  # 30% TF-IDF
        for i, rec in enumerate(social_recs):
            combined[rec] = combined.get(rec, 0) + 0.2 * (top_n - i)  # 20% Social

        return sorted(combined.keys(), key=lambda x: combined[x], reverse=True)[:top_n]