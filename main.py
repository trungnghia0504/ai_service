from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from recommender import Recommender
from bson.objectid import ObjectId
from enum import Enum
import logging
import time
from contextlib import asynccontextmanager
from recommendation_content import ContentRecommender
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

recommender_instance = Recommender()
content_recommender = ContentRecommender()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await recommender_instance.create_indexes()
    await content_recommender.create_indexes()
    logger.info("Indexes created on startup")
    yield

app = FastAPI(
    title="Recommendation API",
    description="API gợi ý bạn bè và nội dung cá nhân hóa bất đồng bộ",
    version="1.0.0",
    lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendationMethod(str, Enum):
    CF = "cf"
    CB = "cb"
    SOCIAL = "social"  # Thêm cho gợi ý nội dung
    HYBRID = "hybrid"

async def get_recommender():
    return recommender_instance

# Dependency cho gợi ý nội dung
async def get_content_recommender():
    return content_recommender

# Endpoint gợi ý nội dung (mới)
@app.get("/recommend-content/{user_id}")
async def get_content_recommendations(
    user_id: str,
    method: RecommendationMethod = Query(default=RecommendationMethod.HYBRID),
    top_n: int = Query(default=10, ge=1, le=100),
    recommender: ContentRecommender = Depends(get_content_recommender)
):
    start_time = time.time()
    logger.info(f"Received content request for user_id={user_id}, method={method}, top_n={top_n}")

    try:
        user_oid = ObjectId(user_id)
    except Exception:
        logger.error(f"Invalid user_id: {user_id}")
        raise HTTPException(status_code=400, detail="Invalid user_id format. Must be a valid ObjectId.")

    try:
        if method == RecommendationMethod.CF:
            recs = await recommender._collaborative_filtering(user_oid, top_n)
        elif method == RecommendationMethod.CB:
            recs = await recommender._content_based_filtering(user_oid, top_n)
        elif method == RecommendationMethod.SOCIAL:
            recs = await recommender._social_filtering(user_oid, top_n)
        else:
            recs = await recommender.hybrid_recommendations(user_oid, top_n)
    except TypeError as e:
        logger.error(f"TypeError: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

    processing_time = time.time() - start_time
    logger.info(f"Completed content request for user_id={user_id} in {processing_time:.2f} seconds")

    return {
        "user_id": user_id,
        "recommended_posts": recs,
        "method": method.value,
        "count": len(recs),
        "processing_time_seconds": round(processing_time, 2)
    }

@app.get("/recommend/{user_id}")
async def get_recommendations(
    user_id: str,
    method: RecommendationMethod = Query(default=RecommendationMethod.HYBRID),
    top_n: int = Query(default=10, ge=1, le=100),
    recommender: Recommender = Depends(get_recommender)
):
    start_time = time.time()
    logger.info(f"Received request for user_id={user_id}, method={method}, top_n={top_n}")

    try:
        user_oid = ObjectId(user_id)
    except Exception:
        logger.error(f"Invalid user_id: {user_id}")
        raise HTTPException(status_code=400, detail="Invalid user_id format. Must be a valid ObjectId.")

    try:
        if method == RecommendationMethod.CF:
            recs = await recommender._collaborative_filtering(user_oid, top_n)
        elif method == RecommendationMethod.CB:
            recs = await recommender._content_based_filtering(user_oid, 0.5, 50, top_n)
        else:
            recs = await recommender.hybrid_recommendations(user_oid, top_n)
    except TypeError as e:
        logger.error(f"TypeError: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

    processing_time = time.time() - start_time
    logger.info(f"Completed request for user_id={user_id} in {processing_time:.2f} seconds")

    return {
        "user_id": user_id,
        "recommendations": recs,
        "method": method.value,
        "count": len(recs),
        "processing_time_seconds": round(processing_time, 2)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)