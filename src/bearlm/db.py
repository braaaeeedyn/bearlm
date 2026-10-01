"""MongoDB Atlas connection helpers (openai backend)."""

from pymongo import MongoClient

from . import config


def get_client() -> MongoClient:
    return MongoClient(config.MONGODB_URI, serverSelectionTimeoutMS=5000)


def get_chunks_collection():
    return get_client()[config.DB_NAME][config.CHUNKS_COLL]


def get_query_logs_collection():
    return get_client()[config.DB_NAME][config.QUERY_LOGS_COLL]


def ping() -> bool:
    get_client().admin.command("ping")
    return True
