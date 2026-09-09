"""MongoDB Atlas connection helpers.

One place that knows how to hand you a `chunks` collection. Everything else
(ingestion, retrieval) calls these instead of building a client itself.

--------------------------------------------------------------------------
SUGGESTIONS — signatures + recipes are given; you fill in the bodies.
--------------------------------------------------------------------------
"""

# TODO(you): imports
#   - `from pymongo import MongoClient`
#   - `from . import config`   (relative import — pulls MONGODB_URI, DB_NAME, etc.)
from pymongo import MongoClient
from . import config

def get_client():  # -> MongoClient
    """Return a MongoClient connected to the Atlas cluster.

    Recipe:
      1. return MongoClient(config.MONGODB_URI)
      2. (optional, nice) pass `serverSelectionTimeoutMS=5000` so a bad URI
         fails in ~5s instead of hanging.
    """
    return MongoClient(config.MONGODB_URI)
    


def get_chunks_collection():  # -> Collection
    """Return the `bearlm.chunks` collection, creating nothing (Mongo is lazy).

    Recipe:
      1. client = get_client()
      2. return client[config.DB_NAME][config.CHUNKS_COLL]
    """
    client = get_client()
    return client[config.DB_NAME][config.CHUNKS_COLL]


def ping() -> bool:
    """Sanity check: can we actually reach the cluster?

    Recipe:
      1. get_client().admin.command("ping")
      2. return True  (let an exception propagate if it fails, or catch + return False)

    Handy to call once from a `__main__` block before you run the full ingest.
    """
    get_client().admin.command("ping")
    return True
