"""
Custom utilities for MongoDB Vector Search lessons
Reusable functions for Lessons 2 and 3
"""

import os
import pandas as pd
import openai
from typing import List, Optional
from pydantic import BaseModel, ValidationError
from datetime import datetime
from pymongo.mongo_client import MongoClient
from pymongo.operations import SearchIndexModel
import time

# Load environment variables
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MONGO_URI = os.environ.get("MONGO_URI")

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class Host(BaseModel):
    host_id: str
    host_url: str
    host_name: str
    host_location: str
    host_about: str
    host_response_time: Optional[str] = None
    host_thumbnail_url: str
    host_picture_url: str
    host_response_rate: Optional[int] = None
    host_is_superhost: bool
    host_has_profile_pic: bool
    host_identity_verified: bool


class Location(BaseModel):
    type: str
    coordinates: List[float]
    is_location_exact: bool


class Address(BaseModel):
    street: str
    government_area: str
    market: str
    country: str
    country_code: str
    location: Location


class Review(BaseModel):
    _id: str
    date: Optional[datetime] = None
    listing_id: str
    reviewer_id: str
    reviewer_name: Optional[str] = None
    comments: Optional[str] = None


class Listing(BaseModel):
    _id: int
    listing_url: str
    name: str
    summary: str
    space: str
    description: str
    neighborhood_overview: Optional[str] = None
    notes: Optional[str] = None
    transit: Optional[str] = None
    access: str
    interaction: Optional[str] = None
    house_rules: str
    property_type: str
    room_type: str
    bed_type: str
    minimum_nights: int
    maximum_nights: int
    cancellation_policy: str
    last_scraped: Optional[datetime] = None
    calendar_last_scraped: Optional[datetime] = None
    first_review: Optional[datetime] = None
    last_review: Optional[datetime] = None
    accommodates: int
    bedrooms: Optional[float] = 0
    beds: Optional[float] = 0
    number_of_reviews: int
    bathrooms: Optional[float] = 0
    amenities: List[str]
    price: int
    security_deposit: Optional[float] = None
    cleaning_fee: Optional[float] = None
    extra_people: int
    guests_included: int
    images: dict
    host: Host
    address: Address
    availability: dict
    review_scores: dict
    reviews: List[Review]
    text_embeddings: List[float]


# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

def process_records(dataset_df):
    """
    Convert pandas DataFrame to validated Listing objects.
    Handles NaT values and validates with Pydantic.

    Args:
        dataset_df: Pandas DataFrame with Airbnb listings

    Returns:
        List of validated listing dictionaries ready for MongoDB
    """
    records = dataset_df.to_dict(orient='records')

    # Handle NaT values (pandas NaT breaks Pydantic)
    for record in records:
        for key, value in record.items():
            # Check if the value is list-like; if so, process each element
            if isinstance(value, list):
                processed_list = [None if pd.isnull(v) else v for v in value]
                record[key] = processed_list
            # For scalar values, convert NaN/NaT to None
            else:
                if pd.isnull(value):
                    record[key] = None

    # Validate with Pydantic
    try:
        listings = [Listing(**record).dict() for record in records]
        print(f"✅ Processed {len(listings)} listings successfully")
        return listings
    except ValidationError as e:
        print(f"❌ Validation error: {e}")
        return []


# ============================================================================
# DATABASE CONNECTION
# ============================================================================

def connect_to_database(database_name="airbnb_dataset", collection_name="listings_reviews"):
    """
    Connect to MongoDB Atlas and return database and collection objects.

    Args:
        database_name: Name of the MongoDB database
        collection_name: Name of the collection

    Returns:
        Tuple of (database, collection)
    """
    if not MONGO_URI:
        raise ValueError("MONGO_URI not set in environment variables")

    client = MongoClient(MONGO_URI, appname="devrel.deeplearningai.lesson2-3.python")
    print("✅ Connection to MongoDB successful")

    db = client.get_database(database_name)
    collection = db.get_collection(collection_name)

    print(f"📋 Database: {database_name}")
    print(f"📋 Collection: {collection_name}")

    return db, collection


# ============================================================================
# VECTOR SEARCH INDEX SETUP
# ============================================================================

def setup_vector_search_index(collection, index_name="vector_index_text"):
    """
    Create a basic vector search index (Lesson 1 style).

    Args:
        collection: MongoDB collection
        index_name: Name for the vector search index
    """
    vector_search_index_model = SearchIndexModel(
        definition={
            "mappings": {
                "dynamic": True,
                "fields": {
                    "text_embeddings": {
                        "dimensions": 1536,
                        "similarity": "cosine",
                        "type": "knnVector",
                    }
                },
            }
        },
        name=index_name,
    )

    # Check if index already exists
    index_exists = False
    for index in collection.list_indexes():
        if index.get('name') == index_name:
            index_exists = True
            break

    if not index_exists:
        try:
            result = collection.create_search_index(model=vector_search_index_model)
            print("Creating index...")
            time.sleep(20)  # Wait for index to initialize
            print(f"✅ Index '{index_name}' created successfully:", result)
            print("💡 Wait a few minutes before conducting searches")
        except Exception as e:
            print(f"❌ Error creating vector search index: {str(e)}")
    else:
        print(f"✅ Index '{index_name}' already exists.")


def setup_vector_search_index_with_filter(collection, index_name="vector_index_with_filter"):
    """
    Create vector search index with filterable fields (Lesson 2 & 3 style).

    Args:
        collection: MongoDB collection
        index_name: Name for the vector search index
    """
    vector_search_index_model = SearchIndexModel(
        definition={
            "mappings": {
                "dynamic": True,
                "fields": {
                    "text_embeddings": {
                        "dimensions": 1536,
                        "similarity": "cosine",
                        "type": "knnVector",
                    },
                    "accommodates": {
                        "type": "number"
                    },
                    "bedrooms": {
                        "type": "number"
                    },
                },
            }
        },
        name=index_name,
    )

    # Check if index already exists
    index_exists = False
    for index in collection.list_indexes():
        if index.get('name') == index_name:
            index_exists = True
            break

    if not index_exists:
        try:
            result = collection.create_search_index(model=vector_search_index_model)
            print("Creating index with filters...")
            time.sleep(20)  # Wait for index to initialize
            print(f"✅ Index '{index_name}' created successfully:", result)
            print("💡 Wait a few minutes before conducting searches")
        except Exception as e:
            print(f"❌ Error creating vector search index: {str(e)}")
    else:
        print(f"✅ Index '{index_name}' already exists.")


# ============================================================================
# EMBEDDING GENERATION
# ============================================================================

def get_embedding(text):
    """
    Generate an embedding for the given text using OpenAI's API.

    Args:
        text: String to convert to embedding

    Returns:
        List of floats (1536 dimensions) or None if error
    """
    # Check for valid input
    if not text or not isinstance(text, str):
        return None

    try:
        # Call OpenAI API to get the embedding
        embedding = openai.embeddings.create(
            input=text,
            model="text-embedding-3-small",
            dimensions=1536
        ).data[0].embedding
        return embedding
    except Exception as e:
        print(f"❌ Error in get_embedding: {e}")
        return None


# ============================================================================
# VECTOR SEARCH FUNCTIONS
# ============================================================================

def vector_search_with_filter(user_query, db, collection, additional_stages=[], vector_index="vector_index_with_filter"):
    """
    Perform a vector search with optional pre-filtering and additional pipeline stages.

    Args:
        user_query: User's search query string
        db: MongoDB database object
        collection: MongoDB collection object
        additional_stages: Additional aggregation stages (like $project, $match)
        vector_index: Name of the vector search index to use

    Returns:
        List of matching documents
    """
    query_embedding = get_embedding(user_query)

    if query_embedding is None:
        return "Invalid query or embedding generation failed."

    # Vector search stage with pre-filtering
    vector_search_stage = {
        "$vectorSearch": {
            "index": vector_index,
            "queryVector": query_embedding,
            "path": "text_embeddings",
            "numCandidates": 150,
            "limit": 20,
            "filter": {
                "$and": [
                    {"accommodates": {"$gte": 2}},
                    {"bedrooms": {"$lte": 7}}
                ]
            },
        }
    }

    # Build pipeline
    pipeline = [vector_search_stage] + additional_stages

    # Execute search
    results = collection.aggregate(pipeline)

    # Get execution stats
    explain_query_execution = db.command(
        'explain', {
            'aggregate': collection.name,
            'pipeline': pipeline,
            'cursor': {}
        },
        verbosity='executionStats'
    )

    vector_search_explain = explain_query_execution['stages'][0]['$vectorSearch']
    millis_elapsed = vector_search_explain['explain']['collectors']['allCollectorStats']['millisElapsed']

    print(f"⚡ Search completed in {millis_elapsed} milliseconds")

    return list(results)
