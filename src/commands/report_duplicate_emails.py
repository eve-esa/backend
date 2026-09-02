"""Report users whose email addresses collide once case is normalised.

Read-only. Run it before anyone makes ``users.email`` unique: building a unique
index on a collection that already holds duplicates fails the build, and doing
that discovery on production at startup is not the moment to find out.

    python -m src.commands.report_duplicate_emails

Exit code 0 with no rows means the index can be tightened.
"""

import asyncio
import logging
import sys

from src.config import configure_logging
from src.database.models.user import User
from src.database.mongo import async_mongo_manager

configure_logging()
logger = logging.getLogger(__name__)


async def report_duplicate_emails() -> int:
    """Print every lowercased address held by more than one user row."""
    await async_mongo_manager.connect()

    pipeline = [
        {
            "$group": {
                "_id": {"$toLower": "$email"},
                "count": {"$sum": 1},
                "ids": {"$push": "$_id"},
            }
        },
        {"$match": {"count": {"$gt": 1}}},
        {"$sort": {"count": -1}},
    ]

    duplicates = 0
    async for row in User.get_collection().aggregate(pipeline):
        duplicates += 1
        ids = ", ".join(str(value) for value in row.get("ids", []))
        print(f"{row['_id']}\t{row['count']}\t{ids}")

    if duplicates == 0:
        print("No duplicate emails: users.email can be made unique.")
    else:
        print(f"{duplicates} address(es) held by more than one user.")
    return duplicates


if __name__ == "__main__":
    sys.exit(0 if asyncio.run(report_duplicate_emails()) == 0 else 1)
