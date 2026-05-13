"""Distance metrics accepted for Supabase Storage vector indexes (vector buckets)."""

from src.enums.base import BaseEnum


class VectorBucketDistanceMetric(BaseEnum):
    """Metrics accepted by Storage vector indexes (Supabase naming)."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    # Supabase docs reference L2; typed clients may widen literal unions over time.
    L2 = "l2"
