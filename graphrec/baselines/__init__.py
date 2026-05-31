"""Non-graph baselines for comparison: popularity, item-CF, matrix factorization."""

from graphrec.baselines.item_cf import ItemCFRecommender
from graphrec.baselines.mf import SVDRecommender
from graphrec.baselines.popularity import PopularityRecommender

__all__ = ["ItemCFRecommender", "PopularityRecommender", "SVDRecommender"]
