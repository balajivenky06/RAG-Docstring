"""
Advanced RAG Variants.
Combines core RAG strategies with advanced reasoning (CoT, ToT, GoT).
"""


from .self_rag import SelfCorrectionRAG
from .reasoning_mixins import CoTMixin, ToTMixin, GoTMixin




# Self-Correction RAG Variants
class CoTSelfCorrectionRAG(CoTMixin, SelfCorrectionRAG):
    """Self-Correction RAG with Chain of Thought reasoning."""
    pass

class ToTSelfCorrectionRAG(ToTMixin, SelfCorrectionRAG):
    """Self-Correction RAG with Tree of Thought reasoning."""
    pass

class GoTSelfCorrectionRAG(GoTMixin, SelfCorrectionRAG):
    """Self-Correction RAG with Graph of Thought reasoning."""
    pass
