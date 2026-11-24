"""Service desk intake-and-triage prototype.

Models read and draft; code decides. Every rule the pipeline applies comes
from a policy table compiled from the knowledge base, except a few hard
limits in `guards.py` that no KB edit can remove.
"""
