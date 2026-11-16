"""Multi-agent trace generation with code-derived labels.

LLM agents (coordinator, researcher, communicator, admin) work on tasks in a
simulated workspace with mock tools.  Nothing has a real side effect.  Several
tasks can run concurrently, so their events interleave.  Whether a step is a
violation is decided by code from tokens planted in the workspace, never by a
person and never by a model.
"""
