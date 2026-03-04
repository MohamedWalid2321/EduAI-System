"""
routes package — each AI module has its own route file.

Import routers here for convenient access from main.py:
    from routes import object_router, gaze_router, face_router
"""

from routes.object_route import router as object_router
from routes.gaze_route import router as gaze_router
from routes.face_route import router as face_router
from routes.new_object_route import router as new_object_route

__all__ = ["object_router", "gaze_router", "face_router", "new_object_route"]
