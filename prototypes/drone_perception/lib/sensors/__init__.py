"""Sensor abstractions for different camera types."""

from .base import BaseSensor
from .monocular import MonocularCamera
from .stereo import StereoCamera, StereoRecordedCamera
from .lidar import LiDARCameraSensor
from .wide_angle import WideAngleCamera, DualWideAngleCamera

__all__ = [
    'BaseSensor',
    'MonocularCamera',
    'StereoCamera',
    'StereoRecordedCamera',
    'LiDARCameraSensor',
    'WideAngleCamera',
    'DualWideAngleCamera'
]
