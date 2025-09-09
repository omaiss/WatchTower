# main.py - Main surveillance system
import cv2
import torch
import numpy as np
import sqlite3
import os
import json
from datetime import datetime
from pathlib import Path
import threading
import time
from dataclasses import dataclass
from collections import defaultdict
import easyocr
import hashlib

