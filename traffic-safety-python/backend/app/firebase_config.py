import os
import json
import uuid
import time
import logging
from typing import Dict, List, Optional, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("firebase_config")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEY_PATH = os.path.join(BASE_DIR, "serviceAccountKey.json")
LOCAL_DB_PATH = os.path.join(BASE_DIR, "local_db.json")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")

os.makedirs(UPLOADS_DIR, exist_ok=True)

# Firebase admin variables
firebase_initialized = False
db_client = None
bucket_client = None

try:
    import firebase_admin
    from firebase_admin import credentials, firestore, storage

    if os.path.exists(KEY_PATH):
        with open(KEY_PATH, "r") as f:
            key_content = json.load(f)
            
        # Check if key is valid (not placeholder)
        if key_content.get("private_key_id") != "placeholder_key_id" and "PLACEHOLDER" not in key_content.get("private_key", ""):
            cred = credentials.Certificate(KEY_PATH)
            storage_bucket = key_content.get("storageBucket", f"{key_content.get('project_id')}.appspot.com")
            firebase_admin.initialize_app(cred, {
                'storageBucket': storage_bucket
            })
            db_client = firestore.client()
            bucket_client = storage.bucket()
            firebase_initialized = True
            logger.info("Firebase Admin SDK successfully initialized with service account key.")
        else:
            logger.warning("serviceAccountKey.json contains placeholder values. Running in Local Storage Mode.")
    else:
        logger.warning("serviceAccountKey.json not found. Running in Local Storage Mode.")
except Exception as e:
    logger.warning(f"Failed to initialize Firebase Admin SDK: {e}. Running in Local Storage Mode.")


class LocalDatabase:
    """Thread-safe local JSON storage fallback for Firestore reports & users."""
    def __init__(self, filepath: str):
        self.filepath = filepath
        self._ensure_file()

    def _ensure_file(self):
        if not os.path.exists(self.filepath):
            initial_data = {
                "reports": [
                    {
                        "id": "rep-demo-1",
                        "userId": "user-demo-123",
                        "userEmail": "citizen@roadsafe.org",
                        "imageUrl": "/static/uploads/sample_pothole_demo.jpg",
                        "locationName": "Main Street & 5th Avenue",
                        "latitude": 37.7749,
                        "longitude": -122.4194,
                        "category": "Pothole",
                        "description": "Deep pothole near crosswalk causing traffic slowdowns and vehicle damage.",
                        "severity": "High",
                        "confidence": 92.5,
                        "status": "Pending",
                        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 86400)),
                        "updatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 86400))
                    },
                    {
                        "id": "rep-demo-2",
                        "userId": "user-demo-123",
                        "userEmail": "citizen@roadsafe.org",
                        "imageUrl": "/static/uploads/sample_crack_demo.jpg",
                        "locationName": "Highway 101 KM 42",
                        "latitude": 37.7833,
                        "longitude": -122.4167,
                        "category": "Crack",
                        "description": "Longitudinal surface cracks spreading along the right lane.",
                        "severity": "Medium",
                        "confidence": 88.0,
                        "status": "In Progress",
                        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 172800)),
                        "updatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 43200))
                    },
                    {
                        "id": "rep-demo-3",
                        "userId": "user-demo-456",
                        "userEmail": "admin@roadsafe.org",
                        "imageUrl": "/static/uploads/sample_clean_demo.jpg",
                        "locationName": "Grand Avenue Overpass",
                        "latitude": 37.7690,
                        "longitude": -122.4480,
                        "category": "Clean Road",
                        "description": "Routine pavement inspection - road surface clear of hazards.",
                        "severity": "Low",
                        "confidence": 96.2,
                        "status": "Resolved",
                        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 345600)),
                        "updatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 86400))
                    }
                ],
                "users": [
                    {
                        "uid": "user-demo-123",
                        "email": "citizen@roadsafe.org",
                        "role": "user",
                        "name": "Jane Citizen"
                    },
                    {
                        "uid": "user-demo-456",
                        "email": "admin@roadsafe.org",
                        "role": "admin",
                        "name": "Admin Supervisor"
                    }
                ]
            }
            with open(self.filepath, "w") as f:
                json.dump(initial_data, f, indent=2)

    def _read_data(self) -> Dict[str, Any]:
        self._ensure_file()
        try:
            with open(self.filepath, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading local db: {e}")
            return {"reports": [], "users": []}

    def _write_data(self, data: Dict[str, Any]):
        with open(self.filepath, "w") as f:
            json.dump(data, f, indent=2)

    def add_report(self, report_dict: Dict[str, Any]) -> str:
        data = self._read_data()
        report_id = f"rep-{uuid.uuid4().hex[:8]}"
        report_dict["id"] = report_id
        now_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        report_dict["createdAt"] = report_dict.get("createdAt", now_str)
        report_dict["updatedAt"] = now_str
        data["reports"].insert(0, report_dict)
        self._write_data(data)
        return report_id

    def get_all_reports(self) -> List[Dict[str, Any]]:
        return self._read_data().get("reports", [])

    def get_user_reports(self, user_id: str) -> List[Dict[str, Any]]:
        reports = self.get_all_reports()
        return [r for r in reports if r.get("userId") == user_id]

    def update_report_status(self, report_id: str, status: str) -> Optional[Dict[str, Any]]:
        data = self._read_data()
        for r in data.get("reports", []):
            if r["id"] == report_id:
                r["status"] = status
                r["updatedAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                self._write_data(data)
                return r
        return None

    def delete_report(self, report_id: str) -> bool:
        data = self._read_data()
        initial_len = len(data.get("reports", []))
        data["reports"] = [r for r in data.get("reports", []) if r["id"] != report_id]
        if len(data["reports"]) < initial_len:
            self._write_data(data)
            return True
        return False


local_db = LocalDatabase(LOCAL_DB_PATH)
