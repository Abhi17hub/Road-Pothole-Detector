import os
import uuid
import time
import logging
from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Header
from pydantic import BaseModel, Field

from app.detector import process_image_bytes
from app.firebase_config import (
    firebase_initialized, db_client, bucket_client, local_db, UPLOADS_DIR
)

router = APIRouter(prefix="/reports", tags=["Road Safety Reports"])
logger = logging.getLogger("report_routes")


class CreateReportPayload(BaseModel):
    userId: str = Field(..., example="user-demo-123")
    userEmail: str = Field(..., example="citizen@roadsafe.org")
    imageUrl: str = Field(..., example="/static/uploads/annotated_123.jpg")
    locationName: str = Field(..., example="Corner of 4th St & Broadway")
    latitude: float = Field(..., example=37.7749)
    longitude: float = Field(..., example=-122.4194)
    category: str = Field(..., example="Pothole")
    description: str = Field(..., example="Deep pothole causing vehicle tire damage.")
    severity: str = Field(..., example="High")
    confidence: float = Field(..., example=94.5)
    status: Optional[str] = "Pending"


class StatusUpdatePayload(BaseModel):
    status: str = Field(..., example="In Progress")


@router.post("/detect-damage")
async def detect_damage(file: UploadFile = File(...)):
    """
    Inference endpoint: Upload image -> Run YOLOv8 & OpenCV computer vision pipeline ->
    Overlay bounding boxes & severity -> Upload annotated image -> Return detection metadata.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File uploaded must be a valid image format (JPEG/PNG/WEBP).")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        annotated_bytes, metadata = process_image_bytes(contents)
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"AI Computer Vision detection failed: {str(e)}")

    filename = f"annotated_{uuid.uuid4().hex[:10]}.jpg"
    image_url = ""

    # Attempt Firebase Storage upload if initialized
    if firebase_initialized and bucket_client:
        try:
            blob = bucket_client.blob(f"road_reports/{filename}")
            blob.upload_from_string(annotated_bytes, content_type="image/jpeg")
            blob.make_public()
            image_url = blob.public_url
        except Exception as fb_err:
            logger.warning(f"Firebase Storage upload failed: {fb_err}. Saving locally.")

    # Fallback / primary local static file saving
    if not image_url:
        local_path = os.path.join(UPLOADS_DIR, filename)
        with open(local_path, "wb") as f:
            f.write(annotated_bytes)
        image_url = f"/static/uploads/{filename}"

    metadata["imageURL"] = image_url
    return {
        "status": "success",
        "data": metadata
    }


@router.post("")
def create_report(payload: CreateReportPayload):
    """Saves a new road hazard report into Firebase Cloud Firestore or Local DB."""
    report_data = payload.dict()
    now_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report_data["createdAt"] = now_str
    report_data["updatedAt"] = now_str
    
    if firebase_initialized and db_client:
        try:
            doc_ref = db_client.collection("reports").document()
            report_data["id"] = doc_ref.id
            doc_ref.set(report_data)
            return {"status": "success", "message": "Report saved to Firestore", "data": report_data}
        except Exception as e:
            logger.warning(f"Firestore save error: {e}. Using local storage.")

    # Local fallback
    report_id = local_db.add_report(report_data)
    report_data["id"] = report_id
    return {"status": "success", "message": "Report saved to local database", "data": report_data}


@router.get("/my-reports")
def get_my_reports(user_id: Optional[str] = "user-demo-123", authorization: Optional[str] = Header(None)):
    """Retrieves reports submitted by logged-in citizen user."""
    if firebase_initialized and db_client:
        try:
            docs = db_client.collection("reports").where("userId", "==", user_id).stream()
            reports = [doc.to_dict() for doc in docs]
            return {"status": "success", "count": len(reports), "data": reports}
        except Exception as e:
            logger.warning(f"Firestore get_my_reports error: {e}")

    reports = local_db.get_user_reports(user_id)
    return {"status": "success", "count": len(reports), "data": reports}


@router.get("/all")
def get_all_reports():
    """Admin endpoint to retrieve all road reports."""
    if firebase_initialized and db_client:
        try:
            docs = db_client.collection("reports").order_by("createdAt", direction="DESCENDING").stream()
            reports = [doc.to_dict() for doc in docs]
            return {"status": "success", "count": len(reports), "data": reports}
        except Exception as e:
            logger.warning(f"Firestore get_all_reports error: {e}")

    reports = local_db.get_all_reports()
    return {"status": "success", "count": len(reports), "data": reports}


@router.patch("/{report_id}/status")
def update_report_status(report_id: str, payload: StatusUpdatePayload):
    """Admin endpoint to update report status (Pending, Verified, In Progress, Resolved, Rejected)."""
    valid_statuses = ["Pending", "Verified", "In Progress", "Resolved", "Rejected"]
    if payload.status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Status must be one of {valid_statuses}")

    if firebase_initialized and db_client:
        try:
            doc_ref = db_client.collection("reports").document(report_id)
            if doc_ref.get().exists:
                doc_ref.update({
                    "status": payload.status,
                    "updatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                })
                updated_doc = doc_ref.get().to_dict()
                return {"status": "success", "message": "Report status updated in Firestore", "data": updated_doc}
        except Exception as e:
            logger.warning(f"Firestore update_status error: {e}")

    updated = local_db.update_report_status(report_id, payload.status)
    if not updated:
        raise HTTPException(status_code=404, detail="Report ID not found.")
    return {"status": "success", "message": "Report status updated", "data": updated}


@router.delete("/{report_id}")
def delete_report(report_id: str):
    """Admin route to delete a report from Firestore / Local DB and remove its image."""
    if firebase_initialized and db_client:
        try:
            doc_ref = db_client.collection("reports").document(report_id)
            doc_snap = doc_ref.get()
            if doc_snap.exists:
                doc_data = doc_snap.to_dict()
                doc_ref.delete()
                # Remove image from storage if bucket available
                img_url = doc_data.get("imageUrl", "")
                if bucket_client and "road_reports/" in img_url:
                    blob_name = "road_reports/" + img_url.split("road_reports/")[-1]
                    try:
                        bucket_client.blob(blob_name).delete()
                    except Exception:
                        pass
                return {"status": "success", "message": "Report deleted from Firestore"}
        except Exception as e:
            logger.warning(f"Firestore delete_report error: {e}")

    deleted = local_db.delete_report(report_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Report ID not found.")
    return {"status": "success", "message": "Report deleted successfully"}
