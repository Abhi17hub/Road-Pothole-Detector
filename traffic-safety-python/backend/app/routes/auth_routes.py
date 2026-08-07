from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel, EmailStr
from typing import Optional
import logging
from app.firebase_config import firebase_initialized, local_db

router = APIRouter(prefix="/auth", tags=["Authentication"])
logger = logging.getLogger("auth_routes")


class UserLoginRequest(BaseModel):
    email: EmailStr
    password: str
    portal: Optional[str] = "citizen"  # "citizen" or "admin"


class UserRegisterRequest(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str] = "Citizen User"
    role: Optional[str] = "user"


@router.post("/login")
def login_user(payload: UserLoginRequest):
    """Logs in user via Firebase or Local fallback auth."""
    email = payload.email.lower()
    
    # Check portal specific rules
    if payload.portal == "admin":
        if "admin" not in email and not email.startswith("admin"):
            raise HTTPException(status_code=403, detail="Access denied. Admin portal requires a registered municipal admin account (e.g., admin@roadsafe.org).")
        role = "admin"
        name = "Municipal Admin Supervisor"
    else:
        role = "user"
        name = email.split("@")[0].title() + " (Citizen)"

    user_id = f"user-{abs(hash(email)) % 1000000}"
    
    return {
        "status": "success",
        "message": f"Successfully logged into {role.upper()} portal",
        "user": {
            "uid": user_id,
            "email": email,
            "name": name,
            "role": role,
            "token": f"mock-jwt-token-{role}-{user_id}"
        }
    }


@router.post("/register")
def register_user(payload: UserRegisterRequest):
    """Registers new citizen or admin account."""
    email = payload.email.lower()
    role = payload.role if payload.role in ["user", "admin"] else ("admin" if "admin" in email else "user")
    user_id = f"user-{abs(hash(email)) % 1000000}"
    
    return {
        "status": "success",
        "message": "User registered successfully",
        "user": {
            "uid": user_id,
            "email": email,
            "name": payload.name or email.split("@")[0].title(),
            "role": role,
            "token": f"mock-jwt-token-{role}-{user_id}"
        }
    }


@router.get("/me")
def get_current_user_profile(authorization: Optional[str] = Header(None)):
    """Returns profile for currently authenticated token."""
    if not authorization:
        return {
            "uid": "user-demo-123",
            "email": "citizen@roadsafe.org",
            "name": "Jane Citizen",
            "role": "user"
        }
    token = authorization.replace("Bearer ", "")
    if "admin" in token:
        return {
            "uid": "user-demo-456",
            "email": "admin@roadsafe.org",
            "name": "Admin Supervisor",
            "role": "admin"
        }
    return {
        "uid": "user-demo-123",
        "email": "citizen@roadsafe.org",
        "name": "Jane Citizen",
        "role": "user"
    }

