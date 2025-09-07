from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.db.session import get_db
from app.models.user import User
from app.models.enums import UserRole
from app.schemas.user import UserOut
from app.core.deps import require_role

router = APIRouter(prefix="/users", tags=["users"])

@router.get("/", response_model=List[UserOut], dependencies=[Depends(require_role(UserRole.admin))])
def list_users(db: Session = Depends(get_db)):
    return db.query(User).order_by(User.created_at.desc()).all()
