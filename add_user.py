from api.data.db_manager import SessionLocal
from api.models.Users import Users
from api.utils import get_password_hash
from loguru import logger

username = "admin"
password = "admin123"

password_hash = get_password_hash(password)

db = SessionLocal()
user = Users(
    username=username,
    password_hash=password_hash
    )
db.add(user)
db.commit()
db.refresh(user)
db.close()
logger.info("New user added in database")
