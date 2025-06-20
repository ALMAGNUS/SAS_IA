from sqlalchemy import Column, Integer, String
from api.data.db_manager import Base

class Users(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String)
    password_hash = Column(String)
