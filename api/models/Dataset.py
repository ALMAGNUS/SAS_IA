from sqlalchemy import Column, Integer, Text
from api.data.db_manager import Base


class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(Integer, primary_key=True, autoincrement=True)
    data = Column(Text)  # Stockage sous forme de string (JSON)
