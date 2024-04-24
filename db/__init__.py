from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Create the SQLite database engine
engine = create_engine('sqlite:///data_migrator.db', echo=True)

# Create a base class for declarative models
Base = declarative_base()

# Define the entity (table) class
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    age = Column(Integer)

# Create the table in the database
Base.metadata.create_all(engine)