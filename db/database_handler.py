from sqlalchemy import Column, Integer, String, create_engine, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

Base = declarative_base()


class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    age = Column(Integer)


class TransactionTrainingData(Base):
    __tablename__ = 'transactions_training_data'
    id = Column(Integer, primary_key=True, autoincrement=True)
    recipient = Column(String)
    booking_text = Column(String)
    purpose_of_transaction = Column(String)
    category = Column(String)

    # If you need an auto-incrementing ID as well, you can include it as follows:
    # id = Column(Integer, primary_key=True, autoincrement=True)


class UserSelectedTransactionsCategories(Base):
    __tablename__ = 'user_selected_transactions_categories'
    user_selection_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    recipient = Column(String)
    booking_text = Column(String)
    purpose_of_transaction = Column(String)
    category = Column(String)

    # Define the relationship with the User model
    user = relationship(User, backref="user_selected_transactions_categories")


class DatabaseHandler:
    def __init__(self, db_uri):
        self.engine = create_engine(db_uri)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def add_user(self, name, age):
        Session = sessionmaker(bind=self.engine)
        session = Session()
        new_user = User(name=name, age=age)
        session.add(new_user)
        session.commit()
        session.close()

    def get_all_users(self):
        Session = sessionmaker(bind=self.engine)
        session = Session()
        users = session.query(User).all()
        session.close()
        return users

    def update_user_age(self, name, new_age):
        Session = sessionmaker(bind=self.engine)
        session = Session()
        user = session.query(User).filter_by(name=name).first()
        if user:
            user.age = new_age
            session.commit()
            session.close()
            return True
        session.close()
        return False

    def add_entry(self, entity):
        Session = sessionmaker(bind=self.engine)
        session = Session()
        session.add(entity)
        session.commit()
        session.close()

    def add_entities(self, entities):
        Session = sessionmaker(bind=self.engine)
        session = Session()
        try:
            # Add all the entities to the session
            session.add_all(entities)
            # Commit the changes to the database
            session.commit()
            # Close the session
            session.close()
            return True
        except Exception as e:
            print(f"Error adding entries: {e}")
            session.rollback()
            session.close()
            return False

    def truncate_table(self, table_name):
        Session = sessionmaker(bind=self.engine)
        session = Session()
        try:
            # Get the table object based on the model
            table = Base.metadata.tables[table_name]

            # Construct the TRUNCATE SQL statement
            truncate_sql = f"TRUNCATE TABLE {table_name}"

            # Execute the SQL statement
            session.execute(truncate_sql)

            # Commit the changes to the database
            session.commit()

            # Close the session
            session.close()
            return True
        except Exception as e:
            print(f"Error truncating table: {e}")
            session.rollback()
            session.close()
            return False

    def get_all_user_selected_transactions_categories(self):
        Session = sessionmaker(bind=self.engine)
        session = Session()
        userSelectedTransactionsCategories = session.query(UserSelectedTransactionsCategories).all()
        session.close()
        return userSelectedTransactionsCategories

    def get_all_transaction_Training_data(self):
        Session = sessionmaker(bind=self.engine)
        session = Session()
        transactionTrainingData = session.query(TransactionTrainingData).all()
        session.close()
        return transactionTrainingData
    def close_connection(self):
        # You can add additional clean-up or close procedures here if necessary.
        pass
