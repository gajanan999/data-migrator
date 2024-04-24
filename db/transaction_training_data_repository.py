from sqlalchemy.exc import NoResultFound

from .database_handler import DatabaseHandler
from db.database_handler import TransactionTrainingData


class TransactionTrainingDataRepository:

    def __init__(self, db_uri):
        self.db_handler = DatabaseHandler(db_uri)

    def add_transaction_training_data(self, entities):
        return self.db_handler.add_entities(entities)

    def delete_all_records(self):
        return self.db_handler.truncate_table(TransactionTrainingData)

    def get_transaction_training_data(self):
        transaction_training_Data = self.db_handler.get_all_transaction_Training_data()
        transaction_training_Data_copy = []
        for entry in transaction_training_Data:
            entry_data = {
                'recipient': entry.recipient,
                'booking_text': entry.booking_text,
                'purpose_of_transaction': entry.purpose_of_transaction,
                'category': entry.category
            }
            transaction_training_Data_copy.append(entry_data)
        return transaction_training_Data_copy



    def close_connection(self):
        self.db_handler.close_connection()
