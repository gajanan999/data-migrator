from sqlalchemy.exc import NoResultFound
from .database_handler import DatabaseHandler
from db.database_handler import UserSelectedTransactionsCategories


class UserCategorySelectionRepository:

    def __init__(self, db_uri):
        self.db_handler = DatabaseHandler(db_uri)

    def get_user_selected_transactions_category(self, user_id, recipient, booking_text, purpose_of_transaction):
        try:
            # Check if the record already exists based on user_id, recipient, booking_text, and purpose_of_transaction
            existing_entry = self.db_handler.session.query(UserSelectedTransactionsCategories).filter_by(
                user_id=user_id,
                recipient=recipient,
                booking_text=booking_text,
                purpose_of_transaction=purpose_of_transaction
            ).one()

            # If the record already exists, you can choose to update it, or you can skip adding it.
            # For example, updating the category:
            return existing_entry.category
        except NoResultFound:
            return ''

    def add_user_selected_transactions_category(self, user_id, recipient, booking_text, purpose_of_transaction,
                                                category):
        try:
            # Check if the record already exists based on user_id, recipient, booking_text, and purpose_of_transaction
            existing_entry = self.db_handler.session.query(UserSelectedTransactionsCategories).filter_by(
                user_id=user_id,
                recipient=recipient,
                booking_text=booking_text,
                purpose_of_transaction=purpose_of_transaction
            ).one()

            # If the record already exists, you can choose to update it, or you can skip adding it.
            # For example, updating the category:
            existing_entry.category = category
            print('User selection is already exits.')
        except NoResultFound:
            print('User selection is not exits.')
            # If the record does not exist, add it to the database
            new_entry = UserSelectedTransactionsCategories(
                user_id=user_id,
                recipient=recipient,
                booking_text=booking_text,
                purpose_of_transaction=purpose_of_transaction,
                category=category
            )
            self.db_handler.add_entry(new_entry)

        self.db_handler.session.commit()

    def get_all_user_selected_transactions_categories(self):
        user_selected_transactions_categories = self.db_handler.get_all_user_selected_transactions_categories()
        user_selected_transactions_categories_data = []
        for entry in user_selected_transactions_categories:
            entry_data = {
                'user_id': entry.user_id,
                'recipient': entry.recipient,
                'booking_text': entry.booking_text,
                'purpose_of_transaction': entry.purpose_of_transaction,
                'category': entry.category
            }
            user_selected_transactions_categories_data.append(entry_data)
        return user_selected_transactions_categories_data

    def close_connection(self):
        self.db_handler.close_connection()
