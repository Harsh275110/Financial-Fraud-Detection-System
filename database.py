import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv

load_dotenv()

class DatabaseConnection:
    def __init__(self):
        self.conn = None
        self.cursor = None
        
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(
                host=os.getenv('DB_HOST'),
                database=os.getenv('DB_NAME'),
                user=os.getenv('DB_USER'),
                password=os.getenv('DB_PASSWORD'),
                port=os.getenv('DB_PORT', 5432)
            )
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            return True
        except Exception as e:
            print(f"Error connecting to database: {str(e)}")
            return False

    def disconnect(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def store_transaction(self, transaction_data, prediction, fraud_probability):
        """Store transaction and prediction in database"""
        try:
            query = """
                INSERT INTO transactions (
                    transaction_id,
                    customer_id,
                    amount,
                    transaction_datetime,
                    merchant_id,
                    merchant_category,
                    predicted_fraud,
                    fraud_probability
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            values = (
                transaction_data['transaction_id'],
                transaction_data['customer_id'],
                transaction_data['amount'],
                transaction_data['transaction_datetime'],
                transaction_data['merchant_id'],
                transaction_data['merchant_category'],
                prediction,
                fraud_probability
            )
            self.cursor.execute(query, values)
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error storing transaction: {str(e)}")
            self.conn.rollback()
            return False

    def get_customer_transactions(self, customer_id):
        """Retrieve customer transaction history"""
        try:
            query = """
                SELECT *
                FROM transactions
                WHERE customer_id = %s
                ORDER BY transaction_datetime DESC
            """
            self.cursor.execute(query, (customer_id,))
            return self.cursor.fetchall()
        except Exception as e:
            print(f"Error retrieving customer transactions: {str(e)}")
            return []

    def get_fraud_statistics(self, start_date=None, end_date=None):
        """Get fraud statistics for reporting"""
        try:
            query = """
                SELECT 
                    DATE(transaction_datetime) as date,
                    COUNT(*) as total_transactions,
                    SUM(CASE WHEN predicted_fraud = true THEN 1 ELSE 0 END) as fraud_count,
                    AVG(CASE WHEN predicted_fraud = true THEN amount ELSE 0 END) as avg_fraud_amount
                FROM transactions
                WHERE ($1 IS NULL OR transaction_datetime >= $1)
                    AND ($2 IS NULL OR transaction_datetime <= $2)
                GROUP BY DATE(transaction_datetime)
                ORDER BY date DESC
            """
            self.cursor.execute(query, (start_date, end_date))
            return self.cursor.fetchall()
        except Exception as e:
            print(f"Error retrieving fraud statistics: {str(e)}")
            return []

    def create_tables(self):
        """Create necessary database tables"""
        try:
            # Create transactions table
            create_transactions_table = """
                CREATE TABLE IF NOT EXISTS transactions (
                    id SERIAL PRIMARY KEY,
                    transaction_id VARCHAR(50) UNIQUE,
                    customer_id VARCHAR(50),
                    amount DECIMAL(10,2),
                    transaction_datetime TIMESTAMP,
                    merchant_id VARCHAR(50),
                    merchant_category VARCHAR(100),
                    predicted_fraud BOOLEAN,
                    fraud_probability FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            self.cursor.execute(create_transactions_table)
            
            # Create indexes
            create_indexes = [
                "CREATE INDEX IF NOT EXISTS idx_customer_id ON transactions(customer_id)",
                "CREATE INDEX IF NOT EXISTS idx_transaction_datetime ON transactions(transaction_datetime)",
                "CREATE INDEX IF NOT EXISTS idx_predicted_fraud ON transactions(predicted_fraud)"
            ]
            
            for index_query in create_indexes:
                self.cursor.execute(index_query)
            
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error creating tables: {str(e)}")
            self.conn.rollback()
            return False 