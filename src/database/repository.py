"""Database repository layer for the electricity price prediction application."""

import logging
import os

import pandas as pd
from sqlmodel import Session, SQLModel, create_engine, text

logger = logging.getLogger(__name__)


class DatabaseRepository:
    """Repository class for database operations."""

    def __init__(self, database_name: str):
        self.database_name = database_name
        self.engine = self._create_engine()

    def _create_engine(self):
        """Create database engine."""
        database_url = os.getenv("DATABASE_URL", f"sqlite:///data/{self.database_name}.db")
        return create_engine(database_url)

    def get_data(self, columns: str, table_name: str, where_clause: str = "") -> pd.DataFrame:
        """Get data from database table."""
        try:
            query = f"SELECT {columns} FROM {table_name}"
            if where_clause:
                query += f" WHERE {where_clause}"

            with Session(self.engine) as session:
                result = session.execute(text(query))
                df = pd.DataFrame(result.fetchall(), columns=result.keys())

                if "index" in df.columns:
                    df = df.set_index("index")

                return df
        except Exception as e:
            logger.error(f"Error getting data from {table_name}: {e}")
            return pd.DataFrame()

    def get_dataset(self, columns: str, dataset_name: str) -> pd.DataFrame:
        """Get dataset from database."""
        return self.get_data(columns, dataset_name)

    def save_df_to_db(self, df: pd.DataFrame, table_name: str, if_exists: str = "replace"):
        """Save DataFrame to database table."""
        try:
            with Session(self.engine) as session:
                df.to_sql(table_name, con=self.engine, if_exists=if_exists, index=True)
                session.commit()
                logger.info(f"Saved DataFrame to {table_name}")
        except Exception as e:
            logger.error(f"Error saving DataFrame to {table_name}: {e}")

    def get_metrics(self, model_name: str) -> pd.DataFrame:
        """Get model metrics from database."""
        return self.get_data("*", f"metrics_{model_name}")

    def save_metrics(self, metrics: pd.DataFrame, model_name: str):
        """Save model metrics to database."""
        self.save_df_to_db(metrics, f"metrics_{model_name}")

    def create_tables(self):
        """Create database tables."""
        SQLModel.metadata.create_all(self.engine)


# Legacy compatibility class
class DB(DatabaseRepository):
    """Legacy DB class for backward compatibility."""

    pass
