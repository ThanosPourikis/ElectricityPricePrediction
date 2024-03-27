from datetime import datetime

from sqlmodel import SQLModel, Field


class SMPModel(SQLModel):
    timestamp: datetime = Field(default=None, primary_key=True)
    value: float


class Dam(SMPModel, table=True):
    pass


class Lida1(SMPModel, table=True):
    pass


class Lida2(SMPModel, table=True):
    pass


class Lida3(SMPModel, table=True):
    pass
