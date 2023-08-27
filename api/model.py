from sqlalchemy import (Column, DateTime,
                        Integer, String, Boolean, ForeignKey, Date, JSON, UniqueConstraint, Enum, Numeric,
                        BigInteger, Computed)
from sqlalchemy import Index
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, declared_attr
from sqlalchemy.sql import func
from typing import Dict

from api.config import WORK_SCHEMA


Base = declarative_base()


class WithID:
    id = Column(Integer(), primary_key=True, comment='Unique ID')


class CreatedUpdated:
    created = Column(DateTime(), server_default=func.now())
    updated = Column(DateTime(), server_default=func.now(), onupdate=func.now())


class ConfigMixin:
    __table_args__ = {'schema': WORK_SCHEMA}


class User(Base, ConfigMixin):
    __tablename__ = "user"
    __table_args__ = {**ConfigMixin.__table_args__, **{'comment': 'User of system'}}
    name = Column(String(), comment='Name')
    lastname = Column(String(), comment='Lastname')
    role = Column(Boolean(), comment='Users role')


class Emotion(Base, ConfigMixin):
    __tablename__ = "emotion"
    __table_args__ = {**ConfigMixin.__table_args__, **{'comment': 'Emotions metrics'}}
    user_id = Column(Integer())
    datetime = Column(String())

    # emotion1
    # emotion2
    # emotion3
    # emotion4
    # emotion5
    # datetime
