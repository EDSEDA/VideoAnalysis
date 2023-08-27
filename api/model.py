from sqlalchemy import (Column, DateTime,
                        Integer, String, Boolean, ForeignKey, Date, JSON, UniqueConstraint, Enum, Numeric,
                        BigInteger, Computed)

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

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
    user_id = Column(Integer(), ForeignKey(User.id, ondelete='CASCADE'), nullable=False, comment='Сборщик эмоций')
    anger = Column(Integer())
    fear = Column(Integer())
    happy = Column(Integer())
    neutral = Column(Integer())
    sadness = Column(Integer())
    surprized = Column(Integer())
    datetime = Column(DateTime())
