#!/usr/bin/python
# -*- coding: utf-*-#!/usr/bin/python
# # -*- coding: utf-*-
import random
from datetime import datetime
from typing import List, Dict

from data_collector import DataReader, CollectedVariable

RANDOM_DATA_READER_TYPE: str = "random"
RANDOM_DATA_READER_MIN_INTERVAL: int = 100


class RandomDataReader(DataReader):
    def __init__(self, reader_type: str, min_value: float = 15.0, max_value: float = 18.0):
        """
        Initialize the reader
        :param reader_type: the type
        :param min_value: the min value
        :param max_value: the max value
        """
        if reader_type is None:
            self._type = RANDOM_DATA_READER_TYPE
        else:
            self._type = reader_type
        self._min = min_value
        self._max = max_value

    def get_type(self) -> str:
        return self._type

    def get_min_interval(self) -> int:
        return RANDOM_DATA_READER_MIN_INTERVAL

    def read(self, variables: List[CollectedVariable]) -> None:
        now: datetime = datetime.now()
        for variable in variables:
            variable.set_time(now)
            variable.set_value(random.uniform(self._min, self._max))
