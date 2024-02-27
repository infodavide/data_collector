# -*- coding: utf-8 -*-
"""
Random data reader that can be configured to define the minimum and maximum bounds.
"""
import random
from datetime import datetime
from typing import List

from data_collector import DataReader, CollectedVariable

RANDOM_DATA_READER_TYPE: str = "random"
RANDOM_DATA_READER_MIN_INTERVAL: int = 100


class RandomDataReader(DataReader):
    """
    Random data reader that can be configured to define the minimum and maximum bounds
    """
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
        """
        See DataReader class
        :return: the type
        """
        return self._type

    def get_min_interval(self) -> int:
        """
        See DataReader class
        :return: the smallest allowed interval
        """
        return RANDOM_DATA_READER_MIN_INTERVAL

    def read(self, variables: List[CollectedVariable]) -> None:
        """
       See DataReader class
       :param variables: the collected variables used to set the results
       """
        now: datetime = datetime.now()
        for variable in variables:
            variable.set_time(now)
            variable.set_value(round(random.uniform(self._min, self._max), 2))
