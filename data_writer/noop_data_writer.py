# -*- coding: utf-8 -*-
"""
Basic reader used to log the context identifier, the address of the variable and the value
"""
import logging
from typing import Dict, List

from data_collector import DataWriter, CollectedVariable


class NoopDataWriter(DataWriter):
    """
    Basic reader used to log the context identifier, the address of the variable and the value
    """
    def __init__(self, logger: logging.Logger):
        """
        Initialize the writer
        """
        self._logger: logging.Logger = logging.getLogger("NoopDataWriter")
        for handler in logger.handlers:
            self._logger.addHandler(handler)
            self._logger.setLevel(logger.level)
        self._count: int = 0
        self._count_by_context_identifiers: Dict[int, int] = {}

    def get_type(self) -> str:
        """
        See DataWriter class
        :return: the type
        """
        return self.__class__.__name__

    def write(self, values: List[CollectedVariable]) -> None:
        """
        See DataWriter class
        :param values: the collected values
        """
        for cv in values:
            self._count += 1
            if cv.get_context_identifier() in self._count_by_context_identifiers.keys():
                self._count_by_context_identifiers[cv.get_context_identifier()] = self._count_by_context_identifiers[cv.get_context_identifier()] + 1
            else:
                self._count_by_context_identifiers[cv.get_context_identifier()] = 1
            self._logger.info('Writing %s, %s=%s', cv.get_context_identifier(), cv.get_address(), cv.get_value())

    def get_count(self) -> int:
        """
        Return the number of writen items
        :return: the number of items
        """
        return self._count

    def get_count_of_context(self, context_identifier: int) -> int:
        """
        Return the number of writen items for a context
        :param context_identifier: the identifier of the context
        :return: the number of items
        """
        if context_identifier in self._count_by_context_identifiers.keys():
            return self._count_by_context_identifiers[context_identifier]
        return 0
