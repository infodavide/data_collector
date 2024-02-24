# -*- coding: utf-*-
# # -*- coding: utf-*-
import logging
from typing import Dict, List

from data_collector import DataWriter, CollectedVariable


class NoopDataWriter(DataWriter):
    def __init__(self, logger: logging.Logger):
        """
        Initialize the writer
        """
        self._logger: logging.Logger = logging.getLogger("NoopDataWriter")
        for handler in logger.handlers:
            self._logger.addHandler(handler)
            self._logger.setLevel(logger.level)
        self._count: int = 0
        self._count_by_context_identifiers: Dict[int, int] = dict()

    def get_type(self) -> str:
        return self.__class__.__name__

    def write(self, values: List[CollectedVariable]) -> None:
        for cv in values:
            self._count += 1
            if cv.get_context_identifier() in self._count_by_context_identifiers.keys():
                self._count_by_context_identifiers[cv.get_context_identifier()] = self._count_by_context_identifiers[cv.get_context_identifier()] + 1
            else:
                self._count_by_context_identifiers[cv.get_context_identifier()] = 1
            self._logger.info('Writing %s, %s=%s' % (cv.get_context_identifier(), cv.get_address(), cv.get_value()))

    def get_count(self) -> int:
        return self._count

    def get_count_of_context(self, context_identifier: int) -> int:
        if context_identifier in self._count_by_context_identifiers.keys():
            return self._count_by_context_identifiers[context_identifier]
        return 0
