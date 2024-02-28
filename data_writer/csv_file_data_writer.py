# -*- coding: utf-8 -*-
"""
Writer to a simple CSV file having the following fields:
- variable identifier,
- context identifier
- variable address,
- variable value,
- time of the collect.
"""
import logging
import os.path
import threading
from pathlib import Path
from data_collector import DataWriter, CollectedVariable


SEPARATOR: str = ','


class CsvFileDataWriter(DataWriter):
    """
    Writer to a simple CSV file
    """
    def __init__(self, logger: logging.Logger, path: str, append: bool = True):
        """
        Initialize the writer
        :param logger: the logger
        :param path: the path of the file
        :param append: the append flag
        """
        self._logger: logging.Logger = logging.getLogger("CsvFileDataWriter")
        for handler in logger.handlers:
            self._logger.addHandler(handler)
            self._logger.setLevel(logger.level)
        self._path: Path = Path(path)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        if not append:
            os.unlink(path)
        self._path.touch()
        if os.stat(path).st_size == 0:
            with self._path.open(mode="a", encoding="utf-8") as f:
                f.write('ID' + SEPARATOR + 'CONTEXT_ID' + SEPARATOR + 'ADDRESS' + SEPARATOR + 'VALUE' + SEPARATOR + 'TIME\n')
        self._lock: threading.RLock = threading.RLock()

    def get_type(self) -> str:
        """
        See DataWriter class
        :return: the type
        """
        return self.__class__.__name__

    def write(self, values: list[CollectedVariable]) -> None:
        """
        See DataWriter class
        :param values: the collected values
        """
        with self._lock:
            with self._path.open(mode="a", encoding="utf-8") as f:
                for cv in values:
                    f.write(str(cv.get_identifier()) + SEPARATOR + str(cv.get_context_identifier()) + SEPARATOR + '"' + cv.get_address() + '"' + SEPARATOR + str(cv.get_value()) + '\n')
                    self._logger.info('Writing %s, %s=%s', cv.get_context_identifier(), cv.get_address(), cv.get_value())
