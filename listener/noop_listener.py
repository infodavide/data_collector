# -*- coding: utf-8 -*-
"""
Listener used to log events
"""
import logging
from typing import Any
from data_collector import DataCollectionListener, DataCollectionContext


class NoopDataCollectionListener(DataCollectionListener):
    """
    Listener used to log events
    """
    def __init__(self, logger: logging.Logger):
        """
        Initialize the writer
        :param logger: the logger
        """
        self._logger: logging.Logger = logging.getLogger("NoopDataCollectionListener")
        for handler in logger.handlers:
            self._logger.addHandler(handler)
            self._logger.setLevel(logger.level)

    def before_close(self, collector: Any) -> None:
        """
        See DataCollectionListener class
        :param collector: the collector
        """
        self._logger.info('Collector is closing...')

    def after_close(self, collector: Any) -> None:
        """
        See DataCollectionListener class
        :param collector: the collector
        """
        self._logger.info('Collector closed')

    def before_start(self, collector: Any, context: DataCollectionContext) -> None:
        """
        See DataCollectionListener class
        :param collector: the collector
        :param context: the context
        """
        self._logger.info('Context %s is starting...', context.get_identifier())

    def after_start(self, collector: Any, context: DataCollectionContext) -> None:
        """
        See DataCollectionListener class
        :param collector: the collector
        :param context: the context
        """
        self._logger.info('Context %s started', context.get_identifier())

    def before_stop(self, collector: Any, context: DataCollectionContext) -> None:
        """
        See DataCollectionListener class
        :param collector: the collector
        :param context: the context
        """
        self._logger.info('Context %s stopping', context.get_identifier())

    def after_stop(self, collector: Any, context: DataCollectionContext) -> None:
        """
        See DataCollectionListener class
        :param collector: the collector
        :param context: the context
        """
        self._logger.info('Context %s stopped', context.get_identifier())
