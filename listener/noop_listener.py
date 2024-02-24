# -*- coding: utf-*-
# # -*- coding: utf-*-
import logging

from data_collector import DataCollectionListener, DataCollectionContext


class NoopDataCollectionListener(DataCollectionListener):
    def __init__(self, logger: logging.Logger):
        """
        Initialize the writer
        """
        self._logger: logging.Logger = logging.getLogger("NoopDataCollectionListener")
        for handler in logger.handlers:
            self._logger.addHandler(handler)
            self._logger.setLevel(logger.level)

    def before_close(self, collector) -> None:
        self._logger.info('Collector is closing...')

    def after_close(self, collector) -> None:
        self._logger.info('Collector closed')

    def before_start(self, collector, context: DataCollectionContext) -> None:
        self._logger.info('Context %s is starting...' % context.get_identifier())

    def after_start(self, collector, context: DataCollectionContext) -> None:
        self._logger.info('Context %s started' % context.get_identifier())

    def before_stop(self, collector, context: DataCollectionContext) -> None:
        self._logger.info('Context %s stopping' % context.get_identifier())

    def after_stop(self, collector, context: DataCollectionContext) -> None:
        self._logger.info('Context %s stopped' % context.get_identifier())
