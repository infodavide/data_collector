#!/usr/bin/python
# -*- coding: utf-*-#!/usr/bin/python
# # -*- coding: utf-*-
import logging
import random
import time
import unittest
from typing import List, Dict
from data_collector import DataCollectionContext, Variable, DataCollector, \
    create_rotating_log, DataReader, DataWriter, CollectedVariable
from data_readers.random_data_reader import RandomDataReader, RANDOM_DATA_READER_MIN_INTERVAL


def build_context(add_unknown: bool = False) -> DataCollectionContext:
    plan: List[Variable] = list()
    plan.append(Variable(1, str(1), 'float', 'random1', 100))
    plan.append(Variable(2, str(2), 'float', 'random1', 200))
    plan.append(Variable(3, str(3), 'float', 'random1', 500))
    for i in range(4, 15):
        plan.append(Variable(i, str(i), 'float', 'random1', RANDOM_DATA_READER_MIN_INTERVAL * random.randint(1, 5)))
    for i in range(15, 21):
        plan.append(Variable(i, str(i), 'float', 'random2', RANDOM_DATA_READER_MIN_INTERVAL * random.randint(1, 5)))
    if add_unknown:
        for i in range(21, 26):
            plan.append(Variable(i, str(i), 'float', 'unknown', RANDOM_DATA_READER_MIN_INTERVAL * random.randint(1, 5)))
    result: DataCollectionContext = DataCollectionContext(1, 200, plan)
    return result


def build_readers() -> List[DataReader]:
    results: List[DataReader] = list()
    results.append(RandomDataReader('random1', 15.0, 18.0))
    results.append(RandomDataReader('random2', -1.0, 1.0))
    return results


# noinspection PyTypeChecker
logger: logging.Logger = create_rotating_log(None, logging.DEBUG)


class NoopDataWriter(DataWriter):
    def __init__(self):
        """
        Initialize the writer
        """
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
            logger.info('Writing %s, %s=%s' % (cv.get_context_identifier(), cv.get_address(), cv.get_value()))

    def get_count(self) -> int:
        return self._count

    def get_count_of_context(self, context_identifier: int) -> int:
        if context_identifier in self._count_by_context_identifiers.keys():
            return self._count_by_context_identifiers[context_identifier]
        return 0


def build_writers() -> List[DataWriter]:
    results: List[DataWriter] = list()
    results.append(NoopDataWriter())
    return results


class DataCollectorTest(unittest.TestCase):

    # noinspection PyArgumentList
    def setUp(self, *args, **kwargs):
        super(DataCollectorTest, self).setUp(*args, **kwargs)
        logger.info('=> Starting test: %s' % self)

    def test_start(self):
        context: DataCollectionContext = build_context()
        readers: List[DataReader] = build_readers()
        collector: DataCollector = DataCollector(logger, readers, build_writers())
        collector.start(context)

        try:
            self.assertIsNotNone(context.get_start_date())
        finally:
            if collector is not None and context is not None:
                collector.stop(context.get_identifier())

    def test_stop(self):
        context: DataCollectionContext = build_context()
        readers: List[DataReader] = build_readers()
        collector: DataCollector = DataCollector(logger, readers, build_writers())
        collector.start(context)
        time.sleep(2.0)
        context = collector.stop(context.get_identifier())

        self.assertIsNotNone(context.get_start_date())
        self.assertIsNotNone(context.get_end_date())
        self.assertFalse(collector.is_context_running(context.get_identifier()))

    def test_close(self):
        context: DataCollectionContext = build_context()
        readers: List[DataReader] = build_readers()
        collector: DataCollector = DataCollector(logger, readers, build_writers())
        collector.start(context)
        time.sleep(2.0)
        collector.close()

        self.assertIsNotNone(context.get_start_date())
        self.assertIsNotNone(context.get_end_date())
        self.assertFalse(collector.is_context_running(context.get_identifier()))
        self.assertFalse(collector.is_running())
