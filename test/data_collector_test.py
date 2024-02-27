#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Tests cases on DataCollector class
"""
import logging
import random
import time
import unittest
from typing import List
from data_collector import DataCollectionContext, Variable, DataCollector, \
    create_rotating_log, DataReader, DataWriter, DataCollectionListener
from data_reader.random_data_reader import RandomDataReader, RANDOM_DATA_READER_MIN_INTERVAL
from data_writer.csv_file_data_writer import CsvFileDataWriter
from data_writer.noop_data_writer import NoopDataWriter
from listener.noop_listener import NoopDataCollectionListener


def build_listener() -> DataCollectionListener:
    return NoopDataCollectionListener(logger)


def build_context(add_unknown: bool = False) -> DataCollectionContext:
    plan: List[Variable] = [Variable(1, str(1), 'float', 'random1', 100), Variable(2, str(2), 'float', 'random1', 200),
                            Variable(3, str(3), 'float', 'random1', 500)]
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
    results: List[DataReader] = [RandomDataReader('random1', 15.0, 18.0), RandomDataReader('random2', -1.0, 1.0)]
    return results


# noinspection PyTypeChecker
logger: logging.Logger = create_rotating_log(None, logging.DEBUG)


def build_writers() -> List[DataWriter]:
    results: List[DataWriter] = [NoopDataWriter(logger), CsvFileDataWriter(logger, '/tmp/data_collector.csv')]
    return results


class DataCollectorTest(unittest.TestCase):

    # noinspection PyArgumentList
    def setUp(self, *args, **kwargs):
        super(DataCollectorTest, self).setUp(*args, **kwargs)
        logger.info('=> Starting test: %s', self)

    def test_start(self):
        context: DataCollectionContext = build_context()
        readers: List[DataReader] = build_readers()
        collector: DataCollector = DataCollector(logger, readers, build_writers(), build_listener())
        collector.start(context)

        try:
            self.assertIsNotNone(context.get_start_date())
        finally:
            if collector is not None and context is not None:
                collector.stop(context.get_identifier())

    def test_stop(self):
        context: DataCollectionContext = build_context()
        readers: List[DataReader] = build_readers()
        collector: DataCollector = DataCollector(logger, readers, build_writers(), build_listener())
        collector.start(context)
        time.sleep(2.0)
        context = collector.stop(context.get_identifier())

        self.assertIsNotNone(context.get_start_date())
        self.assertIsNotNone(context.get_end_date())
        self.assertFalse(collector.is_context_running(context.get_identifier()))

    def test_close(self):
        context: DataCollectionContext = build_context()
        readers: List[DataReader] = build_readers()
        collector: DataCollector = DataCollector(logger, readers, build_writers(), build_listener())
        collector.start(context)
        time.sleep(2.0)
        collector.close()

        self.assertIsNotNone(context.get_start_date())
        self.assertIsNotNone(context.get_end_date())
        self.assertFalse(collector.is_context_running(context.get_identifier()))
        self.assertFalse(collector.is_running())
