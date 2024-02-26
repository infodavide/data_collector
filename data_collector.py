#!/usr/bin/python
# -*- coding: utf-*-
# # -*- coding: utf-*-
import atexit
import logging
import math
import os
import pathlib
import sched
import signal
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from logging.handlers import RotatingFileHandler
from string import Template
from typing import Dict, List, Any, Set
from multiprocessing import Pool


def create_rotating_log(path: str, level: int) -> logging.Logger:
    """
    Create the logger with file rotation
    :param path: the path of the main log file
    :param level: the log level
    :return: the logger
    """
    result: logging.Logger = logging.getLogger("DataCollector")
    # noinspection Spellchecker
    formatter: logging.Formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler: logging.Handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    result.addHandler(console_handler)
    if path is not None:
        path_obj: pathlib.Path = pathlib.Path(path)
        if not os.path.exists(path_obj.parent.absolute()):
            os.makedirs(path_obj.parent.absolute())
        if os.path.exists(path):
            open(path, 'w').close()
        else:
            path_obj.touch()

        file_handler: logging.Handler = RotatingFileHandler(path, maxBytes=1024 * 1024 * 5, backupCount=5)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        result.addHandler(file_handler)
    result.setLevel(level)
    return result


class Variable(object):
    def __init__(self, identifier: int, address: str, value_type: str, reader_type: str, interval: int):
        """
        Initialize the variable
        :param identifier: the identifier of the variable
        :param address: the address of the variable
        :param value_type: the type of the variable
        :param reader_type: the type of the reader to use to collect the associated values
        :param interval: the interval in milliseconds
        """
        self._identifier: int = identifier
        self._address: str = address
        self._value_type: str = value_type
        self._reader_type: str = reader_type
        self._interval: int = interval

    def get_identifier(self) -> int:
        """
        Return the identifier
        :return: the identifier
        """
        return self._identifier

    def get_address(self) -> str:
        """
        Return the address
        :return: the address
        """
        return self._address

    def get_value_type(self) -> str:
        """
        Return the type of the value
        :return: the type
        """
        return self._value_type

    def get_reader_type(self) -> str:
        """
        Return the type of the reader
        :return: the type
        """
        return self._reader_type

    def get_interval(self) -> int:
        """
        Return the interval in milliseconds
        :return: the interval in milliseconds
        """
        return self._interval

    def set_identifier(self, value: int) -> None:
        """
        Set the identifier
        :param value: the identifier
        """
        if value is None or value <= 0:
            raise ValueError("An error occurred", "Invalid value", value)
        self._identifier = value

    def set_address(self, value: str) -> None:
        """
        Set the address
        :param value: the address
        """
        if value is None or len(value) == 0:
            raise ValueError("An error occurred", "Invalid value", value)
        self._address = value

    def set_value_type(self, value: str) -> None:
        """
        Set the type of the value
        :param value: the type
        """
        if value is None or len(value) == 0:
            raise ValueError("An error occurred", "Invalid value", value)
        self._value_type = value

    def set_reader_type(self, value: str) -> None:
        """
        Set the type of the reader
        :param value: the type
        """
        if value is None or len(value) == 0:
            raise ValueError("An error occurred", "Invalid value", value)
        self._reader_type = value

    def set_interval(self, value: int) -> None:
        """
        Set the interval in milliseconds
        :param value: the interval in milliseconds
        """
        if value is None:
            raise ValueError("An error occurred", "Invalid value", value)
        self._interval = value


class CollectedVariable(Variable):
    def __init__(self, identifier: int, context_identifier: int, address: str, value_type: str, reader_type: str, interval: int, value, read_time: datetime):
        """
        Initialize the value
        :param identifier: the identifier of the variable
        :param context_identifier: the identifier of the context
        :param address: the address of the variable
        :param value_type: the type of the variable
        :param reader_type: the type of the reader to use to collect the associated values
        :param interval: the interval in milliseconds
        :param value: the value
        :param read_time: the time of the collect
        """
        super(CollectedVariable, self).__init__(identifier, address, value_type, reader_type, interval)
        self._context_identifier = context_identifier
        self._value = value
        self._time: datetime = read_time

    def get_context_identifier(self) -> int:
        """
        Return the identifier of the associated context
        :return: the identifier
        """
        return self._context_identifier

    def get_value(self) -> Any:
        """
        Return the value
        :return: the value
        """
        return self._value

    def get_time(self) -> datetime:
        """
        Return the time of the collect of the value
        :return: the time
        """
        return self._time

    def set_value(self, value: Any) -> None:
        """
        Set the value
        :param value: the value
        """
        self._value = value

    def set_time(self, value: datetime) -> None:
        """
        Set the time of the collect of the value
        :param value: the time
        """
        self._time = value

    def __str__(self) -> str:
        """
        Return the textual view
        :return: the text
        """
        t = Template('[DataValue value=$v, time=$t]')
        return t.substitute(v=self._value, t=self._time.timestamp())


class IntervalLimitPolicy(Enum):
    IGNORE = 1  # Ignore the variable if limit is reached
    USE_LIMIT = 2  # Use the default interval for the variable if limit is reached
    ERROR = 3  # Raise an error if limit is reached


class DataCollectionContext(object):
    # noinspection PyTypeChecker
    def __init__(self, identifier: int, interval: int, plan: List[Variable]):
        """
        Initialize the context
        :param identifier: the identifier
        :param interval: the default interval in milliseconds
        :param plan: the variables to collect
        """
        self._identifier: int = identifier
        self._interval: int = interval
        self._min_interval: int = 100
        self._plan: List[Variable] = plan
        self._start_date: datetime = None
        self._end_date: datetime = None
        self._life_duration: int = 0
        self._retention: int = 0

    def get_identifier(self) -> int:
        """
        Return the identifier
        :return: the identifier
        """
        return self._identifier

    def get_start_date(self) -> datetime:
        """
        Return the start date
        :return: the datetime
        """
        return self._start_date

    def get_end_date(self) -> datetime:
        """
        Return the end date
        :return: the datetime
        """
        return self._end_date

    def get_interval(self) -> int:
        """
        Return the default interval in milliseconds
        :return: the interval in milliseconds
        """
        return self._interval

    def get_min_interval(self) -> int:
        """
        Return the minimum interval in milliseconds
        :return: the interval in milliseconds
        """
        return self._min_interval

    def get_life_duration(self) -> int:
        """
        Return the maximum life duration in seconds
        :return: the maximum life duration in seconds
        """
        return self._life_duration

    def get_retention(self) -> int:
        """
        Return the maximum retention of the data in hours
        :return: the maximum retention of the data in hours
        """
        return self._retention

    def get_plan(self) -> List[Variable]:
        """
        Return the data collection plan describing the variables to collect.
        :return: the data collection plan
        """
        return self._plan

    def set_identifier(self, value: int) -> None:
        """
        Set the identifier
        :param value: the identifier
        """
        if value is None or value <= 0:
            raise ValueError("An error occurred", "Invalid value", value)
        self._identifier = value

    def set_start_date(self, value: datetime) -> None:
        """
        Set the start date
        :param value: the datetime
        """
        self._start_date = value

    def set_end_date(self, value: datetime) -> None:
        """
        Set the end date
        :param value: the datetime
        """
        self._end_date = value

    def set_interval(self, value: int) -> None:
        """
        Set the default interval in milliseconds
        :param value: the interval in milliseconds
        """
        if value is None or value <= 0:
            raise ValueError("An error occurred", "Invalid value", value)
        self._interval = value

    def set_min_interval(self, value: int) -> None:
        """
        Set the minimum interval in milliseconds
        :param value: the interval in milliseconds
        """
        if value is None or value <= 0:
            raise ValueError("An error occurred", "Invalid value", value)
        self._min_interval = value

    def set_life_duration(self, value: int) -> None:
        """
        Set the maximum life duration in seconds
        :param value: the maximum life duration in seconds
        """
        if value is None or value <= 0:
            raise ValueError("An error occurred", "Invalid value", value)
        self._life_duration = value

    def set_retention(self, value: int) -> None:
        """
        Set the maximum retention of the data in hours
        :param value: the maximum retention of the data in hours
        """
        if value is None or value <= 0:
            raise ValueError("An error occurred", "Invalid value", value)
        self._retention = value

    def set_plan(self, value: List[Variable]) -> None:
        """
        Set the data collection plan
        :param value: the data collection plan
        """
        if value is None:
            raise ValueError("An error occurred", "Invalid value", value)
        self._plan = value


class _DataCollectionContextWrapper(object):
    # noinspection PyTypeChecker
    def __init__(self, context: DataCollectionContext):
        """
        Initialize the wrapper
        :param context: the context
        """
        self._context: DataCollectionContext = context
        self._active: bool = True
        self._life_time: datetime = None

    def get_context(self) -> DataCollectionContext:
        """
        Return the context
        :return: the context
        """
        return self._context

    def is_active(self) -> bool:
        """
        Return true if the context is active
        :return: the boolean
        """
        return self._active

    def get_life_time(self) -> datetime:
        """
        Return the future end time
        :return: the datetime
        """
        return self._life_time

    def set_active(self, value: bool) -> None:
        """
        Set the context as active or not
        :param value: the boolean
        """
        self._active = value

    def set_life_time(self, value: datetime) -> None:
        """
        Set the future end time
        :param value: the datetime
        """
        self._life_time = value


MINIMUM_FREQUENCY_VALUE: int = 20


class _DataCollectorScheduledTask(object):
    # noinspection PyTypeChecker
    def __init__(self, logger: logging.Logger, collector):
        """
        Initialize the task
        :param logger: the parent logger
        :param collector: the collector
        """
        self._logger: logging.Logger = logging.getLogger("DataCollectorScheduledTask")
        for handler in logger.handlers:
            self._logger.addHandler(handler)
            self._logger.setLevel(logger.level)
        self._collector: DataCollector = collector
        self._active: bool = False
        self._lock: threading.RLock = threading.RLock()
        self._future: sched.Event = None
        self._variables_by_step_and_reader: Dict[int, Dict[str, List[CollectedVariable]]] = dict()
        self._interval: int = 0
        self._max_step_value: int = 0
        self._step: int = 0

    def is_active(self) -> bool:
        """
        Return true if the context is active
        :return: the boolean
        """
        return self._active

    def set_active(self, value: bool) -> None:
        """
        Set the context as active or not
        :param value: the boolean
        """
        if self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug('Setting active: %s' % value)
        previous_state: bool = self._active
        self._active = value
        scheduler: sched.scheduler = self._collector.get_scheduler()
        if scheduler is None:
            self._logger.debug('Scheduler is no more available...')
            return
        if not previous_state and value:
            self._logger.debug('Scheduling...')
            scheduler.enter(0, 1, self.run)
            scheduler.run(blocking=False)
        if previous_state and not value and self._future is not None:
            self._logger.debug('Cancelling scheduled task...')
            scheduler.cancel(self._future)
            self._future = None

    # noinspection PyTypeChecker
    def run(self) -> None:
        """
        Do the collect
        """
        scheduler: sched.scheduler = self._collector.get_scheduler()
        if not self._active:
            self._logger.debug('Task is no more active')
            if self._future is not None:
                self._logger.debug('Cancelling scheduled task...')
                scheduler.cancel(self._future)
                self._future = None
            return
        start_time: float = time.time()
        self._logger.debug('Running...')
        with self._lock:
            step: int = self._step
            interval: int = self._interval
            if scheduler is None:
                self._logger.debug('Scheduler is no more available...')
                self._step = 0
                return
            scheduler.enter(interval, 1, self.run)
            self._logger.debug('Getting variables at step: %s' % step)
            variables_by_reader: Dict[str, List[CollectedVariable]] = self._get_variables_at_step(step)
            step += interval
            if step >= self._max_step_value:
                step = 0
            self._step = step
            if not self._active:
                self._logger.debug('Task is no more active')
                if self._future is not None:
                    self._logger.debug('Cancelling scheduled task...')
                    scheduler.cancel(self._future)
                    self._future = None
                return
            for reader_type in variables_by_reader.keys():
                if self._logger.isEnabledFor(logging.DEBUG):
                    self._logger.debug('Creating a new task to read variables associated to reader of type: %s' % reader_type)
                reader: DataReader = self._collector.get_reader(reader_type)
                reader_task: _DataReaderTask = _DataReaderTask(self._logger, self._collector, reader, variables_by_reader[reader_type])
                self._collector.get_thread_pool().apply_async(reader_task.run())
            if self._logger.isEnabledFor(logging.DEBUG):
                self._logger.debug('Completed in %sms' % int((time.time() - start_time) * 1000))

    # noinspection PyTypeChecker
    def append(self, context_wrapper: _DataCollectionContextWrapper) -> int:
        """
        Append the variables of the given wrapper and context
        :param context_wrapper: the wrapped context
        :return: the number of appended variables
        """
        self._logger.debug('Appending...')
        result: int = 0
        with self._lock:
            context: DataCollectionContext = context_wrapper.get_context()
            computed_interval: int = self._interval
            computed_max_step_value: int = self._max_step_value
            variables: List[CollectedVariable] = list()
            if self._logger.isEnabledFor(logging.DEBUG):
                self._logger.debug('Variable(s) before appending: %s, appending %s variables' % (self._len_of_variables_by_step_and_reader(), len(context.get_plan())))
            for variable in context.get_plan():
                cv: CollectedVariable = collected_variable_of(context.get_identifier(), variable, None, None)
                if cv.get_interval() is None or cv.get_interval() <= 0:
                    if self._logger.isEnabledFor(logging.DEBUG):
                        self._logger.debug('Variable: %s has no frequency, using the default one: %sms' % (cv.get_address(), context.get_interval()))
                    cv.set_interval(context.get_interval())
                if self._exists(cv):
                    variables.append(cv)
                    continue
                reader: DataReader = self._collector.get_reader(cv.get_reader_type())
                if reader is None:
                    raise ValueError("An error occurred", "No reader found for variable", cv.get_reader_type())
                value: int = self._compute_min_interval(reader.get_min_interval(), computed_interval, cv)
                if value <= 0:
                    continue
                # We need to compute the interval to take into account the variable having an interval not already handled
                computed_interval = value
                variables.append(cv)
                computed_max_step_value = max(computed_max_step_value, cv.get_interval())
            result = len(variables)
            if result > 0:
                # We need to compute the step to take into account the variable having an interval not already handled
                self._max_step_value = computed_max_step_value
                self._compute_new_step(computed_interval)
                step: int = self._step
                if step == 0:
                    for cv in variables:
                        self._append(0, cv)
                else:
                    for cv in variables:
                        self._append((step + cv.get_interval()) % self._max_step_value, cv)
        if self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug('Variable(s) after appending: %s' % self._len_of_variables_by_step_and_reader())
        return result

    # noinspection PyTypeChecker
    def _append(self, step: int, cv: CollectedVariable):
        if step not in self._variables_by_step_and_reader:
            self._variables_by_step_and_reader[step] = dict()
        if cv.get_reader_type() in self._variables_by_step_and_reader[step]:
            variables = self._variables_by_step_and_reader[step][cv.get_reader_type()]
        else:
            variables = list()
            self._variables_by_step_and_reader[step][cv.get_reader_type()] = variables
        if self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug('Appending variable: %s (%s)' % (cv.get_address(), cv.get_identifier()))
        variables.append(cv)

    def _compute_min_interval(self, reader_min_interval: int, min_interval, cv: CollectedVariable) -> int:
        if min_interval == cv.get_interval:
            return min_interval
        if self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug('Computing minimum interval with previous: %sms and current interval: %sms...' % (min_interval, cv.get_interval()))
        if min_interval <= 0:
            result = cv.get_interval()
        else:
            result = math.gcd(min_interval, cv.get_interval())
        if result < reader_min_interval:
            message = 'Interval limit %sms reached (%sms) by variable: %s (%s)' % (reader_min_interval, cv.get_interval(), cv.get_address(), cv.get_identifier())
        elif result < MINIMUM_FREQUENCY_VALUE:
            message = 'Interval limit %sms reached (%sms) by variable: %s (%s)' % (MINIMUM_FREQUENCY_VALUE, cv.get_interval(), cv.get_address(), cv.get_identifier())
        else:
            return result
        self._logger.warning(message)
        if IntervalLimitPolicy.ERROR == self._collector.get_interval_limit_policy():
            raise ValueError("An error occurred", message, cv.get_interval)
        elif IntervalLimitPolicy.USE_LIMIT == self._collector.get_interval_limit_policy():
            result = math.gcd(min_interval, reader_min_interval)
            self._logger.info('Using interval: %s' % result)
            return result
        self._logger.warning('Ignoring variable: %s (%s)' % (cv.get_address(), cv.get_identifier()))
        return result

    def _compute_new_step(self, interval: int) -> None:
        if interval == self._interval:
            return
        if self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug('Computing the new step using new interval: %sms (previous value: %sms, previous step: %sms)' % (interval, self._interval, self._step))
        delay: int = 0
        if self._future is not None:
            delay = int((time.time() - self._future.time) * 1000)
        diff: int = interval - delay
        if self._step > 0:
            self._step = self._step + self._interval - interval
        self._interval = interval
        if self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug('Interval: %sms, maximum step value: %sms, new step value: %sms' % (self._interval, self._max_step_value, self._step))
        if self._future is not None and diff > interval:
            scheduler: sched.scheduler = self._collector.get_scheduler()
            if scheduler is None:
                self._logger.debug('Scheduler is no more available...')
                return
            scheduler.cancel(self._future)
            scheduled_interval: int = interval - diff
            if self._logger.isEnabledFor(logging.DEBUG):
                self._logger.debug('Scheduling next execution after %sms' % scheduled_interval)
            self._future = scheduler.enter(scheduled_interval, 1, self.run)

    def _exists(self, cv: CollectedVariable) -> bool:
        for variables_by_reader in self._variables_by_step_and_reader.values():
            if cv.get_reader_type() in variables_by_reader:
                for v in variables_by_reader[cv.get_reader_type()]:
                    if v.get_address() == cv.get_address():
                        return True
        return False

    def _get_variables_at_step(self, step: int) -> Dict[str, List[CollectedVariable]]:
        with self._lock:
            result: Dict[str, List[CollectedVariable]] = dict()
            contexts_to_stop: Set[int] = set()
            if step not in self._variables_by_step_and_reader:
                return result
            variables_by_reader: Dict[str, List[CollectedVariable]] = self._variables_by_step_and_reader[step]
            del self._variables_by_step_and_reader[step]
            if self._logger.isEnabledFor(logging.DEBUG):
                self._logger.debug('Preprocessing %s variables(s) for step: %s before reading' % (self._len_of_variables_by_step(step), step))
            now: datetime = datetime.now()
            for reader_type in variables_by_reader.keys():
                if reader_type in result:
                    variables = result[reader_type]
                else:
                    variables = list()
                    result[reader_type] = variables
                for cv in variables_by_reader[reader_type]:
                    context_wrapper: _DataCollectionContextWrapper = self._collector.get_context(cv.get_context_identifier())
                    if context_wrapper is None:
                        if self._logger.isEnabledFor(logging.DEBUG):
                            self._logger.debug('Context wrapper: %s of variable: %s is not valid' % (cv.get_context_identifier(), cv.get_address()))
                        continue
                    if context_wrapper.get_context() is None:
                        if self._logger.isEnabledFor(logging.DEBUG):
                            self._logger.debug('Context: %s of variable: %s is not valid' % (cv.get_context_identifier(), cv.get_address()))
                        continue
                    if context_wrapper.get_context().get_end_date() is not None:
                        if self._logger.isEnabledFor(logging.DEBUG):
                            self._logger.debug('Context: %s of variable: %s has been stopped' % (cv.get_context_identifier(), cv.get_address()))
                        continue
                    if context_wrapper.get_life_time() <= now:
                        if self._logger.isEnabledFor(logging.DEBUG):
                            self._logger.debug('Context: %s of variable: %s has reached its maximum life date' % (cv.get_context_identifier(), cv.get_address()))
                        contexts_to_stop.add(cv.get_context_identifier())
                        continue
                    next_step: int = (step + cv.get_interval()) % self._max_step_value
                    if next_step not in self._variables_by_step_and_reader:
                        self._variables_by_step_and_reader[next_step] = dict()
                    if reader_type not in self._variables_by_step_and_reader[next_step]:
                        self._variables_by_step_and_reader[next_step][reader_type] = list()
                    variables.append(cv)
                    self._variables_by_step_and_reader[next_step][reader_type].append(cv)
                    if self._logger.isEnabledFor(logging.DEBUG):
                        self._logger.debug('Ready to read variable: %s of context: %s, next step will be: %sms' % (cv.get_address(), cv.get_context_identifier(), next_step))
            for context_identifier in contexts_to_stop:
                self._collector.stop(context_identifier)
            if self._logger.isEnabledFor(logging.DEBUG):
                count: int = 0
                for reader_type in result.keys():
                    count += len(result[reader_type])
                self._logger.debug("%s variable(s) to read for step: %s" % (count, step))
            return result

    def _len_of_variables_by_step(self, step: int):
        result: int = 0
        if step in self._variables_by_step_and_reader:
            for variables in self._variables_by_step_and_reader[step].values():
                result += len(variables)
        return result

    def _len_of_variables_by_step_and_reader(self):
        result: int = 0
        for variables_by_reader in self._variables_by_step_and_reader.values():
            for variables in variables_by_reader.values():
                result += len(variables)
        return result

    def remove(self, context_wrapper: _DataCollectionContextWrapper) -> None:
        """
        Remove the variables of the given wrapper and context
        :param context_wrapper: the wrapped context
        """
        with self._lock:
            context: DataCollectionContext = context_wrapper.get_context()
            if self._logger.isEnabledFor(logging.DEBUG):
                self._logger.debug('Variable(s) before removal: %s' % (self._len_of_variables_by_step_and_reader()))
            computed_interval: int = -1
            computed_max_step_value = self._max_step_value
            for variables_by_reader in self._variables_by_step_and_reader.values():
                for reader_type in variables_by_reader.keys():
                    reader: DataReader = self._collector.get_reader(reader_type)
                    filtered_variables = list(filter(lambda cv: cv.get_context_identifier() != context.get_identifier(), variables_by_reader[reader_type]))
                    variables_by_reader[reader_type] = filtered_variables

                    for variable in filtered_variables:
                        computed_interval = self._compute_min_interval(reader.get_min_interval(), computed_interval, variable)
                        computed_max_step_value = max(computed_max_step_value, variable.get_interval())
            if self._len_of_variables_by_step_and_reader() == 0:
                self._interval = 0
                self._max_step_value = 0
                self._active = False
                self._step = 0
            else:
                step_to_remove =  list()
                reader_types_to_remove = list()
                for step in self._variables_by_step_and_reader.keys():
                    variables_by_reader = self._variables_by_step_and_reader[step]
                    reader_types_to_remove.clear()
                    if len(variables_by_reader) == 0:
                        step_to_remove.append(step)
                        continue
                    for reader_type in variables_by_reader.keys():
                        if len(variables_by_reader[reader_type]) == 0:
                            reader_types_to_remove.append(reader_type)
                    for reader_type in reader_types_to_remove:
                        del variables_by_reader[reader_type]
                for step in step_to_remove:
                    del self._variables_by_step_and_reader[step]
                # We need to compute the interval to take into account the removal of variables
                # The step remains unchanged for existing variables
                self._max_step_value = computed_max_step_value
                self._interval = computed_interval
            if self._logger.isEnabledFor(logging.DEBUG):
                self._logger.debug('Variable(s) after removal: %s' % (self._len_of_variables_by_step_and_reader()))


class DataReader(ABC):
    @abstractmethod
    def get_type(self) -> str:
        """
        Return the type of the reader
        :return: the type of the reader
        """
        pass

    @abstractmethod
    def get_min_interval(self) -> int:
        """
        Return the minimum interval in milliseconds
        :return: the minimum interval in milliseconds
        """
        pass

    @abstractmethod
    def read(self, variables: List[CollectedVariable]) -> None:
        """
        Read the values
        :param variables: the variables to read
        """
        pass


class _DataReaderTask(object):
    def __init__(self, logger: logging.Logger, collector, reader: DataReader, variables: List[CollectedVariable]):
        """
        Initialize the task
        :param logger: the parent logger
        :param collector: the collector
        :param reader: the reader
        :param variables: the variables to read
        """
        self._logger: logging.Logger = logging.getLogger("DataReaderTask")
        for handler in logger.handlers:
            self._logger.addHandler(handler)
            self._logger.setLevel(logger.level)
        self._collector: DataCollector = collector
        self._reader: DataReader = reader
        self._variables: List[CollectedVariable] = variables

    # noinspection PyTypeChecker
    def run(self) -> None:
        """
        Do the collect
        """
        start_time: float = time.time()
        if self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug('Starting read of %s variables for reader: %s...' % (len(self._variables), self._reader.get_type()))
        self._reader.read(self._variables)
        if self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug('Read completed in %sms' % int((time.time() - start_time) * 1000))
        for writer in self._collector.get_writers():
            writer_task: _DataWriterTask = _DataWriterTask(self._logger, self._collector, writer, self._variables)
            self._collector.get_thread_pool().apply_async(writer_task.run())
        if self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug('Completed in %sms' % int((time.time() - start_time) * 1000))


class DataWriter(ABC):
    @abstractmethod
    def get_type(self) -> str:
        """
        Return the type of the writer
        :return: the type of the writer
        """
        pass

    @abstractmethod
    def write(self, values: List[CollectedVariable]) -> None:
        """
        write the values
        :param values: the values associated to their addresses
        """
        pass


class _DataWriterTask(object):
    def __init__(self, logger: logging.Logger, collector, writer: DataWriter, variables: List[CollectedVariable]):
        """
        Initialize the task
        :param logger: the parent logger
        :param collector: the collector
        :param writer: the writer
        :param variables: the variables to read
        """
        self._logger: logging.Logger = logging.getLogger("DataWriterTask")
        for handler in logger.handlers:
            self._logger.addHandler(handler)
            self._logger.setLevel(logger.level)
        self._collector: DataCollector = collector
        self._writer: DataWriter = writer
        self._variables: List[CollectedVariable] = variables

    def run(self) -> None:
        """
        Do the writes
        """
        start_time: float = time.time()
        if self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug('Starting write of %s variables for writer: %s...' % (len(self._variables), self._writer.get_type()))
        self._writer.write(self._variables)
        if self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug('Completed in %sms' % int((time.time() - start_time) * 1000))


def collected_variable_of(context_identifier: int, source: Variable, value, read_time: datetime) -> CollectedVariable:
    """
    Initialize the collected variable object using the given variable and context
    :param context_identifier: the identifier of the context
    :param source: the variable
    :param value: the value
    :param read_time: the time of the collect
    :return: the collected variable object
    """
    return CollectedVariable(source.get_identifier(), context_identifier, source.get_address(), source.get_value_type(), source.get_reader_type(), source.get_interval(), value, read_time)


def _thread_initializer():
    threading.current_thread().name = threading.current_thread().name.replace('Thread', 'DataCollector-Thread')


class DataCollectionListener(ABC):
    @abstractmethod
    def before_close(self, collector) -> None:
        """
        Notify closure of the collector
        :param collector: the collector
        """
        pass

    @abstractmethod
    def after_close(self, collector) -> None:
        """
        Notify closure of the collector
        :param collector: the collector
        """
        pass

    @abstractmethod
    def before_start(self, collector, context: DataCollectionContext) -> None:
        """
        Notify the start of the context
        :param collector: the collector
        :param context: the context
        """
        pass

    @abstractmethod
    def after_start(self, collector, context: DataCollectionContext) -> None:
        """
        Notify the start of the context
        :param collector: the collector
        :param context: the context
        """
        pass

    @abstractmethod
    def before_stop(self, collector, context: DataCollectionContext) -> None:
        """
        Notify the stop of the context
        :param collector: the collector
        :param context: the context
        """
        pass

    @abstractmethod
    def after_stop(self, collector, context: DataCollectionContext) -> None:
        """
        Notify the stop of the context
        :param collector: the collector
        :param context: the context
        """
        pass


class DataCollector(object):
    # noinspection PyTypeChecker
    def __init__(self, logger: logging.Logger, readers: List[DataReader], writers: List[DataWriter], listener: DataCollectionListener = None):
        """
        Initialize the context
        :param logger: the logger
        """
        self._logger: logging.Logger = logger
        self._logger.info('Initializing %s', self.__class__.__name__)
        self._readers: Dict[str, DataReader] = dict()
        for reader in readers:
            self._readers[reader.get_type()] = reader
        self._writers: List[DataWriter] = writers
        self._listener: DataCollectionListener = listener
        self._interval_limit_policy: IntervalLimitPolicy = IntervalLimitPolicy.IGNORE
        self._life_duration: int = 0
        self._contexts: Dict[int, _DataCollectionContextWrapper] = dict()
        self._lock: threading.RLock = threading.RLock()
        self._thread_pool: Pool = Pool(initializer=_thread_initializer, maxtasksperchild=5)
        self._scheduler: sched.scheduler = sched.scheduler()
        self._task: _DataCollectorScheduledTask = _DataCollectorScheduledTask(self._logger, self)
        atexit.register(self.close)
        signal.signal(signal.SIGINT, self._close)

    def __del__(self):
        """
        Close the data collector
        """
        self._close()

    # noinspection PyTypeChecker
    def start(self, context: DataCollectionContext) -> DataCollectionContext:
        """
        Start the data collection of the variables of the given context
        :param context: the context
        """
        if context is None:
            raise ValueError("An error occurred", "Context is not valid", context)
        if context.get_identifier() is None:
            raise ValueError("An error occurred", "Identifier of the context is not valid", context)
        if not self.is_running():
            raise ValueError("Data collector has been closed")
        with self._lock:
            if context.get_identifier() in self._contexts.keys():
                self._logger.warning('Context already started')
                return self._contexts[context.get_identifier()]
            self._logger.info('Starting context: %s' % context.get_identifier())
            wrapper: _DataCollectionContextWrapper = _DataCollectionContextWrapper(context)
            context.set_start_date(datetime.now())
            context.set_end_date(None)
            wrapper.set_active(True)
            if context.get_life_duration() <= 0:
                if self._life_duration > 0:
                    wrapper.set_life_time(context.get_start_date() + timedelta(0, self._life_duration))
                else:
                    wrapper.set_life_time(context.get_start_date() + timedelta(1, 0))
            else:
                wrapper.set_life_time(context.get_start_date() + timedelta(0, context.get_life_duration()))
            self._logger.warning('Context life time: %s' % wrapper.get_life_time())
            self._contexts[context.get_identifier()] = wrapper
            if self._listener is not None:
                self._listener.before_start(self, context)
            appended: int = self._task.append(wrapper)
            if self._logger.isEnabledFor(logging.DEBUG):
                self._logger.debug('Appended variables: %s' % appended)
            if appended > 0 and (self._task is None or not self._task.is_active()):
                if self._task is None:
                    self._logger.debug('Creating collector task...')
                    self._task = _DataCollectorScheduledTask(self._logger, self)
                self._logger.debug('Setting collector task as active')
                self._task.set_active(True)
            if self._listener is not None:
                self._listener.after_start(self, context)

    # noinspection PyTypeChecker
    def stop(self, context_identifier: int, warn: bool = True) -> DataCollectionContext:
        """
        Stop the data collection of the variables of the context associated to the given identifier
        :param context_identifier: the identifier of the context
        :param warn: the flag specifying if a warning must be logged if context is not running
        """
        if context_identifier is None:
            raise ValueError("An error occurred", "Identifier of the context is not valid", context_identifier)
        if not self.is_running():
            raise ValueError("Data collector has been closed")
        with self._lock:
            if context_identifier not in self._contexts.keys():
                if warn:
                    self._logger.warning('Context not started %s' % context_identifier)
                return None
            self._logger.info('Stopping context: %s' % context_identifier)
            wrapper: _DataCollectionContextWrapper = self._contexts[context_identifier]
            context: DataCollectionContext = wrapper.get_context()
            if self._listener is not None:
                self._listener.before_stop(self, context)
            context.set_end_date(datetime.now())
            self._task.remove(wrapper)
            del self._contexts[context_identifier]
            if len(self._contexts) == 0 and self._task.is_active():
                self._logger.debug('Setting collector task as inactive')
                self._task.set_active(False)
            if self._listener is not None:
                self._listener.after_stop(self, context)
            return context

    # noinspection PyTypeChecker
    def pause(self, context_identifier: int, warn: bool = True) -> DataCollectionContext:
        """
        Pause the data collection of the variables of the context associated to the given identifier
        :param context_identifier: the identifier of the context
        :param warn: the flag specifying if a warning must be logged if context is not running
        """
        if context_identifier is None:
            raise ValueError("An error occurred", "Identifier of the context is not valid", context_identifier)
        if not self.is_running():
            raise ValueError("Data collector has been closed")
        with self._lock:
            if context_identifier not in self._contexts.keys():
                if warn:
                    self._logger.warning('Context has not been started %s' % context_identifier)
                return None
            wrapper: _DataCollectionContextWrapper = self._contexts[context_identifier]
            if not wrapper.is_active():
                if warn:
                    self._logger.warning('Context is not running')
                return wrapper.get_context()
            if wrapper.get_context().get_end_date() is not None:
                if warn:
                    self._logger.warning('Context has been stopped')
                return wrapper.get_context()
            self._logger.info('Pausing context: %s' % context_identifier)
            wrapper.set_active(False)
            return wrapper.get_context()

    # noinspection PyTypeChecker
    def resume(self, context_identifier: int, warn: bool = True) -> DataCollectionContext:
        """
        Resume the data collection of the variables of the given context
        :param context_identifier: the identifier of the context
        :param warn: the flag specifying if a warning must be logged if context is running
        """
        if context_identifier is None:
            raise ValueError("An error occurred", "Identifier of the context is not valid", context_identifier)
        if not self.is_running():
            raise ValueError("Data collector has been closed")
        with self._lock:
            if context_identifier not in self._contexts.keys():
                if warn:
                    self._logger.warning('Context has not been started')
                return self._contexts[context_identifier].get_context()
            wrapper: _DataCollectionContextWrapper = self._contexts[context_identifier]
            if wrapper.is_active():
                if warn:
                    self._logger.warning('Context is running')
                return wrapper.get_context()
            if wrapper.get_context().get_end_date() is not None:
                if warn:
                    self._logger.warning('Context has been stopped')
                return wrapper.get_context()
            self._logger.info('Resuming context: %s' % wrapper.get_context().get_identifier())
            wrapper.set_active(True)
            if not self._task.is_active():
                self._task.set_active(True)

    def is_running(self) -> bool:
        """
        Check if the collector is still active or not
        :return: the boolean
        """
        return self._scheduler is not None

    def is_context_running(self, context_identifier: int) -> bool:
        """
        Check if the context associated to the given identifier is still running
        :return: the boolean
        """
        with self._lock:
            if context_identifier in self._contexts.keys():
                return self._contexts[context_identifier].is_active() and self._contexts[
                    context_identifier].get_context().get_start_date() is not None and self._contexts[
                           context_identifier].get_context().get_end_date() is None
        return False

    # noinspection PyTypeChecker
    def get_context(self, context_identifier: int) -> _DataCollectionContextWrapper:
        """
        Return the context associated to the given identifier
        :return: the context or None
        """
        with self._lock:
            if context_identifier in self._contexts.keys():
                return self._contexts[context_identifier]
        return None

    def get_interval_limit_policy(self) -> IntervalLimitPolicy:
        """
        Return the interval limit policy
        :return: the interval limit policy
        """
        return self._interval_limit_policy

    def get_life_duration(self) -> int:
        """
        Return the maximum life duration in seconds
        :return: the maximum life duration in seconds
        """
        return self._life_duration

    # noinspection PyTypeChecker
    def get_reader(self, reader_type: str) -> DataReader:
        """
        Return the reader associated to the given type
        :return: the reader or None
        """
        if reader_type in self._readers:
            return self._readers[reader_type]
        return None

    def get_writers(self) -> List[DataWriter]:
        """
        Return the writers
        :return: the writers
        """
        return self._writers

    def get_scheduler(self) -> sched.scheduler:
        """
        Return the scheduler
        :return: the scheduler
        """
        return self._scheduler

    def get_thread_pool(self) -> Pool:
        """
        Return the threads pool
        :return: the threads pool
        """
        return self._thread_pool

    def set_interval_limit_policy(self, value: IntervalLimitPolicy) -> None:
        """
        Set the interval limit policy
        :param value: the interval limit policy
        """
        if value is None:
            self._interval_limit_policy = IntervalLimitPolicy.IGNORE
        else:
            self._interval_limit_policy = value

    def set_life_duration(self, value: int) -> None:
        """
        Set the maximum life duration in seconds
        :param value: the maximum life duration in seconds
        """
        if value is None or value <= 0:
            raise ValueError("An error occurred", "Invalid value", value)
        self._life_duration = value

    def _close(self, warn: bool = False):
        """
        Close the data collector
        :param warn: the flag specifying if a warning must be logged if context is not running
        """
        with self._lock:
            self._logger.info('Closing collector')
            if self._listener is not None:
                self._listener.before_close(self)
            self._logger.info('Stopping contexts: %s...' % (len(self._contexts)))
            identifiers = list()
            for context_identifier in self._contexts.keys():
                identifiers.append(context_identifier)
            for context_identifier in identifiers:
                self.stop(context_identifier, warn)
            self._contexts.clear()
            self._task.set_active(False)
            if self._thread_pool is not None:
                self._thread_pool.close()
                self._thread_pool.join()
                self._thread_pool.terminate()
                self._thread_pool = None
            if self._scheduler is not None:
                self._logger.info('Closing scheduler...')
                self._scheduler = None
            if self._listener is not None:
                self._listener.after_close(self)

    def close(self):
        """
        Close the data collector
        """
        self._close(True)
