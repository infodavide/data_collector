# Data Collector Python module
A data collection module in Python to collect variables using readers and writers.

A module to collect values of variables at fixed interval of time using a context, one or many readers and one or many writers to persist or publish the values.

The implementation of this module is based on the use of the industry. The industry sometimes uses some collects to read the data from an equipment using one or more protocols like Modbus, OPC, etc.

## The context or data collection context

The context (DataCollectionContext class) describes:
- a numeric identifier from an external repository from where it can be retrieved,
- a *minimum interval of time* in milliseconds is available in this object to prevent issues when trying to collect data too fast. Note that each reader can override this value by providing a different minimum interval,
- a *default interval of time* in milliseconds is also available. It will be automatically used if variable does not specify its interval,
- a *duration of the collect* in seconds is used to avoid infinite running. The default life duration of a context is 24 hours,
- a *start and a end date* are available and are filled on start and stop of the collect.
- the variables in its *plan*.

## The variables or data collection plan

The variables (Variable class) are associated to a context and a specific reader.

A variable describes:
- a numeric identifier from an external repository from where it can be retrieved,
- a textual representation of an address is available too for the variable,
- a textual and Python oriented representation of the type of the value associated with the variable.
- a reader is associated to the variable with a textual identifier of type,
- a read interval in milliseconds (optional).

## The readers

THe reader (DataReader abstract class) is in charge of reading a list of variables (CollectedVariables class).

The reader can set the value and the read datetime of the variable.

Reading is done in a different task (internal _DataReaderTask class) and thread per type of reader.

If you need to read values using Modbus, you have to implement your own reader class that extends the DataReader abstract class. The instanciation of your reader is specific and should be done before instanciating the collector.

__Take a look at the file 'data_reader/random_data_reader.py' which implements a reader generating random values for variables.__

## The writers

The writer (DataWriter abstract class) is in charge of writing a list of variables (CollectedVariables class).

Writing is done in a different task (internal _DataWriterTask class) and is triggered by each reader thread.

If you need to write the data to a database or NoSQL repository, you have to implement your own writer class that extends the DataWriter abstract class. The instanciation of your writer is specific and should be done before instanciating the collector.

__Take a look at the file 'data_writer/noop_data_writer.py' which implements a writer logging some data of each retrieved variable and value.__

If you need to write synchronously the data, for example in a file, you have to implement your own writer using a synchrization mechanism to avoid concurrent writes on the same output file (No map and reduce mechanism is provided to write  the data in a single thread at the end of the collect).

__Take a look at the file 'data_writer/csv_file_data_writer.py' which implements a writer using a reentrant lock to write a CSV file.__

## The data collector

The data collector object (DataCollector class) is instantiated using a logger, a list of readers and a list of writers.

To start a data collection, you just have to invoke the start method on the collector object with a data collection context. 

You can start many contexts, the variables will be retrieved only one time if the interval of the collect is the same as the one specified by the running contexts. 

When a context is started, the mechanism will compute a new retrieval interval for a set of variables. For example, if the context A uses a variable with an interval of 300ms and a context B uses the same variable with a different interval of 200ms, then the retrieval will be computed and set to 100ms (gcd).

You can then pause and resume the data collection using the identifier of the context passed on the initial start.

Finally, you can complete the data collection using the stop method on the collector object with the context identifier.

The close method of the collector can called to stop all the contexts, clean up the thread pool, the scheduler.

The events associated with the actions on the collector object can be listened using a derived class of the abstract DataCollectionListener class which defines the following methods:
- before_close: called before closing the collector object,
- after_close: called when all the internal objects of the collector have been cleaned up,
- before_start: called before starting the collect of a context,
- after_start: called when the collect of a context has been started,
- before_stop: called before stopping the collect of a context,
- after_stop: called when the collect of a context has been stopped,

__Take a look at the file 'test/data_collector_test.py' which implements unit tests and build a reader, a writer, a context and the data collector.__

## On going

The current development is *not fully tested*:
- the unit tests have to be completed too.