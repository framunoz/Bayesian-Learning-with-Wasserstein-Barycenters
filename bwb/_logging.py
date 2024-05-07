"""
Script with all the necessary information for the library loggers.
You can use this module by importing it as follows::

    from library_name import _logging as logging

    # The name of the logger should be the same of the module
    logger = logging.get_logger(__name__)

And change the level of any logger with::

    logging.set_level(logging.DEBUG, name=name_of_the_logger)

If you want to change the level of all loggers, omit the name.
"""

import functools
import logging
import threading
import time
import typing as t
import warnings
from datetime import timedelta

# The levels of the severity as in the package 'logging'
CRITICAL = logging.CRITICAL
ERROR = logging.ERROR
WARNING = logging.WARNING
INFO = logging.INFO
DEBUG = logging.DEBUG
NOTSET = logging.NOTSET

# Configure a formatter
GENERIC_FORMATTER: logging.Formatter = logging.Formatter(
    "%(asctime)s: %(levelname)s [%(name)s:%(lineno)s]\n> %(message)s"
)
"""The generic formatter of the loggers."""

# Generic Handler of the package
STREAM_HANDLER: logging.Handler = logging.StreamHandler()
"""The generic handler in the library"""
STREAM_HANDLER.setFormatter(GENERIC_FORMATTER)


# noinspection DuplicatedCode
class _SingletonMeta(type):
    """Metaclass to implements the Singleton Pattern. Obtained from
    https://refactoring.guru/design-patterns/singleton/python/example#example-1
    """

    _instances = {}
    _lock: threading.Lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


class LoggerConfiguration(metaclass=_SingletonMeta):
    """A singleton that manages the configuration of the loggers in the
    library.

    :cvar LEVEL: The general level of the loggers in the library.
        Default is ``WARNING``.
    :cvar FORMATTER: The generic formatter of the loggers.
    :cvar HANDLERS: The list of handlers handled by the library.
        Default is ``[STREAM_HANDLER]``.
    :cvar loggers: The loggers instance of the package.


    Example
    -------

    You can instance a logger configuration with the default values.

    >>> log_conf = LoggerConfiguration().reset()  # Reset is optional
    >>> log_conf
    LoggerConfiguration(LEVEL=30, HANDLERS=[<StreamHandler <stderr> (NOTSET)>])

    Set the level of the loggers.

    >>> log_conf.set_level(logging.INFO)
    LoggerConfiguration(LEVEL=20, HANDLERS=[<StreamHandler <stderr> (NOTSET)>])

    And get a logger with a specific name.

    >>> logger = log_conf.get_logger("ex")
    >>> logger.debug("This is a debug message.")  # This will not show
    """

    LEVEL: int = WARNING
    FORMATTER: logging.Formatter = GENERIC_FORMATTER
    HANDLERS: list[logging.Handler] = [STREAM_HANDLER]
    loggers: dict[str, logging.Logger] = dict()

    def reset(self) -> t.Self:
        """
        Reset the logger configuration to the default values.

        Example
        -------

        We set the level of the loggers to DEBUG.

        >>> log_conf = LoggerConfiguration().set_level(logging.DEBUG)
        >>> log_conf.LEVEL == logging.DEBUG
        True

        And reset the logger configuration.

        >>> _ = log_conf.reset()
        >>> log_conf.LEVEL == logging.WARNING
        True
        """
        self.LEVEL = WARNING
        self.HANDLERS = [STREAM_HANDLER]
        self.loggers = dict()
        return self

    def attach_handlers(
        self, logger: logging.Logger, handlers: list[logging.Handler] = None
    ) -> t.Self:
        """
        Add handlers from the handler list to the logger, if they are
        not already in the logger. If they are already in the logger,
        it does not add them.

        :param logger: The logger instance.
        :param handlers: The list of handlers. Default is the list of
            handlers managed.

        Example
        -------

        We first create a logger that has no handlers. The reset is
        optional.

        >>> log_conf = LoggerConfiguration().reset()
        >>> log_ex = logging.getLogger("attach_handlers_example")
        >>> len(log_ex.handlers)  # There are no handlers in the logger
        0

        We attach the handlers to the logger.

        >>> _ = log_conf.remove_all_handlers(log_ex).attach_handlers(log_ex)
        >>> len(log_ex.handlers)  # Now, there are the default handlers
        1
        >>> log_ex.handlers[0] is log_conf.HANDLERS[0]
        True

        We can attach more handlers to the logger.

        >>> import logging
        >>> handler = logging.StreamHandler()
        >>> log_conf.attach_handlers(log_ex, [handler])
        LoggerConfiguration(LEVEL=30, HANDLERS=[<StreamHandler <stderr> (NOTSET)>])
        >>> len(log_ex.handlers)
        2
        """
        # Set the default handlers
        if handlers is None:
            handlers = self.HANDLERS

        for handler in handlers:
            if handler not in logger.handlers:
                logger.addHandler(handler)

        return self

    def get_logger(
        self,
        name: str,
        level: int = None,
        handlers: list[logging.Handler] = None
    ) -> logging.Logger:
        """
        Gets the logger by its name. Additionally, attach the handlers
        and sets the level.

        :param name: The name of the logger
        :param handlers: A list of handlers to attach to the logger.
        :param level: The level of severity of the logger.
            Default is the general level of the loggers in the library.
        :return: The logger instance.

        Example
        -------

        We get a logger with the name 'get_logger_example'.
        The reset is optional.

        >>> log_conf = LoggerConfiguration().reset()
        >>> log = log_conf.get_logger("get_logger_example")

        We can check if the logger is in the loggers' dictionary.

        >>> log in log_conf.loggers.values()
        True

        We can get the logger again, and it should be the same instance.

        >>> log2 = log_conf.get_logger("get_logger_example")
        >>> log is log2
        True

        We can get the logger with a different level.

        >>> import logging
        >>> log3 = log_conf.get_logger("get_logger_example3", logging.DEBUG)
        >>> log3.level == logging.DEBUG
        True

        We can get the logger with a different handler.

        >>> handler = logging.StreamHandler()
        >>> log4 = log_conf.get_logger("get_logger_example4", handlers=[handler])
        >>> handler in log4.handlers
        True

        We can get the logger with a different level and handler.

        >>> log5 = log_conf.get_logger("get_logger_example5", logging.DEBUG, [handler])
        >>> log5.level == logging.DEBUG and handler in log5.handlers
        True

        We can also use the function ``get_logger`` that is an alias
        of the method.

        >>> log6 = get_logger("get_logger_example6")
        >>> log6 in log_conf.loggers.values()
        True
        """
        if name in self.loggers:
            logger = self.loggers[name]
        else:
            logger = logging.getLogger(name)
        self.loggers[name] = logger
        self.attach_handlers(logger, handlers)
        logger.setLevel(level or self.LEVEL)
        return logger

    def set_level(self, level: int, name: str = None) -> t.Self:
        r"""
        Set the level of severity of the loggers in the library.
        You can specify the logger name to change the level to a
        single logger.

        :param level: The level of severity. See
            `levels
            <https://docs.python.org/es/3/library/logging.html#levels>`_
            from the logging package for further information.
        :param name: The name of the logger to change the level.
            If it is None (default), change the severity of every logger
            in the package.

        Example
        -------

        We set the level of the loggers to DEBUG. The reset is optional.

        >>> import logging
        >>> log_conf = LoggerConfiguration().reset().set_level(logging.DEBUG)
        >>> log_conf.LEVEL == logging.DEBUG
        True

        If we have a logger that was created with the logger
        configuration, it should have the same level.

        >>> log = log_conf.get_logger("example")
        >>> log.level == logging.DEBUG
        True

        And we can change the level of the logger, only with the
        logger configuration.

        >>> _ = log_conf.set_level(logging.INFO)
        >>> log.level == logging.INFO
        True

        We can change the level of a single logger, without changing
        the level of the other loggers.

        >>> log2 = log_conf.get_logger("example2")
        >>> log2.level == logging.INFO
        True
        >>> log_conf.set_level(logging.WARNING, "example")
        LoggerConfiguration(LEVEL=20, HANDLERS=[<StreamHandler <stderr> (NOTSET)>])
        >>> log.level == logging.WARNING
        True
        >>> log2.level == logging.INFO
        True
        >>> log_conf.LEVEL == logging.INFO
        True

        If the name is not in the loggers, it raises a ValueError.

        >>> log_conf.set_level(logging.INFO, "example3")
        Traceback (most recent call last):
            ...
        ValueError: The name must correspond to a logger with that name. There is currently no logger with this name.

        We can also use the function ``set_level`` that is an alias
        of the method.

        >>> set_level(logging.CRITICAL)
        LoggerConfiguration(LEVEL=50, HANDLERS=[<StreamHandler <stderr> (NOTSET)>])
        """
        if name in self.loggers:
            self.loggers[name].setLevel(level)
            return self

        # If name not in loggers, AND it is not a None value...
        elif name is not None:
            raise ValueError(
                "The name must correspond to a logger with that name. "
                "There is currently no logger with this name."
            )

        # Otherwise, iterate over every logger
        for logger in self.loggers.values():
            logger.setLevel(level)
        self.LEVEL = level

        return self

    def __repr__(self):
        return (f"LoggerConfiguration(LEVEL={self.LEVEL}, "
                f"HANDLERS={self.HANDLERS})")

    def remove_all_handlers(self, logger: logging.Logger = None) -> t.Self:
        """
        Removes all the handlers from the logger. If the logger is None,
        it removes all the handlers from every logger in the configuration.
        This would be useful when you want to change the handlers of
        the logger configuration.

        :param logger: The logger to remove the handlers.
            Default is None.

        Example
        -------

        We create a logger. The reset is optional.

        >>> log_conf = LoggerConfiguration().reset()
        >>> log = log_conf.get_logger("remove_all_handlers_example")
        >>> len(log.handlers)  # The handler by default
        1

        And remove the handlers from the logger.

        >>> log_conf.remove_all_handlers(log)
        LoggerConfiguration(LEVEL=30, HANDLERS=[<StreamHandler <stderr> (NOTSET)>])
        >>> len(log.handlers)
        0

        But, if we create another logger, it should have the default handler.

        >>> log2 = log_conf.get_logger("remove_all_handlers_example2")
        >>> len(log2.handlers)
        1

        We can remove all the handlers from every logger in the package.

        >>> log_conf.remove_all_handlers()
        LoggerConfiguration(LEVEL=30, HANDLERS=[])
        >>> len(log.handlers)
        0
        >>> log3 = log_conf.get_logger("remove_all_handlers_example3")
        >>> len(log3.handlers)
        0

        """
        if logger is not None:
            if logger.hasHandlers():
                for handler in logger.handlers:
                    logger.removeHandler(handler)

            return self

        for logger in self.loggers.values():
            if logger.hasHandlers():
                for handler in logger.handlers:
                    logger.removeHandler(handler)

        self.HANDLERS = []

        return self

    def add_handler(
        self,
        handler: logging.Handler,
        logger: logging.Logger = None
    ) -> t.Self:
        """
        Add a handler to the logger. If the logger is None, add the
        handler to every logger in the package.

        :param handler: The handler to add.
        :param logger: The logger to add the handler. Default is None.

        Example
        -------

        We create a handler. The reset is optional.

        >>> log_conf = LoggerConfiguration().reset()
        >>> handler = logging.NullHandler()

        And add the handler to the logger configuration.

        >>> log_conf.add_handler(handler)
        LoggerConfiguration(LEVEL=30, HANDLERS=[<StreamHandler <stderr> (NOTSET)>, <NullHandler (NOTSET)>])

        If we create a logger, it should have the handler.

        >>> log = log_conf.get_logger("add_handler_example")
        >>> handler in log.handlers
        True
        >>> len(log.handlers)
        2

        We can add the handler to a specific logger.

        >>> log2 = logging.getLogger("add_handler_example2")
        >>> _ = log_conf.add_handler(handler, log2)
        >>> handler in log2.handlers
        True
        >>> len(log2.handlers)
        1

        And if we add the handler to the logger configuration, it should
        update on every logger.

        >>> other_handler = logging.StreamHandler()
        >>> _ = log_conf.add_handler(other_handler)
        >>> other_handler in log.handlers
        True
        """
        if logger is not None:
            logger.addHandler(handler)
            return self

        for logger in self.loggers.values():
            if handler not in logger.handlers:
                logger.addHandler(handler)

        self.HANDLERS.append(handler)
        return self

    def set_formatter(
        self,
        handler: logging.Handler = None,
        formatter: logging.Formatter = None,
    ) -> t.Self:
        """
        Set the formatter to the handler. If the formatter is ``None``,
        the default formatter of the package is set. If the handler is
        ``None``, set the formatter to every handler in the package.

        :param handler: The handler to set the formatter.
            Default is None.
        :param formatter: The formatter to set. Default is the default
            formatter of the package.
        :return: The instance of the logger configuration.

        Example
        -------

        We create a logger configuration with the default values.

        >>> log_conf = LoggerConfiguration().reset()

        We can set the default formatter to a handler.

        >>> handler = logging.StreamHandler()
        >>> log_conf.set_formatter(handler)
        LoggerConfiguration(LEVEL=30, HANDLERS=[<StreamHandler <stderr> (NOTSET)>])
        >>> handler.formatter == GENERIC_FORMATTER
        True

        We can set the default formatter to every handler in the package.

        >>> formatter = logging.Formatter("%(message)s")
        >>> other_handler = logging.StreamHandler()
        >>> other_handler.setFormatter(formatter)
        >>> log_conf.add_handler(other_handler)
        LoggerConfiguration(LEVEL=30, HANDLERS=[<StreamHandler <stderr> (NOTSET)>, <StreamHandler <stderr> (NOTSET)>])
        >>> all(handler.formatter == formatter for handler in log_conf.HANDLERS)
        False
        >>> log_conf.set_formatter(formatter=formatter)
        LoggerConfiguration(LEVEL=30, HANDLERS=[<StreamHandler <stderr> (NOTSET)>, <StreamHandler <stderr> (NOTSET)>])
        >>> all(handler.formatter == formatter for handler in log_conf.HANDLERS)
        True
        """
        # Set the default formatter
        formatter_ = formatter or self.FORMATTER

        if handler is not None:
            handler.setFormatter(formatter_)
            return self

        for handler in self.HANDLERS:
            handler.setFormatter(formatter_)

        if formatter is not None:
            self.FORMATTER = formatter

        return self

    @staticmethod
    def raise_warning(
        msg: str,
        logger: logging.Logger,
        warning_category: t.Type[Warning] = None,
        stacklevel: int = 1,
        level: int = WARNING,
        **kwargs
    ) -> None:
        """
        Raise a warning with the logger and raise a warning with the
        warning category.

        :param logger: The logger instance.
        :param warning_category: The warning category to raise.
        :param msg: The message of the warning.
        :param level: The level of the logger. Default is WARNING.
        :param stacklevel: The stack level of the warning. Default is 1.
        :param kwargs: Other arguments to pass to the warning.
        :return: None

        Example
        -------

        We set a logger with the name 'raise_warning_example', and we
        set the stream to a
        `StringIO <https://docs.python.org/3/library/io.html#io.StringIO>`_

        >>> from io import StringIO
        >>> import logging
        >>> log_config = LoggerConfiguration().reset().remove_all_handlers()
        >>> log_stream = StringIO()
        >>> _ = log_config.add_handler(logging.StreamHandler(log_stream))
        >>> log = log_config.get_logger("raise_warning_example", logging.WARNING)

        We can raise a warning with the logger and the warning category.

        >>> # We catch the warning
        >>> with warnings.catch_warnings(record=True) as w:
        ...     log_config.raise_warning("This is a warning.", log, UserWarning)
        >>> print(w[0].message.args[0])
        This is a warning.
        >>> print(w[0].category)
        <class 'UserWarning'>

        And the message should be in the logger.

        >>> print(log_stream.getvalue()[:-1])
        This is a warning.
        >>> _ = log_stream.seek(0)      # Clean the stream
        >>> _ = log_stream.truncate(0)  # Clean the stream


        We can also use the function ``raise_warning`` that is an alias
        of the method.

        >>> raise_warning("This is a warning.", log, UserWarning)
        >>> print(log_stream.getvalue()[:-1])
        This is a warning.

        """
        logger.log(level, msg)
        warnings.warn(msg, warning_category, stacklevel=stacklevel, **kwargs)

    @staticmethod
    def raise_error(
        msg: str,
        logger: logging.Logger,
        error_category: t.Type[Exception],
        level: int = ERROR,
        **kwargs
    ) -> None:
        """
        Raise an error with the logger and raise an error with
        the error category.

        :param logger: The logger instance.
        :param error_category: The error category to raise.
        :param msg: The message of the error.
        :param level: The level of the logger. Default is ERROR.
        :param kwargs: Other arguments to pass to the error.
        :return: None

        Example
        -------

        We set a logger with the name 'raise_error_example', and we
        set the stream to a
        `StringIO <https://docs.python.org/3/library/io.html#io.StringIO>`_

        >>> from io import StringIO
        >>> import logging
        >>> log_config = LoggerConfiguration().reset().remove_all_handlers()
        >>> log_stream = StringIO()
        >>> _ = log_config.add_handler(logging.StreamHandler(log_stream))
        >>> log = log_config.get_logger("raise_error_example", logging.ERROR)

        We can raise an error with the logger and the error category.

        >>> # We catch the error
        >>> try:
        ...     log_config.raise_error("This is an error.", log, AssertionError)
        ... except AssertionError as e:
        ...     print(e)
        This is an error.

        And the message should be in the logger.

        >>> print(log_stream.getvalue()[:-1])
        This is an error.
        >>> _ = log_stream.seek(0)      # Clean the stream
        >>> _ = log_stream.truncate(0)

        We can also use the function ``raise_error`` that is an alias
        of the method.

        >>> try:
        ...     raise_error("This is an error.", log, AssertionError)
        ... except AssertionError as e:
        ...     print(e)
        This is an error.
        """
        logger.log(level, msg)
        # noinspection PyArgumentList
        raise error_category(msg, **kwargs)


# Create the (single) instance of LoggerConfiguration
log_config = LoggerConfiguration()


# Alias of the methods in the instance of LoggerConfiguration
# noinspection PyMissingOrEmptyDocstring
def get_logger(
    name: str,
    level: int = log_config.LEVEL,
    handlers: list[logging.Handler] = None
) -> logging.Logger:
    return log_config.get_logger(name, level, handlers)


# noinspection PyMissingOrEmptyDocstring
def set_level(level: int, name: str = None) -> LoggerConfiguration:
    return log_config.set_level(level, name)


# noinspection PyMissingOrEmptyDocstring
def raise_warning(
    msg: str,
    logger: logging.Logger,
    warning_category: t.Type[Warning] = None,
    stacklevel: int = 1,
    level: int = WARNING,
    **kwargs
) -> None:
    log_config.raise_warning(msg, logger, warning_category, stacklevel, level,
                             **kwargs)


# noinspection PyMissingOrEmptyDocstring
def raise_error(
    msg: str,
    logger: logging.Logger,
    error_category: t.Type[Exception],
    level: int = ERROR,
    **kwargs
) -> None:
    log_config.raise_error(msg, logger, error_category, level, **kwargs)


get_logger.__doc__ = LoggerConfiguration.get_logger.__doc__
set_level.__doc__ = LoggerConfiguration.set_level.__doc__
raise_warning.__doc__ = LoggerConfiguration.raise_warning.__doc__
raise_error.__doc__ = LoggerConfiguration.raise_error.__doc__


def _time_fmt(seconds: float) -> str:
    """
    Format the time in seconds to a string.

    :param seconds: The time in seconds.
    :return: The formatted time in a string.

    Example
    -------

    Format the time in seconds to a string.

    >>> from math import pi
    >>> _time_fmt(pi)
    '0:00:03.141593'

    Format the time in seconds to a string with minutes.

    >>> minute = 60
    >>> _time_fmt(minute + pi)
    '0:01:03.14'

    Format the time in seconds to a string with hours.

    >>> hour = minute * 60
    >>> _time_fmt(hour + minute + pi)
    '1:01:03'

    Format the time in seconds to a string with days.

    >>> day = hour * 24
    >>> _time_fmt(day + hour + minute + pi)
    '1 day, 1:01'
    """
    dt = timedelta(seconds=seconds)
    dt_fmt = str(dt)

    # Return in the format 'days, hours:min'
    if dt.days > 0:
        return dt_fmt[:-10]

    # Return in the format 'hours:min:sec'
    if dt.seconds > 3600:
        return dt_fmt[:-7]

    # Return in the format 'hours:min:sec.microsec (rounded)'
    if dt.seconds > 60:
        return dt_fmt[:-4]

    # Return in the format 'hours:min:sec.microsec'
    return dt_fmt


class register_total_time:
    """
    Class that registers the total time it takes to execute a piece
    of code, and shows it with the logger. It can be used as a context
    manager or as a decorator.

    Example
    -------

    First, we set a logger with the name 'example', and we set the
    stream to a `StringIO
    <https://docs.python.org/3/library/io.html#io.StringIO>`_

    >>> from io import StringIO
    >>> import logging
    >>> log = log_config.get_logger("example", logging.DEBUG)
    >>> log_stream = StringIO()
    >>> log.addHandler(logging.StreamHandler(log_stream))

    You can use the class as a context manager, and it will show the
    time it takes to execute the block of code. You can access the
    elapsed time with the attribute ``elapsed_time``.

    >>> with register_total_time(log) as timer:
    ...     time.sleep(1)
    >>> round(timer.elapsed_time)
    1
    >>> print(log_stream.getvalue()[:-8])  # Remove the microseconds
    Starting the block of code...
    The block of code takes 0:00:01
    >>> log_stream.seek(0)      # Clean the stream
    0
    >>> log_stream.truncate(0)  # Clean the stream
    0

    We can use the class as a decorator for a function.

    >>> @register_total_time(log)
    ... def test_function():
    ...     time.sleep(1)
    >>> test_function()
    >>> print(log_stream.getvalue()[:-8])  # Remove the microseconds
    The function 'test_function' takes 0:00:01

    And even set the level of the logger.

    >>> log2 = log_config.get_logger("example2", logging.WARNING)
    >>> log_stream2 = StringIO()
    >>> log2.addHandler(logging.StreamHandler(log_stream2))
    >>> @register_total_time(log2, logging.INFO)  # This will not show
    ... def test_function():
    ...     time.sleep(1)
    >>> test_function()
    >>> log_stream2.getvalue()  # Nothing to show
    ''
    """

    def __init__(self, logger: logging.Logger, level: int = logging.DEBUG):
        self.logger = logger
        self.level = level
        self.tic = time.perf_counter()
        self.toc = time.perf_counter()

    @property
    def elapsed_time(self):
        """
        The elapsed time between the start and the end of
        the block of code.

        :return: The elapsed time in seconds.
        """
        return self.toc - self.tic

    def __enter__(self):
        self.tic = time.perf_counter()
        self.logger.log(self.level, "Starting the block of code...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.toc = time.perf_counter()
        dt_fmt = _time_fmt(self.elapsed_time)
        self.logger.log(self.level, f"The block of code takes {dt_fmt}")

    def __call__(self, func):
        """
        Decorator to register the total time it takes to execute a
        function, and shows it with the logger.

        :param func: The function to decorate
        :return: The decorated function
        """

        # noinspection PyMissingOrEmptyDocstring
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tic = time.perf_counter()
            result = func(*args, **kwargs)
            toc = time.perf_counter()
            dt_fmt = _time_fmt(toc - tic)
            self.logger.log(
                self.level,
                f"The function '{func.__name__}' takes {dt_fmt}"
            )
            return result

        return wrapper


def register_total_time_function(
    logger: logging.Logger,
    level: int = logging.DEBUG
):
    """
    Wrapper that records the total time it takes to execute a function,
    and shows it with the logger.

    :param logger: A `Logger` instance
    :param level: The level of the logger
    :return: The decorator

    Example
    -------

    We set a logger with the name 'example', and we set the stream to a `StringIO
    <https://docs.python.org/3/library/io.html#io.StringIO>`_

    >>> from io import StringIO
    >>> import logging
    >>> log = log_config.get_logger("example", logging.DEBUG)
    >>> log_stream = StringIO()
    >>> log.addHandler(logging.StreamHandler(log_stream))

    We can use the function decorator for a function.

    >>> @register_total_time_function(log)
    ... def test_function():
    ...     time.sleep(1)
    >>> test_function()
    >>> print(log_stream.getvalue()[:-8])  # Remove the microseconds
    The function 'test_function' takes 0:00:01
    """

    # noinspection PyMissingOrEmptyDocstring
    def decorator(func):
        # noinspection PyMissingOrEmptyDocstring
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tic = time.perf_counter()
            result = func(*args, **kwargs)
            toc = time.perf_counter()
            dt_fmt = _time_fmt(toc - tic)
            logger.log(level, f"The function '{func.__name__}' takes {dt_fmt}")
            return result

        return wrapper

    return decorator


def register_total_time_method(
    logger: logging.Logger,
    level: int = logging.DEBUG
):
    """
    Wrapper that records the total time it takes to execute a method,
    and shows it with the logger.

    :param logger: A `Logger` instance
    :param level: The level of the logger
    :return: The decorator

    Example
    -------

    We set a logger with the name 'example', and we set the stream to a
    `StringIO <https://docs.python.org/3/library/io.html#io.StringIO>`_

    >>> from io import StringIO
    >>> import logging
    >>> log = log_config.get_logger("example", logging.DEBUG)
    >>> log_stream = StringIO()
    >>> log.addHandler(logging.StreamHandler(log_stream))

    We can use the method decorator for a method.

    >>> class Test:
    ...     @register_total_time_method(log)
    ...     def test_method(self):
    ...         time.sleep(1)
    >>> Test().test_method()
    >>> print(log_stream.getvalue()[:-8])  # Remove the microseconds
    Using the method 'Test.test_method'...
    The method 'Test.test_method' takes 0:00:01
    """

    # noinspection PyMissingOrEmptyDocstring
    def decorator(method):
        # noinspection PyMissingOrEmptyDocstring
        @functools.wraps(method)
        def wrapper(*args, **kwargs):
            self: object = args[0]
            logger.log(
                level,
                f"Using the method "
                f"'{self.__class__.__name__}.{method.__name__}'..."
            )
            tic = time.perf_counter()
            result = method(*args, **kwargs)
            toc = time.perf_counter()
            dt_fmt = _time_fmt(toc - tic)
            logger.log(
                level,
                f"The method "
                f"'{self.__class__.__name__}.{method.__name__}' takes {dt_fmt}"
            )
            return result

        return wrapper

    return decorator


def register_init_method(logger: logging.Logger, level: int = logging.DEBUG):
    """
    Logs the use of a method, indicating the name of the method and the
    name of the class.

    :param logger: A `Logger` instance
    :param level: The level of the logger
    :return: The decorator

    Example
    -------

    We set a logger with the name 'example', and we set the stream to a
    `StringIO <https://docs.python.org/3/library/io.html#io.StringIO>`_

    >>> from io import StringIO
    >>> import logging
    >>> log = log_config.get_logger("example", logging.DEBUG)
    >>> log_stream = StringIO()
    >>> log.addHandler(logging.StreamHandler(log_stream))

    We can use the method decorator for a method.

    >>> class Test:
    ...     @register_init_method(log)
    ...     def foo(self):
    ...         pass
    >>> Test().foo()
    >>> print(log_stream.getvalue()[:-1])
    Using the method 'foo' in the class 'Test'.
    """

    # noinspection PyMissingOrEmptyDocstring
    def decorator(method):
        # noinspection PyMissingOrEmptyDocstring
        @functools.wraps(method)
        def wrapper(*args, **kwargs):
            self: object = args[0]  # We are in a method of a class
            class_name = self.__class__.__name__
            method_name = method.__name__
            logger.log(
                level,
                f"Using the method '{method_name}' "
                f"in the class '{class_name}'."
            )
            # Compute the result
            result = method(*args, **kwargs)
            return result

        return wrapper

    return decorator
