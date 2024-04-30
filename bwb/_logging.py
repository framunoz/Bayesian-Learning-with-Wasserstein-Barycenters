"""
Script with all the necessary information for the library loggers.
You can use this module by importing it as follows::

    from library_name import logging
    logger = logging.get_logger(__name__)  # The name of the logger should be the same of the module

And change the level of any logger with::

    logging.set_level(logging.DEBUG, name=name_of_the_logger)

If you want to change the level of all loggers, omit the name.
"""

import functools
import logging
import threading
import time
import warnings
from datetime import timedelta
from typing import Type

# The levels of the severity as in the package 'logging'
CRITICAL = logging.CRITICAL
ERROR = logging.ERROR
WARNING = logging.WARNING
INFO = logging.INFO
DEBUG = logging.DEBUG
NOTSET = logging.NOTSET

# Configure a formatter
FORMATTER: logging.Formatter = logging.Formatter(
    "%(asctime)s: %(levelname)s [%(name)s:%(lineno)s]\n> %(message)s"
)
"""The generic formatter of the loggers."""

# Generic Handler of the package
STREAM_HANDLER: logging.Handler = logging.StreamHandler()
"""The generic handler in the library"""
STREAM_HANDLER.setFormatter(FORMATTER)


# noinspection DuplicatedCode
class _SingletonMeta(type):
    """Metaclass to implements Singleton Pattern. Obtained from
    https://refactoring.guru/design-patterns/singleton/python/example#example-1"""

    _instances = {}
    _lock: threading.Lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


class LoggerConfiguration(metaclass=_SingletonMeta):
    """A singleton that manages the configuration of the loggers in the library."""

    LEVEL: int = WARNING
    """The general level of the loggers in the library."""

    HANDLERS: list[logging.Handler] = [STREAM_HANDLER]
    """The list of handlers handled by the library."""

    loggers: dict[str, logging.Logger] = dict()
    """The loggers instances of the package"""

    def attach_handlers(
        self, logger: logging.Logger, handlers: list[logging.Handler] = None
    ):
        """
        Add handlers from the handler list to the logger, if they are not already in the logger.
        If they are already in the logger, it does not add them.

        :param logger: The logger instance.
        :param handlers: The list of handlers. Default is the list of handlers managed.
        """
        # Set the default handlers
        if handlers is None:
            handlers = self.HANDLERS

        for handler in handlers:
            if handler not in logger.handlers:
                logger.addHandler(handler)

    def get_logger(
        self, name: str, level: int = LEVEL, handlers: list[logging.Handler] = None
    ) -> logging.Logger:
        """
        Gets the logger by its name. Additionally, attach the handlers and sets the level.

        :param name: The name of the logger
        :param handlers: A list of handlers to attach to the logger.
        :param level: The level of severity of the logger. Default is the general level of the
            loggers in the library.
        :return: The logger instance
        """
        if name in self.loggers:
            logger = self.loggers[name]
        else:
            logger = logging.getLogger(name)
        self.loggers[name] = logger
        self.attach_handlers(logger, handlers)
        logger.setLevel(level)
        return logger

    def set_level(self, level: int, name: str = None):
        r"""
        Set the level of severity of the loggers in the library. You can specify the logger name
        to change the level to a single logger.

        :param level: The level of severity. See
            `levels <https://docs.python.org/es/3/library/logging.html#levels>` from the logging
            package for further information.
        :param name: The name of the logger to change the level. If it is None (default),
            change the severity of every logger in the package.
        """
        if name in self.loggers:
            self.loggers[name].setLevel(level)
            return None
        # If name not in loggers, AND it is not a None value...
        elif name is not None:
            raise ValueError(
                "The name must correspond to a logger with that name. There is currently no "
                "logger with this name."
            )
        # Otherwise, iterate over every logger
        for logger in self.loggers.values():
            logger.setLevel(level)
        self.LEVEL = level

    def __repr__(self):
        return f"LoggerConfiguration(LEVEL={self.LEVEL}, HANDLERS={self.HANDLERS})"

    def remove_all_handlers(self, logger: logging.Logger = None):
        """
        Removes all the handlers from the logger. If the logger is None, it removes all the handlers
        from every logger in the package.

        :param logger: The logger to remove the handlers. Default is None.
        """
        if logger is None:
            for logger in self.loggers.values():
                if logger.hasHandlers():
                    for handler in logger.handlers:
                        logger.removeHandler(handler)
        else:
            if logger.hasHandlers():
                for handler in logger.handlers:
                    logger.removeHandler(handler)

        self.HANDLERS = []

    def add_handler(self, handler: logging.Handler, logger: logging.Logger = None):
        """
        Add a handler to the logger. If the logger is None, add the handler to every logger in the
        package.

        :param handler: The handler to add.
        :param logger: The logger to add the handler. Default is None.
        """
        if logger is None:
            for logger in self.loggers.values():
                logger.addHandler(handler)
        else:
            logger.addHandler(handler)

        self.HANDLERS.append(handler)

    def set_default_formatter(self, handler: logging.Handler = None):
        """
        Set the default formatter to the handler. If the handler is None, set the default formatter
        to every handler in the package.

        :param handler: The handler to set the formatter. Default is None.
        """
        if handler is None:
            for handler in self.HANDLERS:
                handler.setFormatter(FORMATTER)
        else:
            handler.setFormatter(FORMATTER)

    @staticmethod
    def raise_warning(
        msg: str,
        logger: logging.Logger,
        warning_category: Type[Warning] = None,
        stacklevel: int = 1,
        level: int = WARNING,
        **kwargs
    ) -> None:
        """
        Raise a warning with the logger and raise a warning with the warning category.

        :param logger: The logger instance.
        :param warning_category: The warning category to raise.
        :param msg: The message of the warning.
        :param level: The level of the logger. Default is WARNING.
        :param stacklevel: The stack level of the warning. Default is 1.
        :param kwargs: Other arguments to pass to the warning.
        :return: None
        """
        logger.log(level, msg)
        warnings.warn(msg, warning_category, stacklevel=stacklevel, **kwargs)

    @staticmethod
    def raise_error(
        msg: str,
        logger: logging.Logger,
        error_category: Type[Exception],
        level: int = ERROR,
        **kwargs
    ) -> None:
        """
        Raise an error with the logger and raise an error with the error category.

        :param logger: The logger instance.
        :param error_category: The error category to raise.
        :param msg: The message of the error.
        :param level: The level of the logger. Default is ERROR.
        :param kwargs: Other arguments to pass to the error.
        :return: None
        """
        logger.log(level, msg)
        raise error_category(msg, **kwargs)


# Create the (single) instance of LoggerConfiguration
log_config = LoggerConfiguration()


# Alias of the methods in the instance of LoggerConfiguration
# noinspection PyMissingOrEmptyDocstring
def get_logger(
    name: str, level: int = log_config.LEVEL, handlers: list[logging.Handler] = None
) -> logging.Logger:
    return log_config.get_logger(name, level, handlers)


# noinspection PyMissingOrEmptyDocstring
def set_level(level: int, name: str = None):
    log_config.set_level(level, name)


# noinspection PyMissingOrEmptyDocstring
def raise_warning(
    msg: str,
    logger: logging.Logger,
    warning_category: Type[Warning] = None,
    stacklevel: int = 1,
    level: int = WARNING,
    **kwargs
) -> None:
    log_config.raise_warning(msg, logger, warning_category, stacklevel, level, **kwargs)


# noinspection PyMissingOrEmptyDocstring
def raise_error(
    msg: str,
    logger: logging.Logger,
    error_category: Type[Exception],
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
    """
    dt = timedelta(seconds=seconds)
    dt_fmt = str(dt)
    if dt.days > 0:  # Return in the format 'days, hours:min'
        return dt_fmt[:-10]
    if dt.seconds > 3600:  # Return in the format 'hours:min:sec'
        return dt_fmt[:-7]
    if dt.seconds > 60:  # Return in the format 'hours:min:sec.microsec (rounded)'
        return dt_fmt[:-4]
    # Return in the format 'hours:min:sec.microsec'
    return dt_fmt


class register_total_time:
    """
    Class that registers the total time it takes to execute a piece of code, and shows it with the
    logger. It can be used as a context manager or as a decorator.
    """

    def __init__(self, logger: logging.Logger, level: int = logging.DEBUG):
        self.logger = logger
        self.level = level
        self.tic = time.perf_counter()
        self.toc = time.perf_counter()

    @property
    def elapsed_time(self):
        """
        The elapsed time between the start and the end of the block of code.

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
        Decorator to register the total time it takes to execute a function, and shows it with the
        logger.

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


def register_total_time_function(logger: logging.Logger, level: int = logging.DEBUG):
    """
    Wrapper that records the total time it takes to execute a function, and shows it with the
    logger.

    :param logger: A `Logger` instance
    :param level: The level of the logger
    :return: The decorator
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


def register_total_time_method(logger: logging.Logger, level: int = logging.DEBUG):
    """
    Wrapper that records the total time it takes to execute a method, and shows it with the
    logger.

    :param logger: A `Logger` instance
    :param level: The level of the logger
    :return: The decorator
    """

    # noinspection PyMissingOrEmptyDocstring
    def decorator(method):
        # noinspection PyMissingOrEmptyDocstring
        @functools.wraps(method)
        def wrapper(*args, **kwargs):
            self: object = args[0]
            logger.log(level, f"Using the method '{self.__class__.__name__}.{method.__name__}'...")
            tic = time.perf_counter()
            result = method(*args, **kwargs)
            toc = time.perf_counter()
            dt_fmt = _time_fmt(toc - tic)
            logger.log(
                level,
                f"The method '{self.__class__.__name__}.{method.__name__}' takes {dt_fmt}"
            )
            return result

        return wrapper

    return decorator


def register_init_method(logger: logging.Logger, level: int = logging.DEBUG):
    """
    Logs the use of a method, indicating the name of the method and the name of the class.

    :param logger: A `Logger` instance
    :param level: The level of the logger
    :return: The decorator
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
                f"Using the method '{method_name}' in the class '{class_name}'."
            )
            # Compute the result
            result = method(*args, **kwargs)
            return result

        return wrapper

    return decorator


def __main():
    from icecream import ic

    _log = log_config.get_logger(__name__)

    # Change to debug
    log_config.set_level(logging.DEBUG)

    ic("Testing class context manager")
    with register_total_time(_log):
        time.sleep(1)

    ic("Testing class context manager part 2")
    with register_total_time(_log) as timer:
        time.sleep(1)

    ic(timer.elapsed_time)

    ic("Testing function decorator with 'register_total_time_function'")

    @register_total_time_function(_log)
    def test_function():
        time.sleep(1)

    test_function()

    ic("Testing function decorator with 'register_total_time'")

    @register_total_time(_log)
    def test_function():
        time.sleep(1)

    test_function()

    # Test method decorator
    ic("Testing method decorator with 'register_total_time_method'")

    class Test:
        @register_total_time_method(_log, logging.WARNING)
        def test_method(self):
            time.sleep(1)

    Test().test_method()


if __name__ == "__main__":
    __main()
