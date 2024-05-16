"""Template Extensions for Home Assistant."""

import base64
import collections
from collections.abc import Iterable
from contextvars import ContextVar
from datetime import date, datetime, timedelta, time
from functools import lru_cache
import json
import logging
import math
from operator import contains
import random
import re
import statistics
from struct import error as StructError, pack, unpack_from
from typing import Any, NoReturn, Optional, TypeVar, Union
from urllib.parse import urlencode as urllib_urlencode

from awesomeversion import AwesomeVersion
import jinja2
from jinja2 import Environment
from jinja2.filters import do_max, do_min
import orjson
import voluptuous as vol

from homeassistant.const import (
    ATTR_ENTITY_ID,
    ATTR_LATITUDE,
    ATTR_LONGITUDE,
    ATTR_PERSONS,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
    UnitOfLength,
)
from homeassistant.core import HomeAssistant, State, valid_entity_id
from homeassistant.exceptions import TemplateError
from homeassistant.helpers import (
    area_registry as ar,
    config_validation as cv,
    device_registry as dr,
    entity as entity_helper,
    entity_registry as er,
    floor_registry as fr,
    issue_registry as ir,
    label_registry as lr,
    location as loc_helper,
)
from homeassistant.helpers.template import (
    AllStates,
    RenderInfo,
    StateTranslated,
    TemplateState,
    _get_template_state_from_state,
    device_attr,
)
from homeassistant.helpers.template_extensions import (
    FeatureCategories,
    TemplateExtension,
)
from homeassistant.util import (
    convert,
    dt as dt_util,
    location as loc_util,
    slugify as slugify_util,
)
from homeassistant.util.json import JsonValueType, json_loads

_LOGGER = logging.getLogger(__name__)
_SENTINEL = object()


class BaseTemplateExtension(TemplateExtension):
    """A base class for template extensions in Home Assistant."""

    template_cv: ContextVar[tuple[str, str] | None] = ContextVar(
        "template_cv", default=None
    )

    def raise_no_default(self, function: str, value: Any) -> NoReturn:
        """Log warning if no default is specified."""
        template, action = self.template_cv.get() or ("", "rendering or compiling")
        raise ValueError(
            f"Template error: {function} got invalid input '{value}' when {action} template"
            f" '{template}' but no default was specified"
        )


class RegexTemplateExtension(BaseTemplateExtension):
    """A template extension that adds regex functions."""

    def __init__(
        self,
        environment: Environment,
        hass: HomeAssistant,
        limited: bool,
        strict: bool,
        render_info: RenderInfo,
    ) -> None:
        """Initialize the regex template extension."""
        super().__init__(environment, hass, limited, strict, render_info)
        self._regex_cache = lru_cache(maxsize=128)(re.compile)

    @staticmethod
    def extension_id() -> str:
        """Return the extension ID."""
        return "homeassistant.regex"

    @staticmethod
    def component() -> str | None:
        """Return the component name."""
        return "homeassistant"

    @TemplateExtension.filter(
        name="regex_match",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.REGEX.value],
    )
    @TemplateExtension.test(
        name="match",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.REGEX.value],
    )
    def regex_match(self, value: Any, find: str = "", ignorecase: bool = False) -> bool:
        """Match value using regex."""
        if not isinstance(value, str):
            value = str(value)
        flags = re.I if ignorecase else 0
        return bool(self._regex_cache(find, flags).match(value))

    @TemplateExtension.filter(
        name="regex_replace",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.REGEX.value],
    )
    def regex_replace(
        self,
        value: Any = "",
        find: str = "",
        replace: Any = "",
        ignorecase: bool = False,
    ) -> str:
        """Replace using regex."""
        if not isinstance(value, str):
            value = str(value)
        flags = re.I if ignorecase else 0
        return str(self._regex_cache(find, flags).sub(replace, value))

    @TemplateExtension.filter(
        name="regex_search",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.REGEX.value],
    )
    @TemplateExtension.test(
        name="search",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.REGEX.value],
    )
    def regex_search(
        self, value: Any, find: str = "", ignorecase: bool = False
    ) -> bool:
        """Search using regex."""
        if not isinstance(value, str):
            value = str(value)
        flags = re.I if ignorecase else 0
        return bool(self._regex_cache(find, flags).search(value))

    @TemplateExtension.filter(
        name="regex_findall",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.REGEX.value],
    )
    def regex_findall(
        self, value: Any, find: str = "", ignorecase: bool = False
    ) -> list[Any]:
        """Find all matches using regex."""
        if not isinstance(value, str):
            value = str(value)
        flags = re.I if ignorecase else 0
        return self._regex_cache(find, flags).findall(value)

    @TemplateExtension.filter(
        name="regex_findall_index",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.REGEX.value],
    )
    def regex_findall_index(
        self, value: Any, find: str = "", index: int = 0, ignorecase: bool = False
    ) -> Any:
        """Find all matches using regex and then pick specific match index."""
        return self.regex_findall(value, find, ignorecase)[index]


class MathTemplateExtension(BaseTemplateExtension):
    """A template extension that adds math functions."""

    @staticmethod
    def extension_id() -> str:
        """Return the extension ID."""
        return "homeassistant.math"

    @staticmethod
    def component() -> str | None:
        """Return the component name."""
        return "homeassistant"

    @TemplateExtension.filter(
        name="round",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.MATH.value],
    )
    def forgiving_round(
        self,
        value: Any,
        precision: int = 0,
        method: str = "common",
        default: Any = _SENTINEL,
    ) -> int | float | Any:
        """Filter to round a value."""
        try:
            # support rounding methods like jinja
            multiplier = float(10**precision)
            if method == "ceil":
                value = math.ceil(float(value) * multiplier) / multiplier
            elif method == "floor":
                value = math.floor(float(value) * multiplier) / multiplier
            elif method == "half":
                value = round(float(value) * 2) / 2
            else:
                # if method is common or something else, use common rounding
                value = round(float(value), precision)
            return int(value) if precision == 0 else value
        except (ValueError, TypeError):
            # If value can't be converted to float
            if default is _SENTINEL:
                self.raise_no_default("round", value)
            return default

    @TemplateExtension.filter(
        name="multiply",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.MATH.value],
    )
    def multiply(
        self, value: Any, amount: float, default: Any = _SENTINEL
    ) -> int | float | Any:
        """Filter to convert value to float and multiply it."""
        try:
            return float(value) * amount
        except (ValueError, TypeError):
            # If value can't be converted to float
            if default is _SENTINEL:
                self.raise_no_default("multiply", value)
            return default

    @TemplateExtension.filter(
        name="log",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.MATH.value],
    )
    @TemplateExtension.function(
        name="log",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.MATH.value],
    )
    def logarithm(
        self, value: Any, base: float = math.e, default: Any = _SENTINEL
    ) -> int | float | Any:
        """Filter and function to get logarithm of the value with a specific base."""
        try:
            base_float = float(base)
        except (ValueError, TypeError):
            if default is _SENTINEL:
                self.raise_no_default("log", base)
            return default
        try:
            value_float = float(value)
            return math.log(value_float, base_float)
        except (ValueError, TypeError):
            if default is _SENTINEL:
                self.raise_no_default("log", value)
            return default

    @TemplateExtension.filter(
        name="sin",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.MATH.value],
    )
    @TemplateExtension.function(
        name="sin",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.MATH.value],
    )
    def sine(self, value: Any, default: Any = _SENTINEL) -> int | float | Any:
        """Filter and function to get sine of the value."""
        try:
            return math.sin(float(value))
        except (ValueError, TypeError):
            if default is _SENTINEL:
                self.raise_no_default("sin", value)
            return default

    @TemplateExtension.filter(
        name="cos",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.MATH.value],
    )
    @TemplateExtension.function(
        name="cos",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.MATH.value],
    )
    def cosine(self, value: Any, default: Any = _SENTINEL) -> int | float | Any:
        """Filter and function to get cosine of the value."""
        try:
            return math.cos(float(value))
        except (ValueError, TypeError):
            if default is _SENTINEL:
                self.raise_no_default("cos", value)
            return default

    @TemplateExtension.filter(
        name="tan",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.MATH.value],
    )
    @TemplateExtension.function(
        name="tan",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.MATH.value],
    )
    def tangent(self, value: Any, default: Any = _SENTINEL) -> int | float | Any:
        """Filter and function to get tangent of the value."""
        try:
            return math.tan(float(value))
        except (ValueError, TypeError):
            if default is _SENTINEL:
                self.raise_no_default("tan", value)
            return default

    @TemplateExtension.filter(
        name="asin",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.MATH.value],
    )
    @TemplateExtension.function(
        name="asin",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.MATH.value],
    )
    def arc_sine(self, value: Any, default: Any = _SENTINEL) -> int | float | Any:
        """Filter and function to get arc sine of the value."""
        try:
            return math.asin(float(value))
        except (ValueError, TypeError):
            if default is _SENTINEL:
                self.raise_no_default("asin", value)
            return default

    @TemplateExtension.filter(
        name="acos",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.MATH.value],
    )
    @TemplateExtension.function(
        name="acos",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.MATH.value],
    )
    def arc_cosine(self, value: Any, default: Any = _SENTINEL) -> int | float | Any:
        """Filter and function to get arc cosine of the value."""
        try:
            return math.acos(float(value))
        except (ValueError, TypeError):
            if default is _SENTINEL:
                self.raise_no_default("acos", value)
            return default

    @TemplateExtension.filter(
        name="atan",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.MATH.value],
    )
    @TemplateExtension.function(
        name="atan",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.MATH.value],
    )
    def arc_tangent(self, value: Any, default: Any = _SENTINEL) -> int | float | Any:
        """Filter and function to get arc tangent of the value."""
        try:
            return math.atan(float(value))
        except (ValueError, TypeError):
            if default is _SENTINEL:
                self.raise_no_default("atan", value)
            return default

    @TemplateExtension.filter(
        name="atan2",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.MATH.value],
    )
    @TemplateExtension.function(
        name="atan2",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.MATH.value],
    )
    def arc_tangent2(self, *args: Any, default: Any = _SENTINEL) -> int | float | Any:
        """Filter and function to calculate four quadrant arc tangent of y / x.

        The parameters to atan2 may be passed either in an iterable or as separate arguments
        The default value may be passed either as a positional or in a keyword argument
        """
        try:
            if 1 <= len(args) <= 2 and isinstance(args[0], (list, tuple)):
                if len(args) == 2 and default is _SENTINEL:
                    # Default value passed as a positional argument
                    default = args[1]
                args = args[0]  # type: ignore[assignment]
            elif len(args) == 3 and default is _SENTINEL:
                # Default value passed as a positional argument
                default = args[2]

            return math.atan2(float(args[0]), float(args[1]))
        except (ValueError, TypeError):
            if default is _SENTINEL:
                self.raise_no_default("atan2", args)
            return default

    @TemplateExtension.filter(
        name="sqrt",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.MATH.value],
    )
    @TemplateExtension.function(
        name="sqrt",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.MATH.value],
    )
    def square_root(self, value: Any, default: Any = _SENTINEL) -> int | float | Any:
        """Filter and function to get square root of the value."""
        try:
            return math.sqrt(float(value))
        except (ValueError, TypeError):
            if default is _SENTINEL:
                self.raise_no_default("sqrt", value)
            return default

    @TemplateExtension.function(
        name="pi",
        limited=True,
        passthrough=True,
        cacheable=True,
        categories=[FeatureCategories.MATH.value],
    )
    def pi(self) -> float:
        """Return the value of pi."""
        return math.pi

    @TemplateExtension.function(
        name="tau",
        limited=True,
        passthrough=True,
        cacheable=True,
        categories=[FeatureCategories.MATH.value],
    )
    def tau(self) -> float:
        """Return the value of tau."""
        return math.pi * 2

    @TemplateExtension.function(
        name="e",
        limited=True,
        passthrough=True,
        cacheable=True,
        categories=[FeatureCategories.MATH.value],
    )
    def e(self) -> float:
        """Return the value of e."""
        return math.e

    @TemplateExtension.filter(
        name="random",
        limited=True,
        categories=[FeatureCategories.MATH.value],
    )
    def random_every_time(self, values: Any) -> Any:
        """Choose a random value.

        Unlike Jinja's random filter,
        this is context-dependent to avoid caching the chosen value.
        """
        return random.choice(values)

    @TemplateExtension.filter(
        name="add",
        limited=True,
        categories=[FeatureCategories.MATH.value],
    )
    def add(self, value, amount, default=_SENTINEL) -> Any:
        """Filter to convert value to float and add it."""
        try:
            return float(value) + amount
        except (ValueError, TypeError):
            # If value can't be converted to float
            if default is _SENTINEL:
                self.raise_no_default("add", value)
            return default


class StatisticTemplateExtension(BaseTemplateExtension):
    """A template extension that adds statistics functions."""

    @staticmethod
    def extension_id() -> str:
        """Return the extension ID."""
        return "homeassistant.statistics"

    @staticmethod
    def component() -> str | None:
        """Return the component name."""
        return "homeassistant"

    @TemplateExtension.filter(
        name="average",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.STASTISTICS.value],
    )
    @TemplateExtension.function(
        name="average",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.STASTISTICS.value],
    )
    def average(self, *args: Any, default: Any = _SENTINEL) -> int | float | Any:
        """Filter and function to calculate the arithmetic mean.

        Calculates of an iterable or of two or more arguments.
        The parameters may be passed as an iterable or as separate arguments.
        """
        if len(args) == 0:
            raise TypeError("average expected at least 1 argument, got 0")
        # If first argument is iterable and more than 1 argument provided but not a named
        # default, then use 2nd argument as default.
        if isinstance(args[0], Iterable):
            average_list = args[0]
            if len(args) > 1 and default is _SENTINEL:
                default = args[1]
        elif len(args) == 1:
            raise TypeError(f"'{type(args[0]).__name__}' object is not iterable")
        else:
            average_list = args
        try:
            return statistics.fmean(average_list)
        except (TypeError, statistics.StatisticsError):
            if default is _SENTINEL:
                self.raise_no_default("average", args)
            return default

    @TemplateExtension.filter(
        name="median",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.STASTISTICS.value],
    )
    @TemplateExtension.function(
        name="median",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.STASTISTICS.value],
    )
    def median(self, *args: Any, default: Any = _SENTINEL) -> int | float | Any:
        """Filter and function to calculate the median.

        Calculates median of an iterable of two or more arguments.
        The parameters may be passed as an iterable or as separate arguments.
        """
        if len(args) == 0:
            raise TypeError("median expected at least 1 argument, got 0")
        # If first argument is a list or tuple and more than 1 argument provided but not a named
        # default, then use 2nd argument as default.
        if isinstance(args[0], Iterable):
            median_list = args[0]
            if len(args) > 1 and default is _SENTINEL:
                default = args[1]
        elif len(args) == 1:
            raise TypeError(f"'{type(args[0]).__name__}' object is not iterable")
        else:
            median_list = args
        try:
            return statistics.median(median_list)
        except (TypeError, statistics.StatisticsError):
            if default is _SENTINEL:
                self.raise_no_default("median", args)
            return default

    @TemplateExtension.filter(
        name="statistical_mode",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.STASTISTICS.value],
    )
    @TemplateExtension.function(
        name="statistical_mode",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.STASTISTICS.value],
    )
    def statistical_mode(self, *args: Any, default: Any = _SENTINEL) -> Any:
        """Filter and function to calculate the statistical mode.

        Calculates mode of an iterable of two or more arguments.
        The parameters may be passed as an iterable or as separate arguments.
        """
        if not args:
            raise TypeError("statistical_mode expected at least 1 argument, got 0")
        # If first argument is a list or tuple and more than 1 argument provided but not a named
        # default, then use 2nd argument as default.
        if len(args) == 1 and isinstance(args[0], Iterable):
            mode_list = args[0]
        elif isinstance(args[0], list | tuple):
            mode_list = args[0]
            if len(args) > 1 and default is _SENTINEL:
                default = args[1]
        elif len(args) == 1:
            raise TypeError(f"'{type(args[0]).__name__}' object is not iterable")
        else:
            mode_list = args
        try:
            return statistics.mode(mode_list)
        except (TypeError, statistics.StatisticsError):
            if default is _SENTINEL:
                self.raise_no_default("statistical_mode", args)
            return default

    @TemplateExtension.function(
        name="min",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.STASTISTICS.value],
    )
    def min(
        self,
        value: Iterable,
        case_sensitive: bool = False,
        attribute: Optional[Union[str, int]] = None,
    ) -> Any:
        """Convert a built-in min Jinja filter to a global function."""
        return do_min(self.environment, value, case_sensitive, attribute)

    @TemplateExtension.function(
        name="max",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.STASTISTICS.value],
    )
    def max(
        self,
        value: Iterable,
        case_sensitive: bool = False,
        attribute: Optional[Union[str, int]] = None,
    ) -> Any:
        """Convert a built-in min Jinja filter to a global function."""
        return do_max(self.environment, value, case_sensitive, attribute)


class DatetimeTemplateExtension(BaseTemplateExtension):
    """A template extension that adds datetime functions."""

    DATE_STR_FORMAT = "%Y-%m-%d %H:%M:%S"

    @staticmethod
    def extension_id() -> str:
        """Return the extension ID."""
        return "homeassistant.datetime"

    @staticmethod
    def component() -> str | None:
        """Return the component name."""
        return "homeassistant"

    @TemplateExtension.filter(
        name="as_datetime",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.DATETIME.value],
    )
    @TemplateExtension.function(
        name="as_datetime",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.DATETIME.value],
    )
    def as_datetime(self, value: Any, default: Any = _SENTINEL) -> Any:
        """Filter and to convert a time string or UNIX timestamp to datetime object."""
        # Return datetime.datetime object without changes
        if type(value) is datetime:
            return value
        # Add midnight to datetime.date object
        if type(value) is date:
            return datetime.combine(value, time(0, 0, 0))
        try:
            # Check for a valid UNIX timestamp string, int or float
            timestamp = float(value)
            return dt_util.utc_from_timestamp(timestamp)
        except (ValueError, TypeError):
            # Try to parse datetime string to datetime object
            try:
                return dt_util.parse_datetime(value, raise_on_error=True)
            except (ValueError, TypeError):
                if default is _SENTINEL:
                    # Return None on string input
                    # to ensure backwards compatibility with HA Core 2024.1 and before.
                    if isinstance(value, str):
                        return None
                    self.raise_no_default("as_datetime", value)
                return default

    @TemplateExtension.filter(
        name="as_timedelta",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.DATETIME.value],
    )
    @TemplateExtension.function(
        name="as_timedelta",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.DATETIME.value],
    )
    def as_timedelta(self, value: str) -> timedelta | None:
        """Parse a ISO8601 duration like 'PT10M' to a timedelta."""
        return dt_util.parse_duration(value)

    @TemplateExtension.filter(
        name="as_timestamp",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.DATETIME.value],
    )
    @TemplateExtension.function(
        name="as_timestamp",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.DATETIME.value],
    )
    def forgiving_as_timestamp(
        self, value: datetime | str, default: Any = _SENTINEL
    ) -> float | Any:
        """Filter and function which tries to convert value to timestamp."""
        try:
            return dt_util.as_timestamp(value)
        except (ValueError, TypeError):
            if default is _SENTINEL:
                self.raise_no_default("as_timestamp", value)
            return default

    @TemplateExtension.filter(
        name="as_local",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.DATETIME.value],
    )
    @TemplateExtension.function(
        name="as_local",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.DATETIME.value],
    )
    def as_local(self, value: datetime, default: Any = _SENTINEL) -> datetime | Any:
        """Convert a UTC datetime object to local time zone."""
        try:
            return dt_util.as_local(value)
        except (ValueError, TypeError):
            if default is _SENTINEL:
                self.raise_no_default("as_local", value)
            return default

    @TemplateExtension.filter(
        name="timestamp_custom",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.DATETIME.value],
    )
    def timestamp_custom(
        self,
        value: Any,
        date_format: str = DATE_STR_FORMAT,
        local: bool = True,
        default: Any = _SENTINEL,
    ) -> str | Any:
        """Filter to convert given timestamp to format."""
        try:
            result = dt_util.utc_from_timestamp(value)
            if local:
                result = dt_util.as_local(result)
            return result.strftime(date_format)
        except (ValueError, TypeError):
            # If timestamp can't be converted
            if default is _SENTINEL:
                self.raise_no_default("timestamp_custom", value)
            return default

    @TemplateExtension.filter(
        name="timestamp_local",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.DATETIME.value],
    )
    def timestamp_local(self, value: Any, default: Any = _SENTINEL) -> datetime | Any:
        """Filter to convert given timestamp to local date/time."""
        try:
            return dt_util.as_local(dt_util.utc_from_timestamp(value)).isoformat()
        except (ValueError, TypeError):
            # If timestamp can't be converted
            if default is _SENTINEL:
                self.raise_no_default("timestamp_local", value)
            return default

    @TemplateExtension.filter(
        name="timestamp_utc",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.DATETIME.value],
    )
    def timestamp_utc(self, value: Any, default: Any = _SENTINEL) -> str | Any:
        """Filter to convert given timestamp to UTC date/time."""
        try:
            return dt_util.utc_from_timestamp(value).isoformat()
        except (ValueError, TypeError):
            # If timestamp can't be converted
            if default is _SENTINEL:
                self.raise_no_default("timestamp_utc", value)
            return default

    @TemplateExtension.function(
        name="timedelta",
        limited=True,
        cacheable=True,
        passthrough=True,
        categories=[FeatureCategories.DATETIME.value],
    )
    def timedelta(self) -> timedelta:  # type: ignore[valid-type]
        """Return a timedelta object."""
        return timedelta

    @TemplateExtension.function(
        name="strptime",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.DATETIME.value],
    )
    def strptime(
        self, string: str, fmt: str, default: Any = _SENTINEL
    ) -> datetime | Any:
        """Parse a time string to datetime."""
        try:
            return datetime.strptime(string, fmt)
        except (ValueError, AttributeError, TypeError):
            if default is _SENTINEL:
                self.raise_no_default("strptime", string)
            return default

    @TemplateExtension.test(
        name="datetime",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.DATETIME.value],
    )
    def _is_datetime(self, value: Any) -> bool:
        """Return whether a value is a datetime."""
        return isinstance(value, datetime)


class DatatypeTemplateExtension(BaseTemplateExtension):
    """A template extension that adds datatype functions."""

    _T = TypeVar("_T")

    @staticmethod
    def extension_id() -> str:
        """Return the extension ID."""
        return "homeassistant.datatype"

    @staticmethod
    def component() -> str | None:
        """Return the component name."""
        return "homeassistant"

    @TemplateExtension.filter(
        name="is_number",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.DATATYPE.value],
    )
    @TemplateExtension.function(
        name="is_number",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.DATATYPE.value],
    )
    @TemplateExtension.test(
        name="number",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.DATATYPE.value],
    )
    def is_number(self, value: Any) -> bool:
        """Try to convert value to a float."""
        try:
            fvalue = float(value)
        except (ValueError, TypeError):
            return False
        if not math.isfinite(fvalue):
            return False
        return True

    @TemplateExtension.filter(
        name="bool",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.DATATYPE.value],
    )
    @TemplateExtension.function(
        name="bool",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.DATATYPE.value],
    )
    def forgiving_boolean(
        self, value: Any, default: Union[_T, object] = _SENTINEL
    ) -> Union[bool, _T, object]:
        """Try to convert value to a boolean, return default if conversion fails."""
        try:
            return cv.boolean(value)
        except vol.Invalid:
            if default is _SENTINEL:
                self.raise_no_default("bool", value)
            return default

    @TemplateExtension.function(
        name="int",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.DATATYPE.value],
    )
    def forgiving_int(
        self, value: Any, default: Any = _SENTINEL, base: int = 10
    ) -> int | Any:
        """Try to convert value to an int, and raise if it fails."""
        result = jinja2.filters.do_int(value, default=default, base=base)
        if result is _SENTINEL:
            self.raise_no_default("int", value)
        return result

    @TemplateExtension.filter(
        name="int",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.DATATYPE.value],
    )
    def forgiving_int_filter(
        self, value: Any, default: Any = _SENTINEL, base: int = 10
    ) -> int | Any:
        """Try to convert value to an int, and raise if it fails."""
        result = jinja2.filters.do_int(value, default=default, base=base)
        if result is _SENTINEL:
            self.raise_no_default("int", value)
        return result

    @TemplateExtension.function(
        name="tuple",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.DATATYPE.value],
    )
    def to_tuple(self, value: Any) -> tuple[Any, ...]:
        """Convert value to tuple."""
        return tuple(value)

    @TemplateExtension.test(
        name="tuple",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.DATATYPE.value],
    )
    def is_tuple(self, value: Any) -> bool:
        """Return whether a value is a tuple."""
        return isinstance(value, tuple)

    @TemplateExtension.function(
        name="set",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.DATATYPE.value],
    )
    def to_set(self, value: Any) -> set[Any]:
        """Convert value to set."""
        return set(value)

    @TemplateExtension.test(
        name="set",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.DATATYPE.value],
    )
    def is_set(self, value: Any) -> bool:
        """Return whether a value is a set."""
        return isinstance(value, set)

    @TemplateExtension.function(
        name="float",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.DATATYPE.value],
    )
    def forgiving_float(self, value: Any, default: Any = _SENTINEL) -> float | Any:
        """Try to convert value to a float."""
        try:
            return float(value)
        except (ValueError, TypeError):
            if default is _SENTINEL:
                self.raise_no_default("float", value)
            return default

    @TemplateExtension.filter(
        name="float",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.DATATYPE.value],
    )
    def forgiving_float_filter(
        self, value: Any, default: Any = _SENTINEL
    ) -> float | Any:
        """Try to convert value to a float."""
        try:
            return float(value)
        except (ValueError, TypeError):
            if default is _SENTINEL:
                self.raise_no_default("float", value)
            return default

    @TemplateExtension.test(
        name="list",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.DATATYPE.value],
    )
    def is_list(self, value: Any) -> bool:
        """Return whether a value is a list."""
        return isinstance(value, list)

    @TemplateExtension.test(
        name="string_like",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.DATATYPE.value, FeatureCategories.STRING.value],
    )
    def _is_string_like(self, value: Any) -> bool:
        """Return whether a value is a string or string like object."""
        return isinstance(value, (str, bytes, bytearray))

    @TemplateExtension.filter(
        name="is_defined",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.DATATYPE.value],
    )
    def fail_when_undefined(self, value: Any) -> Any:
        """Filter to force a failure when the value is undefined."""
        if isinstance(value, jinja2.Undefined):
            value()
        return value


class StringTemplateExtension(BaseTemplateExtension):
    """A template extension that adds string functions."""

    ORJSON_PASSTHROUGH_OPTIONS = (
        orjson.OPT_PASSTHROUGH_DATACLASS | orjson.OPT_PASSTHROUGH_DATETIME
    )

    @staticmethod
    def extension_id() -> str:
        """Return the extension ID."""
        return "homeassistant.string"

    @staticmethod
    def component() -> str | None:
        """Return the component name."""
        return "homeassistant"

    @TemplateExtension.filter(
        name="slugify",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.STRING.value],
    )
    @TemplateExtension.function(
        name="slugify",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.STRING.value],
    )
    def slugify(self, value: Any, separator: str = "_") -> str:
        """Convert a string into a slug, such as what is used for entity ids."""
        return slugify_util(value, separator=separator)

    @TemplateExtension.function(
        name="urlencode",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.STRING.value],
    )
    def urlencode(self, value: Any) -> bytes:
        """Urlencode dictionary and return as UTF-8 string."""
        return urllib_urlencode(value).encode("utf-8")

    @TemplateExtension.filter(
        name="to_json",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.STRING.value],
    )
    def to_json(
        self,
        value: Any,
        ensure_ascii: bool = False,
        pretty_print: bool = False,
        sort_keys: bool = False,
    ) -> str:
        """Convert an object to a JSON string."""

        def to_json_default(obj: Any) -> None:
            """Disable custom types in json serialization."""
            raise TypeError(
                f"Object of type {type(obj).__name__} is not JSON serializable"
            )

        if ensure_ascii:
            # For those who need ascii, we can't use orjson, so we fall back to the json library.
            return json.dumps(
                value,
                ensure_ascii=ensure_ascii,
                indent=2 if pretty_print else None,
                sort_keys=sort_keys,
            )

        option = (
            self.ORJSON_PASSTHROUGH_OPTIONS
            # OPT_NON_STR_KEYS is added as a workaround to
            # ensure subclasses of str are allowed as dict keys
            # See: https://github.com/ijl/orjson/issues/445
            | orjson.OPT_NON_STR_KEYS
            | (orjson.OPT_INDENT_2 if pretty_print else 0)
            | (orjson.OPT_SORT_KEYS if sort_keys else 0)
        )

        return orjson.dumps(
            value,
            option=option,
            default=to_json_default,
        ).decode("utf-8")

    @TemplateExtension.filter(
        name="from_json",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.STRING.value],
    )
    def from_json(self, value: str) -> JsonValueType:
        """Convert a JSON string to an object."""
        return json_loads(value)

    @TemplateExtension.filter(
        name="base64_encode",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.STRING.value],
    )
    def base64_encode(self, value: str) -> str:
        """Perform base64 encode."""
        return base64.b64encode(value.encode("utf-8")).decode("utf-8")

    @TemplateExtension.filter(
        name="base64_decode",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.STRING.value],
    )
    def base64_decode(self, value: str) -> str:
        """Perform base64 denode."""
        return base64.b64decode(value).decode("utf-8")

    @TemplateExtension.filter(
        name="ordinal",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.STRING.value],
    )
    def ordinal(self, value: Any) -> str:
        """Perform ordinal conversion."""
        return str(value) + (
            list(["th", "st", "nd", "rd"] + ["th"] * 6)[(int(str(value)[-1])) % 10]
            if int(str(value)[-2:]) % 100 not in range(11, 14)
            else "th"
        )

    @TemplateExtension.filter(
        name="ord",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.STRING.value],
    )
    def ord(self, value: str) -> int:
        """Perform ordinal conversion."""
        return ord(value)

    @TemplateExtension.filter(
        name="contains",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.STRING.value],
    )
    @TemplateExtension.test(
        name="contains",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.STRING.value],
    )
    def contains(self, value: Any, search: Any) -> bool:
        """Check if value contains search."""
        return contains(value, search)

    @TemplateExtension.filter(
        name="version",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.STRING.value],
    )
    @TemplateExtension.function(
        name="version",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.STRING.value],
    )
    def version(self, value: Any) -> AwesomeVersion:
        """Filter and function to get version object of the value."""
        return AwesomeVersion(value)


class BinaryTemplateExtension(BaseTemplateExtension):
    """A template extension that adds binary functions."""

    @staticmethod
    def extension_id() -> str:
        """Return the extension ID."""
        return "homeassistant.binary"

    @staticmethod
    def component() -> str | None:
        """Return the component name."""
        return "homeassistant"

    @TemplateExtension.filter(
        name="bitwise_and",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.BINARY.value],
    )
    def bitwise_and(self, first_value: Any, second_value: Any) -> Any:
        """Perform a bitwise and operation."""
        return first_value & second_value

    @TemplateExtension.filter(
        name="bitwise_or",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.BINARY.value],
    )
    def bitwise_or(self, first_value: Any, second_value: Any) -> Any:
        """Perform a bitwise or operation."""
        return first_value | second_value

    @TemplateExtension.filter(
        name="bitwise_xor",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.BINARY.value],
    )
    def bitwise_xor(self, first_value: Any, second_value: Any) -> Any:
        """Perform a bitwise xor operation."""
        return first_value ^ second_value

    @TemplateExtension.filter(
        name="pack",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.BINARY.value],
    )
    @TemplateExtension.function(
        name="pack",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.BINARY.value],
    )
    def struct_pack(self, value: Any | None, format_string: str) -> bytes | None:
        """Pack an object into a bytes object."""
        try:
            return pack(format_string, value)
        except StructError:
            _LOGGER.warning(
                (
                    "Template warning: 'pack' unable to pack object '%s' with type '%s' and"
                    " format_string '%s' see https://docs.python.org/3/library/struct.html"
                    " for more information"
                ),
                str(value),
                type(value).__name__,
                format_string,
            )
            return None

    @TemplateExtension.filter(
        name="unpack",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.BINARY.value],
    )
    @TemplateExtension.function(
        name="unpack",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.BINARY.value],
    )
    def struct_unpack(
        self, value: bytes, format_string: str, offset: int = 0
    ) -> Any | None:
        """Unpack an object from bytes an return the first native object."""
        try:
            return unpack_from(format_string, value, offset)[0]
        except StructError:
            _LOGGER.warning(
                (
                    "Template warning: 'unpack' unable to unpack object '%s' with"
                    " format_string '%s' and offset %s see"
                    " https://docs.python.org/3/library/struct.html for more information"
                ),
                value,
                format_string,
                offset,
            )
            return None


class ControlFlowExtension(BaseTemplateExtension):
    """A template extension that adds control flow functions."""

    @staticmethod
    def extension_id() -> str:
        """Return the extension ID."""
        return "homeassistant.control_flow"

    @staticmethod
    def component() -> str | None:
        """Return the component name."""
        return "homeassistant"

    @TemplateExtension.filter(
        name="iif",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.CONTROL_FLOW.value],
    )
    @TemplateExtension.function(
        name="iif",
        limited=True,
        cacheable=True,
        categories=[FeatureCategories.CONTROL_FLOW.value],
    )
    def iif(
        self,
        value: Any,
        if_true: Any = True,
        if_false: Any = False,
        if_none: Any = _SENTINEL,
    ) -> Any:
        """Immediate if function/filter that allow for common if/else constructs.

        https://en.wikipedia.org/wiki/IIf

        Examples:
            {{ is_state("device_tracker.frenck", "home") | iif("yes", "no") }}
            {{ iif(1==2, "yes", "no") }}
            {{ (1 == 1) | iif("yes", "no") }}

        """
        if value is None and if_none is not _SENTINEL:
            return if_none
        if bool(value):
            return if_true
        return if_false


class CoreExtension(BaseTemplateExtension):
    """A template extension that adds core functions."""

    @staticmethod
    def extension_id() -> str:
        """Return the extension ID."""
        return "homeassistant.core"

    @staticmethod
    def component() -> str | None:
        """Return the component name."""
        return "homeassistant"

    @TemplateExtension.filter(
        name="device_entities",
        limited=True,
        categories=[FeatureCategories.CORE.value],
    )
    @TemplateExtension.function(
        name="device_entities",
        limited=True,
        categories=[FeatureCategories.CORE.value],
    )
    def device_entities(self, _device_id: str) -> Iterable[str]:
        """Get entity ids for entities tied to a device."""
        entity_reg = er.async_get(self.hass)
        entries = er.async_entries_for_device(entity_reg, _device_id)
        return [entry.entity_id for entry in entries]

    @TemplateExtension.filter(
        name="device_id",
        categories=[FeatureCategories.CORE.value],
    )
    @TemplateExtension.function(
        name="device_id",
        categories=[FeatureCategories.CORE.value],
    )
    def device_id(self, entity_id_or_device_name: str) -> str | None:
        """Get a device ID from an entity ID or device name."""
        entity_reg = er.async_get(self.hass)
        entity = entity_reg.async_get(entity_id_or_device_name)
        if entity is not None:
            return entity.device_id

        dev_reg = dr.async_get(self.hass)
        return next(
            (
                device_id
                for device_id, device in dev_reg.devices.items()
                if (name := device.name_by_user or device.name)
                and (str(entity_id_or_device_name) == name)
            ),
            None,
        )

    @TemplateExtension.filter(
        name="device_attr",
        categories=[FeatureCategories.CORE.value],
    )
    @TemplateExtension.function(
        name="device_attr",
        categories=[FeatureCategories.CORE.value],
    )
    def device_attr(self, device_or_entity_id: str, attr_name: str) -> Any:
        """Get the device specific attribute."""
        device_reg = dr.async_get(self.hass)
        if not isinstance(device_or_entity_id, str):
            raise TemplateError("Must provide a device or entity ID")
        device = None
        if (
            "." in device_or_entity_id
            and (_device_id := self.device_id(device_or_entity_id)) is not None
        ):
            device = device_reg.async_get(_device_id)
        elif "." not in device_or_entity_id:
            device = device_reg.async_get(device_or_entity_id)
        if device is None or not hasattr(device, attr_name):
            return None
        return getattr(device, attr_name)

    @TemplateExtension.test(
        name="is_device_attr",
        categories=[FeatureCategories.CORE.value],
    )
    @TemplateExtension.function(
        name="is_device_attr",
        categories=[FeatureCategories.CORE.value],
    )
    def is_device_attr(
        self, device_or_entity_id: str, attr_name: str, attr_value: Any
    ) -> bool:
        """Test if a device's attribute is a specific value."""
        return bool(
            device_attr(self.hass, device_or_entity_id, attr_name) == attr_value
        )

    @TemplateExtension.filter(
        name="config_entry_id",
        limited=True,
        categories=[FeatureCategories.CORE.value],
    )
    @TemplateExtension.function(
        name="config_entry_id",
        limited=True,
        categories=[FeatureCategories.CORE.value],
    )
    def config_entry_id(self, entity_id: str) -> str | None:
        """Get an config entry ID from an entity ID."""
        entity_reg = er.async_get(self.hass)
        if entity := entity_reg.async_get(entity_id):
            return entity.config_entry_id
        return None

    @TemplateExtension.function(
        name="issues",
        limited=True,
        categories=[FeatureCategories.CORE.value],
    )
    def issues(self) -> dict[tuple[str, str], dict[str, Any]]:
        """Return all open issues."""
        current_issues = ir.async_get(self.hass).issues
        # Use JSON for safe representation
        return {k: v.to_json() for (k, v) in current_issues.items()}

    @TemplateExtension.filter(
        name="issue",
        limited=True,
        categories=[FeatureCategories.CORE.value],
    )
    @TemplateExtension.function(
        name="issue",
        limited=True,
        categories=[FeatureCategories.CORE.value],
    )
    def issue(self, domain: str, issue_id: str) -> dict[str, Any] | None:
        """Get issue by domain and issue_id."""
        result = ir.async_get(self.hass).async_get_issue(domain, issue_id)
        if result:
            return result.to_json()
        return None

    @TemplateExtension.function(
        name="areas",
        limited=True,
        categories=[FeatureCategories.CORE.value],
    )
    def areas(self) -> Iterable[str | None]:
        """Return all areas."""
        return list(ar.async_get(self.hass).areas)

    @TemplateExtension.filter(
        name="area_id",
        categories=[FeatureCategories.CORE.value],
    )
    @TemplateExtension.function(
        name="area_id",
        categories=[FeatureCategories.CORE.value],
    )
    def area_id(self, lookup_value: str) -> str | None:
        """Get the area ID from an area name, device id, or entity id."""
        area_reg = ar.async_get(self.hass)
        if area := area_reg.async_get_area_by_name(str(lookup_value)):
            return area.id

        ent_reg = er.async_get(self.hass)
        dev_reg = dr.async_get(self.hass)

        try:
            cv.entity_id(lookup_value)
        except vol.Invalid:
            pass
        else:
            if entity := ent_reg.async_get(lookup_value):
                # If entity has an area ID, return that
                if entity.area_id:
                    return entity.area_id
                # If entity has a device ID, return the area ID for the device
                if entity.device_id and (device := dev_reg.async_get(entity.device_id)):
                    return device.area_id

        # Check if this could be a device ID
        if device := dev_reg.async_get(lookup_value):
            return device.area_id

        return None

    @TemplateExtension.filter(
        name="area_name",
        categories=[FeatureCategories.CORE.value],
    )
    @TemplateExtension.function(
        name="area_name",
        categories=[FeatureCategories.CORE.value],
    )
    def area_name(self, lookup_value: str) -> str | None:
        """Get the area name from an area id, device id, or entity id."""

        def get_area_name(area_reg: ar.AreaRegistry, valid_area_id: str) -> str:
            """Get area name from valid area ID."""
            area = area_reg.async_get_area(valid_area_id)
            assert area
            return area.name

        area_reg = ar.async_get(self.hass)
        if area := area_reg.async_get_area(lookup_value):
            return area.name

        dev_reg = dr.async_get(self.hass)
        ent_reg = er.async_get(self.hass)

        try:
            cv.entity_id(lookup_value)
        except vol.Invalid:
            pass
        else:
            if entity := ent_reg.async_get(lookup_value):
                # If entity has an area ID, get the area name for that
                if entity.area_id:
                    return get_area_name(area_reg, entity.area_id)
                # If entity has a device ID and the device exists with an area ID, get the
                # area name for that
                if (
                    entity.device_id
                    and (device := dev_reg.async_get(entity.device_id))
                    and device.area_id
                ):
                    return get_area_name(area_reg, device.area_id)

        if (device := dev_reg.async_get(lookup_value)) and device.area_id:
            return get_area_name(area_reg, device.area_id)

        return None

    @TemplateExtension.filter(
        name="area_entities",
        limited=True,
        categories=[FeatureCategories.CORE.value],
    )
    @TemplateExtension.function(
        name="area_entities",
        limited=True,
        categories=[FeatureCategories.CORE.value],
    )
    def area_entities(self, area_id_or_name: str) -> Iterable[str]:
        """Return entities for a given area ID or name."""
        _area_id: str | None
        # if area_name returns a value, we know the input was an ID, otherwise we
        # assume it's a name, and if it's neither, we return early
        if self.area_name(area_id_or_name) is None:
            _area_id = self.area_id(area_id_or_name)
        else:
            _area_id = area_id_or_name
        if _area_id is None:
            return []
        ent_reg = er.async_get(self.hass)
        entity_ids = [
            entry.entity_id for entry in er.async_entries_for_area(ent_reg, _area_id)
        ]
        dev_reg = dr.async_get(self.hass)
        # We also need to add entities tied to a device in the area that don't themselves
        # have an area specified since they inherit the area from the device.
        entity_ids.extend(
            [
                entity.entity_id
                for device in dr.async_entries_for_area(dev_reg, _area_id)
                for entity in er.async_entries_for_device(ent_reg, device.id)
                if entity.area_id is None
            ]
        )
        return entity_ids

    @TemplateExtension.filter(
        name="area_devices",
        limited=True,
        categories=[FeatureCategories.CORE.value],
    )
    @TemplateExtension.function(
        name="area_devices",
        limited=True,
        categories=[FeatureCategories.CORE.value],
    )
    def area_devices(self, area_id_or_name: str) -> Iterable[str]:
        """Return device IDs for a given area ID or name."""
        _area_id: str | None
        # if area_name returns a value, we know the input was an ID, otherwise we
        # assume it's a name, and if it's neither, we return early
        if self.area_name(area_id_or_name) is not None:
            _area_id = area_id_or_name
        else:
            _area_id = self.area_id(area_id_or_name)
        if _area_id is None:
            return []
        dev_reg = dr.async_get(self.hass)
        entries = dr.async_entries_for_area(dev_reg, _area_id)
        return [entry.id for entry in entries]

    @TemplateExtension.function(
        name="floors",
        limited=True,
        categories=[FeatureCategories.CORE.value],
    )
    def floors(self) -> Iterable[str | None]:
        """Return all floors."""
        floor_registry = fr.async_get(self.hass)
        return [floor.floor_id for floor in floor_registry.async_list_floors()]

    @TemplateExtension.filter(
        name="floor_id",
        categories=[FeatureCategories.CORE.value],
    )
    @TemplateExtension.function(
        name="floor_id",
        categories=[FeatureCategories.CORE.value],
    )
    def floor_id(self, lookup_value: Any) -> str | None:
        """Get the floor ID from a floor name."""
        floor_registry = fr.async_get(self.hass)
        if floor := floor_registry.async_get_floor_by_name(str(lookup_value)):
            return floor.floor_id

        if aid := self.area_id(lookup_value):
            area_reg = ar.async_get(self.hass)
            if area := area_reg.async_get_area(aid):
                return area.floor_id

        return None

    @TemplateExtension.filter(
        name="floor_name",
        categories=[FeatureCategories.CORE.value],
    )
    @TemplateExtension.function(
        name="floor_name",
        categories=[FeatureCategories.CORE.value],
    )
    def floor_name(self, lookup_value: str) -> str | None:
        """Get the floor name from a floor id."""
        floor_registry = fr.async_get(self.hass)
        if floor := floor_registry.async_get_floor(lookup_value):
            return floor.name

        if aid := self.area_id(lookup_value):
            area_reg = ar.async_get(self.hass)
            if (
                (area := area_reg.async_get_area(aid))
                and area.floor_id
                and (floor := floor_registry.async_get_floor(area.floor_id))
            ):
                return floor.name

        return None

    @TemplateExtension.filter(
        name="floor_areas",
        limited=True,
        categories=[FeatureCategories.CORE.value],
    )
    @TemplateExtension.function(
        name="floor_areas",
        limited=True,
        categories=[FeatureCategories.CORE.value],
    )
    def floor_areas(self, floor_id_or_name: str) -> Iterable[str]:
        """Return area IDs for a given floor ID or name."""
        _floor_id: str | None
        # If floor_name returns a value, we know the input was an ID, otherwise we
        # assume it's a name, and if it's neither, we return early
        if self.floor_name(floor_id_or_name) is not None:
            _floor_id = floor_id_or_name
        else:
            _floor_id = self.floor_id(floor_id_or_name)
        if _floor_id is None:
            return []

        area_reg = ar.async_get(self.hass)
        entries = ar.async_entries_for_floor(area_reg, _floor_id)
        return [entry.id for entry in entries if entry.id]

    @TemplateExtension.filter(
        name="integration_entities",
        limited=True,
        categories=[FeatureCategories.CORE.value],
    )
    @TemplateExtension.function(
        name="integration_entities",
        limited=True,
        categories=[FeatureCategories.CORE.value],
    )
    def integration_entities(self, entry_name: str) -> Iterable[str]:
        """Get entity ids for entities tied to an integration/domain.

        Provide entry_name as domain to get all entity id's for a integration/domain
        or provide a config entry title for filtering between instances of the same
        integration.
        """

        # Don't allow searching for config entries without title
        if not entry_name:
            return []

        # first try if there are any config entries with a matching title
        entities: list[str] = []
        ent_reg = er.async_get(self.hass)
        for entry in self.hass.config_entries.async_entries():
            if entry.title != entry_name:
                continue
            entries = er.async_entries_for_config_entry(ent_reg, entry.entry_id)
            entities.extend(entry.entity_id for entry in entries)
        if entities:
            return entities

        # fallback to just returning all entities for a domain
        # pylint: disable-next=import-outside-toplevel
        from homeassistant.helpers.entity import entity_sources

        return [
            entity_id
            for entity_id, info in entity_sources(self.hass).items()
            if info["domain"] == entry_name
        ]

    @TemplateExtension.filter(
        name="labels",
        limited=True,
        categories=[FeatureCategories.CORE.value],
    )
    @TemplateExtension.function(
        name="labels",
        limited=True,
        categories=[FeatureCategories.CORE.value],
    )
    def labels(self, lookup_value: Any = None) -> Iterable[str | None]:
        """Return all labels, or those from a area ID, device ID, or entity ID."""
        label_reg = lr.async_get(self.hass)
        if lookup_value is None:
            return list(label_reg.labels)

        ent_reg = er.async_get(self.hass)

        lookup_value = str(lookup_value)

        try:
            cv.entity_id(lookup_value)
        except vol.Invalid:
            pass
        else:
            if entity := ent_reg.async_get(lookup_value):
                return list(entity.labels)

        # Check if this could be a device ID
        dev_reg = dr.async_get(self.hass)
        if device := dev_reg.async_get(lookup_value):
            return list(device.labels)

        # Check if this could be a area ID
        area_reg = ar.async_get(self.hass)
        if area := area_reg.async_get_area(lookup_value):
            return list(area.labels)

        return []

    @TemplateExtension.filter(
        name="label_id",
        categories=[FeatureCategories.CORE.value],
    )
    @TemplateExtension.function(
        name="label_id",
        categories=[FeatureCategories.CORE.value],
    )
    def label_id(self, lookup_value: Any) -> str | None:
        """Get the label ID from a label name."""
        label_reg = lr.async_get(self.hass)
        if label := label_reg.async_get_label_by_name(str(lookup_value)):
            return label.label_id
        return None

    @TemplateExtension.filter(
        name="label_name",
        categories=[FeatureCategories.CORE.value],
    )
    @TemplateExtension.function(
        name="label_name",
        categories=[FeatureCategories.CORE.value],
    )
    def label_name(self, lookup_value: str) -> str | None:
        """Get the label name from a label ID."""
        label_reg = lr.async_get(self.hass)
        if label := label_reg.async_get_label(lookup_value):
            return label.name
        return None

    @TemplateExtension.filter(
        name="label_areas",
        limited=True,
        categories=[FeatureCategories.CORE.value],
    )
    @TemplateExtension.function(
        name="label_areas",
        limited=True,
        categories=[FeatureCategories.CORE.value],
    )
    def label_areas(self, label_id_or_name: str) -> Iterable[str]:
        """Return areas for a given label ID or name."""

        if (_label_id := self._label_id_or_name(label_id_or_name)) is None:
            return []
        area_reg = ar.async_get(self.hass)
        entries = ar.async_entries_for_label(area_reg, _label_id)
        return [entry.id for entry in entries]

    def _label_id_or_name(self, label_id_or_name: str) -> str | None:
        """Get the label ID from a label name or ID."""
        # If label_name returns a value, we know the input was an ID, otherwise we
        # assume it's a name, and if it's neither, we return early.
        if self.label_name(label_id_or_name) is not None:
            return label_id_or_name
        label_id: str | None = self.label_id(label_id_or_name)
        return label_id

    @TemplateExtension.filter(
        name="label_devices",
        limited=True,
        categories=[FeatureCategories.CORE.value],
    )
    @TemplateExtension.function(
        name="label_devices",
        limited=True,
        categories=[FeatureCategories.CORE.value],
    )
    def label_devices(self, label_id_or_name: str) -> Iterable[str]:
        """Return device IDs for a given label ID or name."""
        if (_label_id := self._label_id_or_name(label_id_or_name)) is None:
            return []
        dev_reg = dr.async_get(self.hass)
        entries = dr.async_entries_for_label(dev_reg, _label_id)
        return [entry.id for entry in entries]

    @TemplateExtension.filter(
        name="label_entities",
        limited=True,
        categories=[FeatureCategories.CORE.value],
    )
    @TemplateExtension.function(
        name="label_entities",
        limited=True,
        categories=[FeatureCategories.CORE.value],
    )
    def label_entities(self, label_id_or_name: str) -> Iterable[str]:
        """Return entities for a given label ID or name."""
        if (_label_id := self._label_id_or_name(label_id_or_name)) is None:
            return []
        ent_reg = er.async_get(self.hass)
        entries = er.async_entries_for_label(ent_reg, _label_id)
        return [entry.entity_id for entry in entries]

    @TemplateExtension.filter(
        name="expand",
        categories=[FeatureCategories.CORE.value],
    )
    @TemplateExtension.function(
        name="expand",
        categories=[FeatureCategories.CORE.value],
    )
    def expand(self, *args: Any) -> Iterable[State]:
        """Expand out any groups and zones into entity states."""

        search = list(args)
        found = {}
        sources = entity_helper.entity_sources(self.hass)
        while search:
            entity = search.pop()
            if isinstance(entity, str):
                entity_id = entity
                if (entity := self._get_state(entity)) is None:
                    continue
            elif isinstance(entity, State):
                entity_id = entity.entity_id
            elif isinstance(entity, collections.abc.Iterable):
                search += entity
                continue
            else:
                # ignore other types
                continue

            if entity_id in found:
                continue

            domain = entity.domain
            if domain == "group" or (
                (source := sources.get(entity_id)) and source["domain"] == "group"
            ):
                # Collect state will be called in here since it's wrapped
                if group_entities := entity.attributes.get(ATTR_ENTITY_ID):
                    search += group_entities
            elif domain == "zone":
                if zone_entities := entity.attributes.get(ATTR_PERSONS):
                    search += zone_entities
            else:
                self._collect_state(entity_id)
                found[entity_id] = entity

        return list(found.values())

    def _collect_state(self, entity_id: str) -> None:
        if (entity_collect := self.render_info.get()) is not None:
            entity_collect.entities.add(entity_id)  # type: ignore[attr-defined]

    @TemplateExtension.function(
        name="closest",
        categories=[FeatureCategories.CORE.value],
    )
    def closest(self, *args: Any) -> State | None:
        """Find closest entity.

        Closest to home:
            closest(states)
            closest(states.device_tracker)
            closest('group.children')
            closest(states.group.children)

        Closest to a point:
            closest(23.456, 23.456, 'group.children')
            closest('zone.school', 'group.children')
            closest(states.zone.school, 'group.children')

        As a filter:
            states | closest
            states.device_tracker | closest
            ['group.children', states.device_tracker] | closest
            'group.children' | closest(23.456, 23.456)
            states.device_tracker | closest('zone.school')
            'group.children' | closest(states.zone.school)

        """
        if len(args) == 1:
            latitude: float | None = self.hass.config.latitude
            longitude: float | None = self.hass.config.longitude
            entities = args[0]

        elif len(args) == 2:
            point_state = self._resolve_state(args[0])

            if point_state is None:
                _LOGGER.warning("Closest:Unable to find state %s", args[0])
                return None
            if not loc_helper.has_location(point_state):
                _LOGGER.warning(
                    "Closest:State does not contain valid location: %s", point_state
                )
                return None

            latitude = point_state.attributes.get(ATTR_LATITUDE)
            longitude = point_state.attributes.get(ATTR_LONGITUDE)

            entities = args[1]

        else:
            latitude = convert(args[0], float)
            longitude = convert(args[1], float)

            if latitude is None or longitude is None:
                _LOGGER.warning(
                    "Closest:Received invalid coordinates: %s, %s", args[0], args[1]
                )
                return None

            entities = args[2]

        states = self.expand(entities)

        # state will already be wrapped here
        return loc_helper.closest(latitude, longitude, states)

    def _resolve_state(self, entity_id_or_state: Any) -> State | TemplateState | None:
        """Return state or entity_id if given."""

        if isinstance(entity_id_or_state, State):
            return entity_id_or_state
        if isinstance(entity_id_or_state, str):
            return self._get_state(entity_id_or_state)
        return None

    @TemplateExtension.filter(
        name="closest",
        categories=[FeatureCategories.CORE.value],
    )
    def closest_filter(self, *args: Any) -> Any:
        """Call closest as a filter. Need to reorder arguments."""
        new_args = list(args[1:])
        new_args.append(args[0])
        return self.closest(*new_args)

    @TemplateExtension.function(
        name="distance",
        categories=[FeatureCategories.CORE.value],
    )
    def distance(self, *args: Any) -> float | None:
        """Calculate distance.

        Will calculate distance from home to a point or between points.
        Points can be passed in using state objects or lat/lng coordinates.
        """
        locations = []

        to_process = list(args)

        while to_process:
            value = to_process.pop(0)
            if isinstance(value, str) and not valid_entity_id(value):
                point_state = None
            else:
                point_state = self._resolve_state(value)

            if point_state is None:
                # We expect this and next value to be lat&lng
                if not to_process:
                    _LOGGER.warning(
                        "Distance:Expected latitude and longitude, got %s", value
                    )
                    return None

                value_2 = to_process.pop(0)
                latitude = convert(value, float)
                longitude = convert(value_2, float)

                if latitude is None or longitude is None:
                    _LOGGER.warning(
                        "Distance:Unable to process latitude and longitude: %s, %s",
                        value,
                        value_2,
                    )
                    return None

            else:
                if not loc_helper.has_location(point_state):
                    _LOGGER.warning(
                        "Distance:State does not contain valid location: %s",
                        point_state,
                    )
                    return None

                latitude = point_state.attributes.get(ATTR_LATITUDE)
                longitude = point_state.attributes.get(ATTR_LONGITUDE)

            locations.append((latitude, longitude))

        if len(locations) == 1:
            return self.hass.config.distance(*locations[0])

        return self.hass.config.units.length(
            loc_util.distance(*locations[0] + locations[1]), UnitOfLength.METERS
        )

    @TemplateExtension.function(
        name="is_hidden_entity",
        categories=[FeatureCategories.CORE.value],
    )
    @TemplateExtension.test(
        name="is_hidden_entity",
        categories=[FeatureCategories.CORE.value],
    )
    def is_hidden_entity(self, entity_id: str) -> bool:
        """Test if an entity is hidden."""
        entity_reg = er.async_get(self.hass)
        entry = entity_reg.async_get(entity_id)
        return entry is not None and entry.hidden

    @TemplateExtension.function(
        name="is_state",
        categories=[FeatureCategories.CORE.value],
    )
    @TemplateExtension.test(
        name="is_state",
        categories=[FeatureCategories.CORE.value],
    )
    def is_state(self, entity_id: str, state: str | list[str]) -> bool:
        """Test if a state is a specific value."""
        state_obj = self._get_state(entity_id)
        return state_obj is not None and (
            state_obj.state == state
            or isinstance(state, list)
            and state_obj.state in state
        )

    def _get_state(self, entity_id: str) -> TemplateState | None:
        return _get_template_state_from_state(
            self.hass, entity_id, self.hass.states.get(entity_id)
        )

    @TemplateExtension.function(
        name="is_state_attr",
        categories=[FeatureCategories.CORE.value],
    )
    @TemplateExtension.test(
        name="is_state_attr",
        categories=[FeatureCategories.CORE.value],
    )
    def is_state_attr(self, entity_id: str, name: str, value: Any) -> bool:
        """Test if a state's attribute is a specific value."""
        attr = self.state_attr(entity_id, name)
        return attr is not None and attr == value

    @TemplateExtension.function(
        name="state_attr",
        categories=[FeatureCategories.CORE.value],
    )
    @TemplateExtension.filter(
        name="state_attr",
        categories=[FeatureCategories.CORE.value],
    )
    def state_attr(self, entity_id: str, name: str) -> Any:
        """Get a specific attribute from a state."""
        if (state_obj := self._get_state(entity_id)) is not None:
            return state_obj.attributes.get(name)
        return None

    @TemplateExtension.function(
        name="states",
        passthrough=True,
        categories=[FeatureCategories.CORE.value],
    )
    @TemplateExtension.filter(
        name="states",
        passthrough=True,
        categories=[FeatureCategories.CORE.value],
    )
    def states(self) -> AllStates:
        """Get all states."""
        return AllStates(self.hass)

    @TemplateExtension.function(
        name="state_translated",
        passthrough=True,
        categories=[FeatureCategories.CORE.value],
    )
    @TemplateExtension.filter(
        name="state_translated",
        passthrough=True,
        categories=[FeatureCategories.CORE.value],
    )
    def state_translated(self) -> StateTranslated:
        """Get all states."""
        return StateTranslated(self.hass)

    @TemplateExtension.function(
        name="has_value",
        categories=[FeatureCategories.CORE.value],
    )
    @TemplateExtension.filter(
        name="has_value",
        categories=[FeatureCategories.CORE.value],
    )
    @TemplateExtension.test(
        name="has_value",
        categories=[FeatureCategories.CORE.value],
    )
    def has_value(self, entity_id: str) -> bool:
        """Test if an entity has a valid value."""
        state_obj = self._get_state(entity_id)

        return state_obj is not None and (
            state_obj.state not in [STATE_UNAVAILABLE, STATE_UNKNOWN]
        )

    @TemplateExtension.function(
        name="utcnow",
        categories=[FeatureCategories.CORE.value],
    )
    def utcnow(self) -> datetime:
        """Record fetching utcnow."""
        if (render_info := self.render_info.get()) is not None:
            render_info.has_time = True

        return dt_util.utcnow()

    @TemplateExtension.function(
        name="now",
        categories=[FeatureCategories.CORE.value],
    )
    def now(self) -> datetime:
        """Record fetching now."""
        if (render_info := self.render_info.get()) is not None:
            render_info.has_time = True

        return dt_util.now()

    @TemplateExtension.filter(
        name="relative_time",
        categories=[FeatureCategories.CORE.value],
    )
    @TemplateExtension.function(
        name="relative_time",
        categories=[FeatureCategories.CORE.value],
    )
    def relative_time(self, value: Any) -> Any:
        """Take a datetime and return its "age" as a string.

        The age can be in second, minute, hour, day, month or year. Only the
        biggest unit is considered, e.g. if it's 2 days and 3 hours, "2 days" will
        be returned.
        If the input datetime is in the future,
        the input datetime will be returned.

        If the input are not a datetime object the input will be returned unmodified.

        Note: This template function is deprecated in favor of `time_until`, but is still
        supported so as not to break old templates.
        """
        if (render_info := self.render_info.get()) is not None:
            render_info.has_time = True

        if not isinstance(value, datetime):
            return value
        if not value.tzinfo:
            value = dt_util.as_local(value)
        if dt_util.now() < value:
            return value
        return dt_util.get_age(value)

    @TemplateExtension.filter(
        name="today_at",
        categories=[FeatureCategories.CORE.value],
    )
    @TemplateExtension.function(
        name="today_at",
        categories=[FeatureCategories.CORE.value],
    )
    def today_at(self, time_str: str = "") -> datetime:
        """Record fetching now where the time has been replaced with value."""
        if (render_info := self.render_info.get()) is not None:
            render_info.has_time = True

        today = dt_util.start_of_local_day()
        if not time_str:
            return today

        if (time_today := dt_util.parse_time(time_str)) is None:
            raise ValueError(
                f"could not convert {type(time_str).__name__} to datetime: '{time_str}'"
            )

        return datetime.combine(today, time_today, today.tzinfo)

    @TemplateExtension.filter(
        name="time_since",
        categories=[FeatureCategories.CORE.value],
    )
    @TemplateExtension.function(
        name="time_since",
        categories=[FeatureCategories.CORE.value],
    )
    def time_since(self, value: Any | datetime, precision: int = 1) -> Any:
        """Take a datetime and return its "age" as a string.

        The age can be in seconds, minutes, hours, days, months and year.

        precision is the number of units to return, with the last unit rounded.

        If the value not a datetime object the input will be returned unmodified.
        """
        if (render_info := self.render_info.get()) is not None:
            render_info.has_time = True

        if not isinstance(value, datetime):
            return value
        if not value.tzinfo:
            value = dt_util.as_local(value)
        if dt_util.now() < value:
            return value

        return dt_util.get_age(value, precision)

    @TemplateExtension.filter(
        name="time_until",
        categories=[FeatureCategories.CORE.value],
    )
    @TemplateExtension.function(
        name="time_until",
        categories=[FeatureCategories.CORE.value],
    )
    def time_until(self, value: Any | datetime, precision: int = 1) -> Any:
        """Take a datetime and return the amount of time until that time as a string.

        The time until can be in seconds, minutes, hours, days, months and years.

        precision is the number of units to return, with the last unit rounded.

        If the value not a datetime object the input will be returned unmodified.
        """
        if (render_info := self.render_info.get()) is not None:
            render_info.has_time = True

        if not isinstance(value, datetime):
            return value
        if not value.tzinfo:
            value = dt_util.as_local(value)
        if dt_util.now() > value:
            return value

        return dt_util.get_time_remaining(value, precision)
