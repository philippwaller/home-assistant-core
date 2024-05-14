"""Template extensions for Home Assistant."""

from abc import abstractmethod
from collections.abc import Callable
import copy
from enum import Enum
from functools import partial, wraps
import inspect
from typing import Any, Generic, NoReturn, Optional, ParamSpec, TypeVar

from jinja2 import Environment, pass_context
from jinja2.ext import Extension

from homeassistant.const import (
    ATTR_EXTENSION_ID,
    EVENT_TEMPLATE_EXTENSION_REGISTERED,
    EVENT_TEMPLATE_EXTENSION_REMOVED,
    FEAT_DATA_ATTRIBUTE,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import TemplateError
from homeassistant.util.async_ import run_callback_threadsafe

from .translation import async_get_translations

_SENTINEL = object()

_R = TypeVar("_R")
_P = ParamSpec("_P")


def async_setup(hass: HomeAssistant) -> bool:
    """Set up the template extensions registry."""
    TemplateExtensionRegistry(hass)
    return True


class TemplateExtensionFeature:
    """A template extension feature."""

    class FeatureCategory(Enum):
        """Categories for TemplateExtensionFeature."""

        TEXT_MANIPULATION = "Text Manipulation"
        NUMERIC_OPERATION = "Number Operations"
        DATA_FORMATTING = "Data Formatting"
        CONDITIONAL_PROCESSING = "Conditional Processing"
        TYPE_TESTING = "Type Testing"
        PATTERN_MATCHING = "Pattern Matching"
        DATE_AND_TIME = "Datetime and Time"

    class FeatureType(Enum):
        """Types of template extension features."""

        FILTER = "filter"
        GLOBAL = "global"
        TEST = "test"

    def __init__(
        self,
        feature_type: FeatureType,
        name: str,
        func: Callable,
        passthrough: bool = False,
        cacheable: bool = False,
        limited: bool = False,
        short_description: str | None = None,
        description: str | None = None,
        categories: list[str] | None = None,
    ) -> None:
        """Initialize a new template extension feature."""
        self._feature_type = feature_type
        self._name = name
        self._func = func
        self._cacheable = cacheable
        self._passthrough = passthrough
        self._limited = limited
        self._description = description
        self._short_description = short_description
        self._categories = categories

        sig = inspect.signature(func)
        self._sig = sig.replace(parameters=list(sig.parameters.values())[1:])

    @property
    def feature_type(self) -> FeatureType:
        """Return the feature type."""
        return self._feature_type

    @property
    def feature_id(self) -> str:
        """Return the feature ID."""
        return f"{self._feature_type.value}.{self._name.lower().replace(' ', '_')}"

    @property
    def name(self) -> str:
        """Return the feature name."""
        return self._name

    @property
    def func(self) -> Callable:
        """Return the feature function."""
        return self._func

    @property
    def passthrough(self) -> bool:
        """Return if the feature should assign the return value instead of the feature function."""
        return self._passthrough

    @property
    def cacheable(self) -> bool:
        """Return if the feature is cacheable."""
        return self._cacheable

    @property
    def limited(self) -> bool:
        """Return if the feature is available in limited mode."""
        return self._limited

    @property
    def short_description(self) -> str | None:
        """Return the short description of the feature."""
        return self._short_description

    @property
    def description(self) -> str | None:
        """Return the description of the feature."""
        return self._description

    @property
    def categories(self) -> list[str] | None:
        """Return the categories of the feature."""
        return self._categories

    @property
    def signature(self) -> inspect.Signature:
        """Return the signature of the feature."""
        # remove the first parameter (self)
        return self._sig

    async def localize(
        self, hass: HomeAssistant, component: str, extension_id: str, language: str
    ) -> "TemplateExtensionFeature":
        """Localize the feature and return a new instance."""
        translations = await async_get_translations(hass, language, "template")

        extension_id = extension_id.replace(".", "-")
        feature_key = f"component.{component}.template.extensions.{extension_id}.{self.feature_type.value}.{self.name}"

        if self.short_description is None:
            short_description_key = f"{feature_key}.short_description"
            short_description = translations.get(short_description_key) or None
        else:
            short_description = (
                translations.get(self.short_description) or self.short_description
            )

        if self.description is None:
            description_key = f"{feature_key}.description"
            description = translations.get(description_key) or None
        else:
            description = translations.get(self.description) or self.description

        return TemplateExtensionFeature(
            feature_type=self.feature_type,
            name=self.name,
            func=self.func,
            limited=self.limited,
            short_description=short_description,
            description=description,
            categories=self.categories,
        )

    def update_categories(self, categories: list[str]) -> "TemplateExtensionFeature":
        """Update the categories of the feature and return a new instance."""
        return TemplateExtensionFeature(
            feature_type=self.feature_type,
            name=self.name,
            func=self.func,
            limited=self.limited,
            short_description=self.short_description,
            description=self.description,
            categories=categories,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the feature."""
        return {
            "feature_id": self.feature_id,
            "feature_type": self.feature_type,
            "name": self.name,
            "limited": self.limited,
            "short_description": self.short_description,
            "description": self.description,
            "categories": self.categories,
            "signature": str(self.signature),
        }


class TemplateExtensionMeta(type):
    """Metaclass for TemplateExtensionEnvironment."""

    _features: list[TemplateExtensionFeature]

    def __new__(
        mcs,  # noqa: N804
        name: str,
        bases: tuple[type, ...],
        class_dict: dict[str, Any],
    ) -> "TemplateExtensionMeta":
        """Create a new TemplateExtension class."""
        ext = super().__new__(mcs, name, bases, class_dict)
        if not hasattr(ext, "_features"):
            ext._features = []
        else:
            ext._features = [
                copy.copy(f) for f in ext._features
            ]  # Kopieren der Features für jede Subklasse

        for attr in class_dict.values():
            if hasattr(attr, "_feature_data"):
                for feature_data in attr._feature_data:
                    ext._features.append(feature_data)

        return ext


class TemplateExtension(Extension, metaclass=TemplateExtensionMeta):
    """Base class for template extensions."""

    _features: list[TemplateExtensionFeature]

    def __init__(
        self,
        environment: Environment,
        hass: HomeAssistant,
        limited: bool,
        strict: bool,
        render_info: Any,
    ) -> None:
        """Initialize the template extension."""
        # pylint: disable=import-outside-toplevel
        from .template import RenderInfo

        super().__init__(environment)
        self._hass = hass
        self._limited = limited
        self._strict = strict
        self._render_info: RenderInfo = render_info
        self._register_features()

    @staticmethod
    @abstractmethod
    def extension_id() -> str:
        """Override this in child classes to return the correct extension ID."""
        raise NotImplementedError("Must be implemented by subclass.")

    @staticmethod
    @abstractmethod
    def component() -> str | None:
        """Override this in child classes and return component/integration name to enable localisation for features."""
        return None

    @property
    def limited(self) -> bool:
        """Return if the extension is in limited mode."""
        return self._limited

    @property
    def strict(self) -> bool:
        """Return if the extension is in strict mode."""
        return self._strict

    @property
    def render_info(self) -> Any:
        """Return the render info."""
        return self._render_info

    @property
    def features(self) -> list[TemplateExtensionFeature]:
        """Return the features of the extension."""
        return self._features

    @property
    def hass(self) -> HomeAssistant:
        """Return the HomeAssistant instance."""
        return self._hass

    def _register_features(self) -> None:
        """Register the features of the extension."""
        type_map = {
            TemplateExtensionFeature.FeatureType.FILTER: self.environment.filters,
            TemplateExtensionFeature.FeatureType.TEST: self.environment.tests,
            TemplateExtensionFeature.FeatureType.GLOBAL: self.environment.globals,
        }

        for feature in self._features:
            if self.limited and not feature.limited:
                func = self._unsupported_mode(feature.feature_id)
            else:
                func = feature.func.__get__(self, type(self))  # pylint: disable=unnecessary-dunder-call
                if not feature.cacheable:
                    func = self._wrap_context(func)

            type_map[feature.feature_type][feature.name] = (
                func() if feature.passthrough else func
            )

    def _unsupported_mode(self, name: str) -> Callable[[], NoReturn]:
        """Return a function that raises an error for unsupported features."""

        def unsupported(*args: Any, **kwargs: Any) -> NoReturn:
            raise TemplateError(
                f"Use of '{name}' is not supported in limited templates"
            )

        return unsupported

    def as_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the extension."""
        return {
            "extension_id": self.extension_id(),
            "features": [feature.as_dict() for feature in self.features],
            "limited": self.limited,
            "strict": self.strict,
        }

    def _wrap_context(
        self,
        func: Callable[..., _R],
        jinja_context: Callable[
            [Callable[..., _R]],
            Callable[..., _R],
        ] = pass_context,
    ) -> Callable[..., _R]:
        """Wrap function that depend on hass."""

        @wraps(func)
        def wrapper(_: Any = None, *args: _P.args, **kwargs: _P.kwargs) -> _R:
            return func(*args, **kwargs)

        return jinja_context(wrapper)

    @classmethod
    def filter(
        cls,
        name: str,
        limited: bool = False,
        passthrough: bool = False,
        cacheable: bool = False,
        short_description: str | None = None,
        description: str | None = None,
        categories: list[str] | None = None,
    ) -> Callable:
        """Register a filter feature using this decorator."""

        def decorator(func: Callable) -> Callable:
            if not hasattr(func, FEAT_DATA_ATTRIBUTE):
                setattr(func, FEAT_DATA_ATTRIBUTE, [])

            getattr(func, FEAT_DATA_ATTRIBUTE).append(
                TemplateExtensionFeature(
                    feature_type=TemplateExtensionFeature.FeatureType.FILTER,
                    name=name,
                    func=func,
                    passthrough=passthrough,
                    cacheable=cacheable,
                    limited=limited,
                    description=description,
                    short_description=short_description,
                    categories=categories,
                )
            )
            return func

        return decorator

    @classmethod
    def test(
        cls,
        name: str,
        limited: bool = False,
        passthrough: bool = False,
        cacheable: bool = False,
        short_description: str | None = None,
        description: str | None = None,
        categories: list[str] | None = None,
    ) -> Callable:
        """Register a test feature using this decorator."""

        def decorator(func: Callable) -> Callable:
            if not hasattr(func, FEAT_DATA_ATTRIBUTE):
                setattr(func, FEAT_DATA_ATTRIBUTE, [])

            getattr(func, FEAT_DATA_ATTRIBUTE).append(
                TemplateExtensionFeature(
                    feature_type=TemplateExtensionFeature.FeatureType.TEST,
                    name=name,
                    func=func,
                    passthrough=passthrough,
                    cacheable=cacheable,
                    limited=limited,
                    description=description,
                    short_description=short_description,
                    categories=categories,
                )
            )
            return func

        return decorator

    @classmethod
    def function(
        cls,
        name: str,
        limited: bool = False,
        passthrough: bool = False,
        cacheable: bool = False,
        short_description: str | None = None,
        description: str | None = None,
        categories: list[str] | None = None,
    ) -> Callable:
        """Register a global feature using this decorator."""

        def decorator(func: Callable) -> Callable:
            if not hasattr(func, FEAT_DATA_ATTRIBUTE):
                setattr(func, FEAT_DATA_ATTRIBUTE, [])

            getattr(func, FEAT_DATA_ATTRIBUTE).append(
                TemplateExtensionFeature(
                    feature_type=TemplateExtensionFeature.FeatureType.GLOBAL,
                    name=name,
                    func=func,
                    passthrough=passthrough,
                    cacheable=cacheable,
                    limited=limited,
                    description=description,
                    short_description=short_description,
                    categories=categories,
                )
            )
            return func

        return decorator


class FeatureCategories(Enum):
    """Categories for core functions."""

    BINARY = "Binary"
    CONTROL_FLOW = "Control Flow"
    CORE = "Core"
    DATATYPE = "Datatype"
    DATETIME = "Datetime"
    MATH = "Math"
    REGEX = "Regex"
    STASTISTICS = "Statistics"
    STRING = "String"


T = TypeVar("T", bound=TemplateExtension)


class HassAwareTemplateExtension(Generic[T]):
    """Wrap a TemplateExtension with the Home Assistant (hass) instance."""

    def __init__(
        self, environment: Environment, limited: bool, strict: bool, render_info: Any
    ) -> None:
        """Initialize the wrapper extension with environment and render info."""


class ContextAwareTemplateExtension:
    """Base class to wrap a HassAwareTemplateExtension with the limited and strict context."""

    def __init__(self, environment: Environment) -> None:
        """Initialize the extension."""


class TemplateExtensionRegistry:
    """A singleton registry for managing template extensions throughout the application lifecycle of Home Assistant.

    This class is instantiated during the boot process of Home Assistant and maintains its entries in memory across the application lifecycle.
    It should be accessed through static method get_instance() rather than by direct instantiation.
    """

    _instance: Optional["TemplateExtensionRegistry"] = None
    _extensions: dict[str, type[HassAwareTemplateExtension]] = {}
    _hass: HomeAssistant

    # __slots__ = ("_extensions", "_hass")

    def __init__(self, hass: HomeAssistant) -> None:
        """Private constructor. Use get_instance() to get the singleton instance."""
        if TemplateExtensionRegistry._instance is not None:
            raise RuntimeError(
                "This class is a singleton! Call get_instance() instead."
            )

        self._hass = hass
        TemplateExtensionRegistry._instance = self

    @staticmethod
    def get_instance() -> "TemplateExtensionRegistry":
        """Return the singleton instance of the TemplateExtensionRegistry."""
        if TemplateExtensionRegistry._instance is None:
            raise RuntimeError("TemplateExtensionRegistry has not been initialized.")
        return TemplateExtensionRegistry._instance

    @property
    def extensions(self) -> dict[str, type[HassAwareTemplateExtension]]:
        """Return dictionary with per domain a list of available extensions."""
        return run_callback_threadsafe(
            self._hass.loop, partial(self.async_extensions)
        ).result()

    @callback
    def async_extensions(
        self,
    ) -> dict[str, type[HassAwareTemplateExtension[TemplateExtension]]]:
        """Return dictionary with per domain a list of available extensions.

        This method makes a copy of the registry. This function is expensive,
        and should only be used if has_extension is not sufficient.

        This method must be run in the event loop.
        """
        return {
            uid: copy.deepcopy(extension) for uid, extension in self._extensions.items()
        }

    def has_extension(self, extension_id: str) -> bool:
        """Test if specified extension exists."""
        return extension_id.lower() in self._extensions

    def register(
        self,
        extension: TemplateExtension,
    ) -> None:
        """Register a template extension."""
        run_callback_threadsafe(
            self._hass.loop, partial(self.async_register, extension)
        ).result()

    @callback
    def async_register(
        self,
        extension: type[TemplateExtension],
    ) -> None:
        """Register a template extension.

        This method must be run in the event loop.
        """
        uid = extension.extension_id().lower()
        self._extensions[uid] = self._wrap_template_extension(extension)

        # self._extensions[uid] = hassExtension

        self._hass.bus.async_fire(
            EVENT_TEMPLATE_EXTENSION_REGISTERED, {ATTR_EXTENSION_ID: uid}
        )

    def remove(self, extension: TemplateExtension) -> None:
        """Remove a registered extension from extension handler."""
        run_callback_threadsafe(
            self._hass.loop, partial(self.async_remove, extension)
        ).result()

    @callback
    def async_remove(self, extension: TemplateExtension) -> None:
        """Remove a registered extension from extension handler.

        This method must be run in the event loop.
        """
        uid = extension.extension_id().lower()
        self._extensions.pop(uid, None)

        self._hass.bus.async_fire(
            EVENT_TEMPLATE_EXTENSION_REMOVED, {ATTR_EXTENSION_ID: uid}
        )

    def _wrap_template_extension(
        self,
        extension_class: type[T],
    ) -> "type[HassAwareTemplateExtension[T]]":
        """Wrap a TemplateExtension with the current hass instance."""
        if not issubclass(extension_class, TemplateExtension):
            raise TypeError(
                "Expected extension_class to be a subclass of TemplateExtension"
            )

        hass = self._hass

        class HassAwareTemplateExtensionWrapper(
            extension_class,  # type: ignore[valid-type,misc]
            HassAwareTemplateExtension[T],
        ):
            """Wrap a Extension with the current hass instance."""

            def __init__(
                self,
                environment: Environment,
                limited: bool,
                strict: bool,
                render_info: Any,
            ) -> None:
                """Initialize the extension."""
                extension_class.__init__(
                    self, environment, hass, limited, strict, render_info
                )

        return HassAwareTemplateExtensionWrapper
