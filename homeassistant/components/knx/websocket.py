"""KNX Websocket API."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from functools import cache, wraps
from typing import TYPE_CHECKING, Any, Final, overload

import knx_frontend as knx_panel
import voluptuous as vol
from xknx.telegram import Telegram
from xknxproject.exceptions import XknxProjectException

from homeassistant.components import panel_custom, websocket_api
from homeassistant.components.http import StaticPathConfig
from homeassistant.const import CONF_ENTITY_ID, CONF_PLATFORM, Platform
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.typing import UNDEFINED
from homeassistant.util.ulid import ulid_now

from .const import DOMAIN, KNX_MODULE_KEY
from .entity import EntityConfiguration, Persistable
from .schema import SchemaSerializer
from .sensor import UiSensorConfig
from .storage.config_store import ConfigStoreException
from .storage.const import CONF_DATA
from .storage.entity_store_schema import (
    CREATE_ENTITY_BASE_SCHEMA,
    UPDATE_ENTITY_BASE_SCHEMA,
)
from .storage.entity_store_validation import (
    EntityStoreValidationException,
    EntityStoreValidationSuccess,
    validate_entity_data,
)
from .telegrams import SIGNAL_KNX_TELEGRAM, TelegramDict

if TYPE_CHECKING:
    from knx_module import KNXModule

URL_BASE: Final = "/knx_static"


async def register_panel(hass: HomeAssistant) -> None:
    """Register the KNX Panel and Websocket API."""
    websocket_api.async_register_command(hass, ws_info)
    websocket_api.async_register_command(hass, ws_project_file_process)
    websocket_api.async_register_command(hass, ws_project_file_remove)
    websocket_api.async_register_command(hass, ws_group_monitor_info)
    websocket_api.async_register_command(hass, ws_group_telegrams)
    websocket_api.async_register_command(hass, ws_subscribe_telegram)
    websocket_api.async_register_command(hass, ws_get_knx_project)
    websocket_api.async_register_command(hass, ws_validate_entity)
    websocket_api.async_register_command(hass, ws_create_entity)
    websocket_api.async_register_command(hass, ws_update_entity)
    websocket_api.async_register_command(hass, ws_delete_entity)
    websocket_api.async_register_command(hass, ws_get_entity_config)
    websocket_api.async_register_command(hass, ws_get_entity_entries)
    websocket_api.async_register_command(hass, ws_create_device)
    websocket_api.async_register_command(hass, ws_get_entity_schemas)
    websocket_api.async_register_command(hass, ws_create_platform_entity)

    if DOMAIN not in hass.data.get("frontend_panels", {}):
        await hass.http.async_register_static_paths(
            [
                StaticPathConfig(
                    URL_BASE,
                    path=knx_panel.locate_dir(),
                    cache_headers=knx_panel.is_prod_build,
                )
            ]
        )
        await panel_custom.async_register_panel(
            hass=hass,
            frontend_url_path=DOMAIN,
            webcomponent_name=knx_panel.webcomponent_name,
            sidebar_title=DOMAIN.upper(),
            sidebar_icon="mdi:bus-electric",
            module_url=f"{URL_BASE}/{knx_panel.entrypoint_js}",
            embed_iframe=True,
            require_admin=True,
        )


type KnxWebSocketCommandHandler = Callable[
    [HomeAssistant, KNXModule, websocket_api.ActiveConnection, dict[str, Any]], None
]
type KnxAsyncWebSocketCommandHandler = Callable[
    [HomeAssistant, KNXModule, websocket_api.ActiveConnection, dict[str, Any]],
    Awaitable[None],
]


@overload
def provide_knx(
    func: KnxAsyncWebSocketCommandHandler,
) -> websocket_api.const.AsyncWebSocketCommandHandler: ...
@overload
def provide_knx(
    func: KnxWebSocketCommandHandler,
) -> websocket_api.const.WebSocketCommandHandler: ...


def provide_knx(
    func: KnxAsyncWebSocketCommandHandler | KnxWebSocketCommandHandler,
) -> (
    websocket_api.const.AsyncWebSocketCommandHandler
    | websocket_api.const.WebSocketCommandHandler
):
    """Websocket decorator to provide a KNXModule instance."""

    def _send_not_loaded_error(
        connection: websocket_api.ActiveConnection, msg_id: int
    ) -> None:
        connection.send_error(
            msg_id,
            websocket_api.const.ERR_HOME_ASSISTANT_ERROR,
            "KNX integration not loaded.",
        )

    if asyncio.iscoroutinefunction(func):

        @wraps(func)
        async def with_knx(
            hass: HomeAssistant,
            connection: websocket_api.ActiveConnection,
            msg: dict[str, Any],
        ) -> None:
            """Add KNX Module to call function."""
            try:
                knx = hass.data[KNX_MODULE_KEY]
            except KeyError:
                _send_not_loaded_error(connection, msg["id"])
                return
            await func(hass, knx, connection, msg)

    else:

        @wraps(func)
        def with_knx(
            hass: HomeAssistant,
            connection: websocket_api.ActiveConnection,
            msg: dict[str, Any],
        ) -> None:
            """Add KNX Module to call function."""
            try:
                knx = hass.data[KNX_MODULE_KEY]
            except KeyError:
                _send_not_loaded_error(connection, msg["id"])
                return
            func(hass, knx, connection, msg)

    return with_knx


def vol_invalid_response(exc: vol.Invalid) -> dict[str, Any]:
    """Format a Voluptuous validation exception into a structured error response."""
    errors: list[dict[str, Any]] = [
        {"path": [str(p) for p in error.path], "error": error.error_message}
        for error in (exc.errors if isinstance(exc, vol.MultipleInvalid) else [exc])
    ]

    return {
        "success": False,
        "errors": errors,
    }


@websocket_api.require_admin
@websocket_api.websocket_command(
    {
        vol.Required("type"): "knx/info",
    }
)
@provide_knx
@callback
def ws_info(
    hass: HomeAssistant,
    knx: KNXModule,
    connection: websocket_api.ActiveConnection,
    msg: dict,
) -> None:
    """Handle get info command."""
    _project_info = None
    if project_info := knx.project.info:
        _project_info = {
            "name": project_info["name"],
            "last_modified": project_info["last_modified"],
            "tool_version": project_info["tool_version"],
            "xknxproject_version": project_info["xknxproject_version"],
        }

    connection.send_result(
        msg["id"],
        {
            "version": knx.xknx.version,
            "connected": knx.xknx.connection_manager.connected.is_set(),
            "current_address": str(knx.xknx.current_address),
            "project": _project_info,
        },
    )


@websocket_api.require_admin
@websocket_api.websocket_command(
    {
        vol.Required("type"): "knx/get_knx_project",
    }
)
@websocket_api.async_response
@provide_knx
async def ws_get_knx_project(
    hass: HomeAssistant,
    knx: KNXModule,
    connection: websocket_api.ActiveConnection,
    msg: dict,
) -> None:
    """Handle get KNX project."""
    knxproject = await knx.project.get_knxproject()
    connection.send_result(
        msg["id"],
        {
            "project_loaded": knx.project.loaded,
            "knxproject": knxproject,
        },
    )


@websocket_api.require_admin
@websocket_api.websocket_command(
    {
        vol.Required("type"): "knx/project_file_process",
        vol.Required("file_id"): str,
        vol.Required("password"): str,
    }
)
@websocket_api.async_response
@provide_knx
async def ws_project_file_process(
    hass: HomeAssistant,
    knx: KNXModule,
    connection: websocket_api.ActiveConnection,
    msg: dict,
) -> None:
    """Handle get info command."""
    try:
        await knx.project.process_project_file(
            xknx=knx.xknx,
            file_id=msg["file_id"],
            password=msg["password"],
        )
    except (ValueError, XknxProjectException) as err:
        # ValueError could raise from file_upload integration
        connection.send_error(
            msg["id"], websocket_api.ERR_HOME_ASSISTANT_ERROR, str(err)
        )
        return

    connection.send_result(msg["id"])


@websocket_api.require_admin
@websocket_api.websocket_command(
    {
        vol.Required("type"): "knx/project_file_remove",
    }
)
@websocket_api.async_response
@provide_knx
async def ws_project_file_remove(
    hass: HomeAssistant,
    knx: KNXModule,
    connection: websocket_api.ActiveConnection,
    msg: dict,
) -> None:
    """Handle get info command."""
    await knx.project.remove_project_file()
    connection.send_result(msg["id"])


@websocket_api.require_admin
@websocket_api.websocket_command(
    {
        vol.Required("type"): "knx/group_monitor_info",
    }
)
@provide_knx
@callback
def ws_group_monitor_info(
    hass: HomeAssistant,
    knx: KNXModule,
    connection: websocket_api.ActiveConnection,
    msg: dict,
) -> None:
    """Handle get info command of group monitor."""
    recent_telegrams = [*knx.telegrams.recent_telegrams]
    connection.send_result(
        msg["id"],
        {
            "project_loaded": knx.project.loaded,
            "recent_telegrams": recent_telegrams,
        },
    )


@websocket_api.require_admin
@websocket_api.websocket_command(
    {
        vol.Required("type"): "knx/group_telegrams",
    }
)
@provide_knx
@callback
def ws_group_telegrams(
    hass: HomeAssistant,
    knx: KNXModule,
    connection: websocket_api.ActiveConnection,
    msg: dict,
) -> None:
    """Handle get group telegrams command."""
    connection.send_result(
        msg["id"],
        knx.telegrams.last_ga_telegrams,
    )


@websocket_api.require_admin
@websocket_api.websocket_command(
    {
        vol.Required("type"): "knx/subscribe_telegrams",
    }
)
@callback
def ws_subscribe_telegram(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict,
) -> None:
    """Subscribe to incoming and outgoing KNX telegrams."""

    @callback
    def forward_telegram(_telegram: Telegram, telegram_dict: TelegramDict) -> None:
        """Forward telegram to websocket subscription."""
        connection.send_event(
            msg["id"],
            telegram_dict,
        )

    connection.subscriptions[msg["id"]] = async_dispatcher_connect(
        hass,
        signal=SIGNAL_KNX_TELEGRAM,
        target=forward_telegram,
    )
    connection.send_result(msg["id"])


@websocket_api.require_admin
@websocket_api.websocket_command(
    {
        vol.Required("type"): "knx/validate_entity",
        **CREATE_ENTITY_BASE_SCHEMA,
    }
)
@callback
def ws_validate_entity(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict,
) -> None:
    """Validate entity data."""
    try:
        validate_entity_data(msg)
    except EntityStoreValidationException as exc:
        connection.send_result(msg["id"], exc.validation_error)
        return
    connection.send_result(
        msg["id"], EntityStoreValidationSuccess(success=True, entity_id=None)
    )


@websocket_api.require_admin
@websocket_api.websocket_command(
    {
        vol.Required("type"): "knx/create_entity",
        **CREATE_ENTITY_BASE_SCHEMA,
    }
)
@websocket_api.async_response
@provide_knx
async def ws_create_entity(
    hass: HomeAssistant,
    knx: KNXModule,
    connection: websocket_api.ActiveConnection,
    msg: dict,
) -> None:
    """Create entity in entity store and load it."""
    try:
        validated_data = validate_entity_data(msg)
    except EntityStoreValidationException as exc:
        connection.send_result(msg["id"], exc.validation_error)
        return
    try:
        entity_id = await knx.config_store.create_entity(
            # use validation result so defaults are applied
            validated_data[CONF_PLATFORM],
            validated_data[CONF_DATA],
        )
    except ConfigStoreException as err:
        connection.send_error(
            msg["id"], websocket_api.const.ERR_HOME_ASSISTANT_ERROR, str(err)
        )
        return
    connection.send_result(
        msg["id"], EntityStoreValidationSuccess(success=True, entity_id=entity_id)
    )


@websocket_api.require_admin
@websocket_api.websocket_command(
    {
        vol.Required("type"): "knx/update_entity",
        **UPDATE_ENTITY_BASE_SCHEMA,
    }
)
@websocket_api.async_response
@provide_knx
async def ws_update_entity(
    hass: HomeAssistant,
    knx: KNXModule,
    connection: websocket_api.ActiveConnection,
    msg: dict,
) -> None:
    """Update entity in entity store and reload it."""
    try:
        validated_data = validate_entity_data(msg)
    except EntityStoreValidationException as exc:
        connection.send_result(msg["id"], exc.validation_error)
        return
    try:
        await knx.config_store.update_entity(
            validated_data[CONF_PLATFORM],
            validated_data[CONF_ENTITY_ID],
            validated_data[CONF_DATA],
        )
    except ConfigStoreException as err:
        connection.send_error(
            msg["id"], websocket_api.const.ERR_HOME_ASSISTANT_ERROR, str(err)
        )
        return
    connection.send_result(
        msg["id"], EntityStoreValidationSuccess(success=True, entity_id=None)
    )


@websocket_api.require_admin
@websocket_api.websocket_command(
    {
        vol.Required("type"): "knx/delete_entity",
        vol.Required(CONF_ENTITY_ID): str,
    }
)
@websocket_api.async_response
@provide_knx
async def ws_delete_entity(
    hass: HomeAssistant,
    knx: KNXModule,
    connection: websocket_api.ActiveConnection,
    msg: dict,
) -> None:
    """Delete entity from entity store and remove it."""
    try:
        await knx.config_store.delete_entity(msg[CONF_ENTITY_ID])
    except ConfigStoreException as err:
        connection.send_error(
            msg["id"], websocket_api.const.ERR_HOME_ASSISTANT_ERROR, str(err)
        )
        return
    connection.send_result(msg["id"])


@websocket_api.require_admin
@websocket_api.websocket_command(
    {
        vol.Required("type"): "knx/get_entity_entries",
    }
)
@provide_knx
@callback
def ws_get_entity_entries(
    hass: HomeAssistant,
    knx: KNXModule,
    connection: websocket_api.ActiveConnection,
    msg: dict,
) -> None:
    """Get entities configured from entity store."""
    entity_entries = [
        entry.extended_dict for entry in knx.config_store.get_entity_entries()
    ]
    connection.send_result(msg["id"], entity_entries)


@websocket_api.require_admin
@websocket_api.websocket_command(
    {
        vol.Required("type"): "knx/get_entity_config",
        vol.Required(CONF_ENTITY_ID): str,
    }
)
@provide_knx
@callback
def ws_get_entity_config(
    hass: HomeAssistant,
    knx: KNXModule,
    connection: websocket_api.ActiveConnection,
    msg: dict,
) -> None:
    """Get entity configuration from entity store."""
    try:
        config_info = knx.config_store.get_entity_config(msg[CONF_ENTITY_ID])
    except ConfigStoreException as err:
        connection.send_error(
            msg["id"], websocket_api.const.ERR_HOME_ASSISTANT_ERROR, str(err)
        )
        return
    connection.send_result(msg["id"], config_info)


@websocket_api.require_admin
@websocket_api.websocket_command(
    {
        vol.Required("type"): "knx/create_device",
        vol.Required("name"): str,
        vol.Optional("area_id"): str,
    }
)
@provide_knx
@callback
def ws_create_device(
    hass: HomeAssistant,
    knx: KNXModule,
    connection: websocket_api.ActiveConnection,
    msg: dict,
) -> None:
    """Create a new KNX device."""
    identifier = f"knx_vdev_{ulid_now()}"
    device_registry = dr.async_get(hass)
    _device = device_registry.async_get_or_create(
        config_entry_id=knx.entry.entry_id,
        manufacturer="KNX",
        name=msg["name"],
        identifiers={(DOMAIN, identifier)},
    )
    device_registry.async_update_device(
        _device.id,
        area_id=msg.get("area_id") or UNDEFINED,
        configuration_url=f"homeassistant://knx/entities/view?device_id={_device.id}",
    )
    connection.send_result(msg["id"], _device.dict_repr)


@websocket_api.require_admin
@websocket_api.websocket_command(
    {
        vol.Required("type"): "knx/get_entity_schemas",
    }
)
@provide_knx
@callback
def ws_get_entity_schemas(
    hass: HomeAssistant,
    knx: KNXModule,
    connection: websocket_api.ActiveConnection,
    msg: dict,
) -> None:
    """Retrieve schema definitions for generating dynamic forms in the UI.

    This WebSocket command provides schema definitions required for configuring
    KNX entities in Home Assistant. The schemas are designed to enable the frontend
    to dynamically generate forms for user input, ensuring that the configuration
    matches the expected structure and validation rules.

    The schemas are retrieved dynamically using the `get_schema` method from
    configuration classes (e.g., `UiSensorConfig`). They are serialized into a
    JSON-compatible format via the `SchemaSerializer.convert` method and sent to
    the frontend through the WebSocket connection.

    Args:
        hass: The Home Assistant core object.
        knx: The KNX integration module, provided via the
             `@provide_knx` decorator.
        connection: The WebSocket connection
                    instance to communicate with the frontend.
        msg: The incoming WebSocket message containing the command details.

    Raises:
        ValueError: If there is an issue serializing the schemas.

    """

    supportedConfigs: tuple[type[EntityConfiguration], ...] = (UiSensorConfig,)

    payload = tuple(
        SchemaSerializer.convert(config.get_schema()) for config in supportedConfigs
    )

    connection.send_result(msg["id"], payload)


@cache
def get_entity_config_class(platform: str) -> type[EntityConfiguration] | None:
    """Map supported platform types to their configuration classes."""
    supported_config_classes: dict[str, type[EntityConfiguration]] = {
        str(Platform.SENSOR): UiSensorConfig,
    }
    return supported_config_classes.get(platform)


@websocket_api.require_admin
@websocket_api.websocket_command(
    {
        vol.Required("type"): "knx/create_platform_entity",
        vol.Required(CONF_DATA): vol.Schema(
            {CONF_PLATFORM: str}, extra=vol.ALLOW_EXTRA
        ),
    }
)
@websocket_api.async_response
@provide_knx
async def ws_create_platform_entity(
    hass: HomeAssistant,
    knx: KNXModule,
    connection: websocket_api.ActiveConnection,
    msg: dict,
) -> None:
    """Create an entity from the given configuration data via WebSocket.

    This function retrieves the configuration data from the incoming message,
    identifies the appropriate configuration class based on the platform, and
    attempts to create the corresponding entity in the KNX config store. If
    successful, a success response containing the new entity's ID is sent back
    to the client; otherwise, an error response is returned.

    Args:
        hass: The Home Assistant instance.
        knx: An instance of the KNXModule, which provides access to the KNX config store.
        connection: The active WebSocket connection used for sending responses.
        msg: A dictionary containing the incoming WebSocket message data.

    Returns:
        None. Responses are sent back to the WebSocket client via `connection.send_result`
        or `connection.send_error`.

    Raises:
        TypeError: If the resolved configuration object is not an instance of ``Persistable``.

    """
    # Collect frequently accessed fields.
    message_id: int = msg["id"]
    data: dict[str, Any] = msg[CONF_DATA]
    platform: str = data[CONF_PLATFORM]

    if not (config_class := get_entity_config_class(platform)):
        connection.send_error(
            message_id,
            websocket_api.const.ERR_NOT_SUPPORTED,
            f"Unsupported platform: '{platform}'",
        )
        return

    # Validate and construct config.
    try:
        config = config_class.from_dict(data)

    except vol.Invalid as exc:
        connection.send_result(message_id, vol_invalid_response(exc))
        return

    # Ensure the config is persistable, then create the entity.
    if not isinstance(config, Persistable):
        connection.send_error(
            message_id,
            websocket_api.const.ERR_INVALID_FORMAT,
            f"Config class {config_class} must implement Persistable",
        )
        return

    try:
        entity_id = await knx.config_store.create_entity(
            platform, config.to_storage_dict()
        )
    except ConfigStoreException as err:
        connection.send_error(
            message_id,
            websocket_api.const.ERR_HOME_ASSISTANT_ERROR,
            str(err),
        )
        return

    # Send success response with the new entity ID.
    connection.send_result(
        message_id,
        EntityStoreValidationSuccess(success=True, entity_id=entity_id),
    )
