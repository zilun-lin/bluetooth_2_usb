from dataclasses import dataclass
from typing import Sequence

from adafruit_hid import find_device
from evdev import InputDevice, InputEvent
from usb_hid import Device

from .evdev import ecodes
from .logging import get_logger

_logger = get_logger()

MAX_CONTACTS = 5
TOUCHPAD_REPORT_ID = 0x04
LOGICAL_MAX = 32767
# Per-contact: 1 (tip+pad) + 1 (id) + 2 (X) + 2 (Y) = 6 bytes
# Total: 5*6 + 1 (count) + 1 (button+pad) = 32 bytes
IN_REPORT_LENGTH = MAX_CONTACTS * 6 + 2


def _build_finger_collection() -> bytes:
    """Build the HID descriptor bytes for a single Finger logical collection."""
    return bytes((
        0x09, 0x22,        #   Usage (Finger)
        0xA1, 0x02,        #   Collection (Logical)

        # Tip Switch (1 bit)
        0x09, 0x42,        #     Usage (Tip Switch)
        0x15, 0x00,        #     Logical Minimum (0)
        0x25, 0x01,        #     Logical Maximum (1)
        0x75, 0x01,        #     Report Size (1)
        0x95, 0x01,        #     Report Count (1)
        0x81, 0x02,        #     Input (Data, Variable, Absolute)

        # Padding (7 bits)
        0x75, 0x07,        #     Report Size (7)
        0x95, 0x01,        #     Report Count (1)
        0x81, 0x03,        #     Input (Constant)

        # Contact Identifier (8 bits)
        0x09, 0x51,        #     Usage (Contact Identifier)
        0x75, 0x08,        #     Report Size (8)
        0x95, 0x01,        #     Report Count (1)
        0x15, 0x00,        #     Logical Minimum (0)
        0x25, MAX_CONTACTS - 1,  # Logical Maximum (4)
        0x81, 0x02,        #     Input (Data, Variable, Absolute)

        # X (16 bits)
        0x05, 0x01,        #     Usage Page (Generic Desktop)
        0x09, 0x30,        #     Usage (X)
        0x75, 0x10,        #     Report Size (16)
        0x95, 0x01,        #     Report Count (1)
        0x15, 0x00,        #     Logical Minimum (0)
        0x26, 0xFF, 0x7F,  #     Logical Maximum (32767)
        0x55, 0x0E,        #     Unit Exponent (-2)
        0x65, 0x11,        #     Unit (cm)
        0x35, 0x00,        #     Physical Minimum (0)
        0x46, 0x40, 0x06,  #     Physical Maximum (1600 = 16.00 cm)
        0x81, 0x02,        #     Input (Data, Variable, Absolute)

        # Y (16 bits)
        0x09, 0x31,        #     Usage (Y)
        0x46, 0x7E, 0x04,  #     Physical Maximum (1150 = 11.50 cm)
        0x81, 0x02,        #     Input (Data, Variable, Absolute)

        # Reset global items so they don't leak into subsequent fields
        0x55, 0x00,        #     Unit Exponent (0)
        0x65, 0x00,        #     Unit (None)
        0x45, 0x00,        #     Physical Maximum (0) - undefined

        # End Finger collection
        0x05, 0x0D,        #     Usage Page (Digitizer) - restore for next finger
        0xC0,              #   End Collection
    ))


def _build_touchpad_descriptor() -> bytes:
    """Build the complete HID report descriptor for a multitouch touchpad."""
    header = bytes((
        0x05, 0x0D,        # Usage Page (Digitizer)
        0x09, 0x05,        # Usage (Touch Pad)
        0xA1, 0x01,        # Collection (Application)
        0x85, TOUCHPAD_REPORT_ID,  # Report ID (4)
    ))

    fingers = b""
    for _ in range(MAX_CONTACTS):
        fingers += _build_finger_collection()

    footer = bytes((
        # Contact Count (8 bits)
        0x05, 0x0D,        #   Usage Page (Digitizer)
        0x09, 0x54,        #   Usage (Contact Count)
        0x75, 0x08,        #   Report Size (8)
        0x95, 0x01,        #   Report Count (1)
        0x15, 0x00,        #   Logical Minimum (0)
        0x25, MAX_CONTACTS, #  Logical Maximum (5)
        0x81, 0x02,        #   Input (Data, Variable, Absolute)

        # Button 1 (1 bit)
        0x05, 0x09,        #   Usage Page (Button)
        0x09, 0x01,        #   Usage (Button 1)
        0x15, 0x00,        #   Logical Minimum (0)
        0x25, 0x01,        #   Logical Maximum (1)
        0x75, 0x01,        #   Report Size (1)
        0x95, 0x01,        #   Report Count (1)
        0x81, 0x02,        #   Input (Data, Variable, Absolute)

        # Padding (7 bits)
        0x75, 0x07,        #   Report Size (7)
        0x95, 0x01,        #   Report Count (1)
        0x81, 0x03,        #   Input (Constant)

        0xC0,              # End Collection
    ))

    return header + fingers + footer


TOUCHPAD_DESCRIPTOR = _build_touchpad_descriptor()

TOUCHPAD_DEVICE = Device(
    descriptor=TOUCHPAD_DESCRIPTOR,
    usage_page=0x0D,
    usage=0x05,
    report_ids=(TOUCHPAD_REPORT_ID,),
    in_report_lengths=(IN_REPORT_LENGTH,),
    out_report_lengths=(0,),
    name="touchpad",
)


def is_multitouch_device(device: InputDevice) -> bool:
    """
    Check whether an evdev InputDevice supports multitouch Protocol B
    by looking for ABS_MT_POSITION_X in its capabilities.

    :param device: The evdev input device to check
    :return: True if the device supports multitouch
    :rtype: bool
    """
    try:
        caps = device.capabilities()
        abs_caps = caps.get(ecodes.EV_ABS, [])
        abs_codes = set()
        for item in abs_caps:
            if isinstance(item, tuple):
                abs_codes.add(item[0])
            else:
                abs_codes.add(item)
        return ecodes.ABS_MT_POSITION_X in abs_codes
    except Exception:
        return False


@dataclass
class ContactSlot:
    """State of a single finger contact in a multitouch slot."""

    active: bool = False
    tracking_id: int = -1
    x: int = 0
    y: int = 0


class MultitouchState:
    """
    Per-device multitouch state machine for Linux Protocol B (slot-based).

    Tracks up to MAX_CONTACTS finger slots. ABS_MT_SLOT selects the active slot,
    subsequent ABS_MT_* events modify it. On SYN_REPORT the accumulated frame
    is ready to be sent as a single HID report.
    """

    def __init__(self, device: InputDevice) -> None:
        """
        :param device: The evdev input device (used to read axis ranges)
        """
        self._slots: list[ContactSlot] = [
            ContactSlot() for _ in range(MAX_CONTACTS)
        ]
        self._current_slot: int = 0
        self._button_left: bool = False
        self._dirty: bool = False

        self._x_min, self._x_max = self._get_axis_range(
            device, ecodes.ABS_MT_POSITION_X
        )
        self._y_min, self._y_max = self._get_axis_range(
            device, ecodes.ABS_MT_POSITION_Y
        )
        _logger.debug(
            f"MultitouchState: X range [{self._x_min}, {self._x_max}], "
            f"Y range [{self._y_min}, {self._y_max}]"
        )

    @staticmethod
    def _get_axis_range(device: InputDevice, axis_code: int) -> tuple[int, int]:
        """
        Read the min/max range for an absolute axis from the device capabilities.

        :param device: The evdev input device
        :param axis_code: The ABS_* axis code
        :return: (min, max) tuple
        """
        try:
            caps = device.capabilities(absinfo=True)
            abs_caps = caps.get(ecodes.EV_ABS, [])
            for code, absinfo in abs_caps:
                if code == axis_code:
                    return absinfo.min, absinfo.max
        except Exception:
            pass
        return 0, LOGICAL_MAX

    def process_event(self, input_event: InputEvent) -> bool:
        """
        Feed a raw InputEvent into the state machine.

        :param input_event: The evdev InputEvent
        :return: True when SYN_REPORT is received (a complete frame is ready)
        :rtype: bool
        """
        ev_type = input_event.type
        code = input_event.code
        value = input_event.value

        if ev_type == ecodes.EV_ABS:
            self._handle_abs(code, value)
        elif ev_type == ecodes.EV_KEY:
            self._handle_key(code, value)
        elif ev_type == ecodes.EV_SYN and code == ecodes.SYN_REPORT:
            if self._dirty:
                self._dirty = False
                return True

        return False

    def _handle_abs(self, code: int, value: int) -> None:
        if code == ecodes.ABS_MT_SLOT:
            if 0 <= value < MAX_CONTACTS:
                self._current_slot = value
            return

        slot = self._slots[self._current_slot]
        self._dirty = True

        if code == ecodes.ABS_MT_TRACKING_ID:
            if value == -1:
                slot.active = False
                slot.tracking_id = -1
                slot.x = 0
                slot.y = 0
            else:
                slot.active = True
                slot.tracking_id = value
        elif code == ecodes.ABS_MT_POSITION_X:
            slot.x = value
        elif code == ecodes.ABS_MT_POSITION_Y:
            slot.y = value

    def _handle_key(self, code: int, value: int) -> None:
        if code == ecodes.BTN_LEFT:
            self._button_left = bool(value)
            self._dirty = True

    @property
    def slots(self) -> list[ContactSlot]:
        return self._slots

    @property
    def button_left(self) -> bool:
        return self._button_left

    @property
    def contact_count(self) -> int:
        return sum(1 for s in self._slots if s.active)

    def scale_x(self, raw: int) -> int:
        """Scale a raw device X coordinate to 0-32767."""
        span = self._x_max - self._x_min
        if span <= 0:
            return 0
        return max(0, min(LOGICAL_MAX, (raw - self._x_min) * LOGICAL_MAX // span))

    def scale_y(self, raw: int) -> int:
        """Scale a raw device Y coordinate to 0-32767."""
        span = self._y_max - self._y_min
        if span <= 0:
            return 0
        return max(0, min(LOGICAL_MAX, (raw - self._y_min) * LOGICAL_MAX // span))


class TouchpadGadget:
    """
    Wraps a usb_hid.Device for sending multitouch touchpad HID reports.
    Follows the same pattern as adafruit_hid.mouse.Mouse.
    """

    def __init__(self, devices: Sequence) -> None:
        """
        :param devices: Sequence of usb_hid.Device objects to search
        """
        self._device = find_device(devices, usage_page=0x0D, usage=0x05)
        self._report = bytearray(IN_REPORT_LENGTH)

    def send_report(self, state: MultitouchState) -> None:
        """
        Pack the current MultitouchState into the HID report and send it.

        :param state: The multitouch state machine with current slot data
        """
        report = self._report

        for i in range(MAX_CONTACTS):
            offset = i * 6
            slot = state.slots[i]

            if slot.active:
                report[offset] = 0x01
                report[offset + 1] = i
                x = state.scale_x(slot.x)
                y = state.scale_y(slot.y)
                report[offset + 2] = x & 0xFF
                report[offset + 3] = (x >> 8) & 0xFF
                report[offset + 4] = y & 0xFF
                report[offset + 5] = (y >> 8) & 0xFF
            else:
                report[offset] = 0
                report[offset + 1] = 0
                report[offset + 2] = 0
                report[offset + 3] = 0
                report[offset + 4] = 0
                report[offset + 5] = 0

        count_offset = MAX_CONTACTS * 6
        report[count_offset] = state.contact_count
        report[count_offset + 1] = 0x01 if state.button_left else 0x00

        self._device.send_report(self._report, TOUCHPAD_REPORT_ID)

    def release_all(self) -> None:
        """Send an empty report (no contacts, no buttons)."""
        for i in range(len(self._report)):
            self._report[i] = 0
        self._device.send_report(self._report, TOUCHPAD_REPORT_ID)
