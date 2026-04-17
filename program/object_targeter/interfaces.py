from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass
from enum import Enum, auto

class IAngleCalculator(ABC):
    @abstractmethod
    def calculate(self, x: float, y: float) -> tuple[float, float]: ...

class ILogger(ABC):
    @abstractmethod
    def trace(self, msg: str): ...
    @abstractmethod
    def info(self, msg: str): ...
    @abstractmethod
    def warning(self, msg: str): ...
    @abstractmethod
    def error(self, msg: str): ...
    

@dataclass
class WorkConfig:
    names: dict[str, str]
    names_updated: bool
    conf: float
    target_track: int | None
    to_work: bool
    

class IWorkConfigManager(ABC):
    @abstractmethod
    def place(self, ru_text: str): ...

    @abstractmethod
    def add(self, ru_text: str): ...

    @property
    @abstractmethod
    def config(self) -> WorkConfig: ...

    @abstractmethod
    def update_names(self) -> WorkConfig: ...

    @property
    @abstractmethod
    def target_track(self) -> int | None: ...

    @target_track.setter
    @abstractmethod
    def target_track(self, value: int | None): ...

    @property
    @abstractmethod
    def conf(self) -> float: ...

    @conf.setter
    @abstractmethod
    def conf(self, value: float): ...

    @property
    @abstractmethod
    def colors(self): ...

    @property
    @abstractmethod
    def to_work(self) -> bool: ...

    @abstractmethod
    def stop(self): ...
    
    

class CommandType(Enum):
    ZOOM_IN  = auto()
    ZOOM_OUT = auto()
    PLACE    = auto()
    ADD      = auto()
    FOLLOW   = auto()
    EXIT     = auto()
    UNKNOWN  = auto()
    CONF     = auto()

@dataclass
class Command:
    type: CommandType
    text: str = ""

class ICommandParser(ABC):
    @abstractmethod
    def get_triggers(cls) -> list[str]: ...
    @abstractmethod
    def parse(self, text: str, to_add: bool = False) -> Command: ...


class ISerialWriter(ABC):
    @property
    @abstractmethod
    def not_send_zone(self) -> tuple[float, float]: ...

    @not_send_zone.setter
    @abstractmethod
    def not_send_zone(self, zone: tuple[float, float]): ...

    @abstractmethod
    def update_notsend_zone_by_size(self, size: tuple[int, int]): ...

    @property
    @abstractmethod
    def coords(self) -> tuple[int, int] | None: ...

    @coords.setter
    @abstractmethod
    def coords(self, coords: tuple[int, int] | None): ...

    @abstractmethod
    def start(self): ...

    @abstractmethod
    def stop(self): ...

class IZoomController(ABC):
    @abstractmethod
    def zoom_in(self): ...

    @abstractmethod
    def zoom_out(self): ...

    @property
    @abstractmethod
    def zoom(self) -> int: ...

    @abstractmethod
    def get_state(self) -> tuple[int, tuple[int, int]]: ...

    @abstractmethod
    def apply(self, frame: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def to_original_coords(self, cx: int, cy: int) -> tuple[int, int]: ...

class IPreprocessor(ABC):
    @abstractmethod
    def apply(self, frame: np.ndarray) -> np.ndarray: ...

class ISmoothingFilter(ABC):
    @abstractmethod
    def update(self, coords: tuple[int, int] | None) -> tuple[int, int] | None: ...

    @abstractmethod
    def reset(self): ...

class IObjectSelector(ABC):
    @staticmethod
    @abstractmethod
    def select(results, mode: str = 'first',
               target: int | None = None) -> tuple[int, int] | None: ...

class IAnalyzer(ABC):
    @abstractmethod
    def get_results(self): ...


class IIOOperator(ABC):
    @property
    @abstractmethod
    def analyzer(self) -> IAnalyzer | None: ...

    @analyzer.setter
    @abstractmethod
    def analyzer(self, analyzer: IAnalyzer): ...

    @abstractmethod
    def get_latest_raw(self) -> np.ndarray | None: ...

    @abstractmethod
    def start(self): ...

    @abstractmethod
    def stop(self): ...


class GUIEventType(Enum):
    EXIT = auto()
    VIDEO_SOURCE_CHANGED = auto()
    AUDIO_SOURCE_CHANGED = auto()
    AUDIO_SETTINGS_CHANGED = auto()
    CLASS_NAMES_CHANGED = auto()
    TOGGLE_WORK = auto()
    CONF_CHANGED = auto()
    TARGET_TRACK_CHANGED = auto()
    ZOOM_IN = auto()
    ZOOM_OUT = auto()
    UNKNOWN = auto()


@dataclass(frozen=True)
class DeviceInfo:
    id: str
    label: str


@dataclass(frozen=True)
class GUIEvent:
    type: GUIEventType
    value: object | None = None


class IIOOperatorGUI(ABC):
    """
    Minimalistic GUI focused on video rendering with a compact control panel.

    This interface is intentionally UI-toolkit agnostic (OpenCV/Qt/GTK/etc.).
    Implementations should avoid noisy console output and prefer concise UI status.
    """

    @abstractmethod
    def start(self): ...

    @abstractmethod
    def stop(self): ...

    @abstractmethod
    def render_frame(self, frame: np.ndarray): ...

    @abstractmethod
    def set_overlay_text(self, lines: list[str]): ...

    @abstractmethod
    def set_status(self, text: str): ...

    @abstractmethod
    def list_video_devices(self) -> list[DeviceInfo]: ...

    @abstractmethod
    def list_audio_devices(self) -> list[DeviceInfo]: ...

    @abstractmethod
    def set_video_device(self, device_id: str): ...

    @abstractmethod
    def set_audio_device(self, device_id: str): ...

    @abstractmethod
    def poll_event(self) -> GUIEvent | None: ...


class IVideoAnalyzer(IAnalyzer):
    @abstractmethod
    def start(self): ...

    @abstractmethod
    def stop(self): ...


class IAudioRecorder(ABC):
    @abstractmethod
    def start(self): ...

    @abstractmethod
    def stop(self): ...
    
class IThreadManager(ABC):
    @abstractmethod
    def start(self): ...
 
    @abstractmethod
    def stop(self): ...
