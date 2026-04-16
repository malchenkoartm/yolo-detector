import sounddevice as sd
import vosk
import threading
import json
import time
from ru_word2number import w2n
import numpy as np

from interfaces import IWorkConfigManager, IZoomController, CommandType, ICommandParser, ILogger, IAudioRecorder

from threadmanager import ThreadManager

class AudioRecorder(ThreadManager, IAudioRecorder):
    def __init__(self, config_manager: IWorkConfigManager, zoom: IZoomController, parser: ICommandParser,
                 logger: ILogger | None = None,
                 model_name = "vosk-model-small-ru-0.22",
                 fs = 44100,
                 device: int | str | None = None):
        super().__init__()

        self.__config_manager = config_manager
        self.__zoom = zoom
        self.__parser = parser
        self.__logger = logger

        self.__fs = fs
        self.__device = self.__normalize_device(device)
        self.__gain = 1.0
        self.__gate_threshold = 0.0
        self.__audio_lock = threading.Lock()

        self.__model = vosk.Model(model_name)
        self.__listen_thread = None

        if self.__logger:
            self.__logger.info(f"fs: {self.__fs}")
            try:
                info = self.__query_input_device_info(self.__device)
                self.__logger.info(f"Микрофон: {info['name']}")
            except Exception as e:
                self.__logger.warning(f"[Audio] не удалось определить микрофон: {e}")

    def __normalize_device(self, device: int | str | None) -> int | str | None:
        if device is None:
            return None
        if isinstance(device, str):
            value = device.strip()
            if not value:
                return None
            if value.isdigit():
                return int(value)
            return value
        return int(device)

    def __query_input_device_info(self, device: int | str | None):
        if device is not None:
            return sd.query_devices(device)
        default_in = sd.default.device[0] if sd.default.device else None
        if default_in is not None and default_in >= 0:
            return sd.query_devices(default_in)
        all_devs = sd.query_devices()
        for idx, d in enumerate(all_devs):
            if int(d.get("max_input_channels", 0)) > 0:
                return sd.query_devices(idx)
        raise RuntimeError("input device not found")

    def set_gain(self, gain: float):
        with self.__audio_lock:
            self.__gain = max(0.1, float(gain))

    def set_gate_threshold(self, thr: float):
        with self.__audio_lock:
            self.__gate_threshold = max(0.0, float(thr))

    def set_fs(self, fs: int):
        with self.__audio_lock:
            self.__fs = int(fs)
            
    def __words_to_num(self, text: str) -> int | float | None:
        try:
            return w2n.word_to_num(text)
        except Exception as e:
            if self.__logger:
                self.__logger.warning(f"Convert words to num exception {e}")
        return None

    def __handle_command(self, command, text):
        if self.__logger:
            self.__logger.info(f"Команда: {command.type.name} — '{text}'")
        if command.type == CommandType.ZOOM_IN:
            self.__zoom.zoom_in()
        elif command.type == CommandType.ZOOM_OUT:
            self.__zoom.zoom_out()
        elif command.type == CommandType.EXIT:
            self.__config_manager.stop()
        elif command.type == CommandType.ADD:
            self.__config_manager.add(command.text)
        elif command.type == CommandType.PLACE:
            self.__config_manager.place(command.text)
        elif command.type == CommandType.FOLLOW:
            self.__config_manager.target_track = self.__words_to_num(command.text)
        elif command.type == CommandType.CONF:
            conf = self.__words_to_num(command.text)
            if conf:
                self.__config_manager.conf = conf / 100
            

    def __listen_loop(self):
        rec = None
        handled = False

        TRIGGERS = self.__parser.get_triggers()

        handled = False

        def callback(indata, frames, t, status):
            nonlocal handled
            if self._stop_event.is_set():
                return

            # Apply mixer settings (gain + simple RMS gate)
            with self.__audio_lock:
                gain = float(self.__gain)
                thr = float(self.__gate_threshold)

            if thr > 0:
                rms = float(np.sqrt(np.mean(indata.astype(np.float32) ** 2)) / 32768.0)
                if rms < thr:
                    return

            if gain != 1.0:
                x = indata.astype(np.float32) * gain
                x = np.clip(x, -32768, 32767).astype(np.int16)
                buf = x.tobytes()
            else:
                buf = indata.tobytes()

            if rec is None:
                return

            if rec.AcceptWaveform(buf):
                text = json.loads(rec.Result()).get("text", "").strip().lower()
                handled = False

                if not text:
                    return

                if self.__logger:
                    self.__logger.info(f"Found text {text}")

                for trigger in TRIGGERS:
                    idx = text.find(trigger)
                    if idx != -1:
                        after = text[idx + len(trigger):].strip()
                        full = trigger + (" " + after if after else "")
                        command = self.__parser.parse(full)
                        if command.type != CommandType.UNKNOWN:
                            self.__handle_command(command, full)
                        break

        while not self._stop_event.is_set():
            with self.__audio_lock:
                fs = int(self.__fs)
                dev = self.__normalize_device(self.__device)

            try:
                rec = vosk.KaldiRecognizer(self.__model, fs)
                with sd.InputStream(
                    callback=callback,
                    channels=1,
                    samplerate=fs,
                    dtype="int16",
                    blocksize=int(fs * 0.3),
                    device=dev,
                ):
                    self._stop_event.wait(timeout=0.25)
            except sd.PortAudioError as e:
                msg = str(e)
                if self.__logger:
                    self.__logger.warning(f"[Audio] PortAudioError: {msg}")

                # Try fallback sample rates if device rejects the current one.
                fallback = []
                try:
                    info = self.__query_input_device_info(dev)
                    d_fs = int(float(info.get("default_samplerate", 0)) or 0)
                    if d_fs:
                        fallback.append(d_fs)
                except Exception:
                    pass

                fallback += [16000, 44100, 48000, 8000]
                tried = set()
                switched = False
                for f in fallback:
                    if f in tried:
                        continue
                    tried.add(f)
                    with self.__audio_lock:
                        self.__fs = int(f)
                    switched = True
                    if self.__logger:
                        self.__logger.warning(f"[Audio] retry with fs={f}")
                    break

                # Invalid/unavailable input device fallback.
                if ("invalid device" in msg.lower() or "device unavailable" in msg.lower()):
                    with self.__audio_lock:
                        self.__device = None
                    switched = True
                    if self.__logger:
                        self.__logger.warning("[Audio] fallback to default input device")

                if not switched:
                    time.sleep(0.5)
            except Exception as e:
                if self.__logger:
                    self.__logger.error(f"[Audio] listen loop error: {e}")
                time.sleep(0.5)

        if self.__logger:
            self.__logger.info("Exit listen loop")

    def stop(self):
        self._stop_event.set()
        if self.__listen_thread and self.__listen_thread.is_alive():
            self.__listen_thread.join(timeout=3)
            if self.__listen_thread.is_alive() and self.__logger:
                self.__logger.warning("Listen поток не завершился вовремя.")

    def start(self):
        self._stop_event.clear()
        self.__listen_thread = threading.Thread(
            target=self.__listen_loop, daemon=False
        )
        self.__listen_thread.start()

        print("Голосовые команды:\n"
              "\t'найти <предмет>'       - заменить искомые объекты\n"
              "\t'добавить <предмет>'    - добавить объект\n"
              "\t'следить <id>'          - фокусировать вниание на находку id\n"
              "\t'уверенность <процент>' - задать порог уверенности для модели\n"
              "\t'приблизить'            - приблизить\n"
              "\t'отдалить'              - отдалить\n"
              "\t'стоп'                  - выйти\n")

        while self.__config_manager.to_work:
            time.sleep(0.1)

        self._stop_event.set()
        self.__listen_thread.join(timeout=3)
        if self.__listen_thread.is_alive() and self.__logger:
            self.__logger.warning("Listen поток не завершился вовремя.")

        if self.__logger:
            self.__logger.info("Exit Audio Recorder")