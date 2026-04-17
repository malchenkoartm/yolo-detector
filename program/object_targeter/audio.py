import sounddevice as sd
import vosk
import threading
import json
import time
import os
import sys
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
        self.__model_name = model_name
        self.__model = None
        self.__voice_enabled = True
        try:
            self.__model = vosk.Model(model_name)
        except Exception as e:
            self.__voice_enabled = False
            if self.__logger:
                model_hint = os.path.abspath(model_name)
                self.__logger.warning(f"[Audio] Vosk model is unavailable: {e}")
                self.__logger.warning(f"[Audio] Voice control disabled. Expected model dir: {model_hint}")
        self.__listen_thread = None
        self.__input_channels = 1
        self.__dtype = "int16"
        self.__level = 0.0
        if self.__device is None:
            pref = self.__preferred_windows_input_device()
            if pref is not None:
                self.__device = pref

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

    def __preferred_windows_input_device(self) -> int | None:
        if not sys.platform.startswith("win"):
            return None
        try:
            hostapis = sd.query_hostapis()
            name_to_idx = {str(h.get("name", "")).lower(): i for i, h in enumerate(hostapis)}
            preferred_hosts = [
                name_to_idx.get("windows directsound"),
                name_to_idx.get("windows wdm-ks"),
                name_to_idx.get("windows wasapi"),
                name_to_idx.get("mme"),
            ]
            all_devs = sd.query_devices()
            for hidx in preferred_hosts:
                if hidx is None:
                    continue
                for idx, d in enumerate(all_devs):
                    if int(d.get("max_input_channels", 0)) > 0 and int(d.get("hostapi", -1)) == int(hidx):
                        return idx
        except Exception:
            return None
        return None

    def __device_candidates(self, requested: int | str | None) -> list[int | str]:
        out: list[int | str] = []
        if requested is not None:
            out.append(requested)
        try:
            if self.__device is not None and self.__device not in out:
                out.append(self.__device)
        except Exception:
            pass
        try:
            pref = self.__preferred_windows_input_device()
            if pref is not None and pref not in out:
                out.append(pref)
        except Exception:
            pass
        try:
            all_devs = sd.query_devices()
            for idx, d in enumerate(all_devs):
                if int(d.get("max_input_channels", 0)) > 0 and idx not in out:
                    out.append(idx)
        except Exception:
            pass
        return out

    def __pick_working_stream_config(self, requested_dev: int | str | None) -> tuple[int | str | None, int, int, str] | None:
        rates = [int(self.__fs), 48000, 44100, 16000, 8000]
        dtypes = ["int16", "float32"]
        def _probe_cb(indata, frames, t, status):
            return
        for dev in self.__device_candidates(requested_dev):
            try:
                info = sd.query_devices(dev)
                max_ch = max(1, int(info.get("max_input_channels", 1) or 1))
                default_sr = int(float(info.get("default_samplerate", 0)) or 0)
            except Exception:
                max_ch = 1
                default_sr = 0
            local_rates = rates.copy()
            if default_sr and default_sr not in local_rates:
                local_rates.insert(1, default_sr)
            for r in local_rates:
                for ch in [1, min(2, max_ch)]:
                    for dt in dtypes:
                        try:
                            sd.check_input_settings(device=dev, channels=ch, samplerate=int(r), dtype=dt)
                            stream = sd.InputStream(
                                callback=_probe_cb,
                                channels=ch,
                                samplerate=int(r),
                                dtype=dt,
                                blocksize=0,
                                device=dev,
                            )
                            stream.start()
                            stream.stop()
                            stream.close()
                            return dev, int(r), int(ch), dt
                        except Exception:
                            continue
        return None

    def __query_input_device_info(self, device: int | str | None):
        if device is not None:
            return sd.query_devices(device)
        win_pref = self.__preferred_windows_input_device()
        if win_pref is not None:
            return sd.query_devices(win_pref)
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

    def get_level(self) -> float:
        with self.__audio_lock:
            return float(self.__level)
            
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
        fail_count = 0

        if not self.__voice_enabled or self.__model is None:
            if self.__logger:
                self.__logger.warning("[Audio] Listen loop idle: voice model not loaded")
            while not self._stop_event.is_set():
                time.sleep(0.25)
            return

        TRIGGERS = self.__parser.get_triggers()

        handled = False

        def callback(indata, frames, t, status):
            nonlocal handled
            if self._stop_event.is_set():
                return

            # Vosk expects mono stream; convert to mono if needed.
            if getattr(indata, "ndim", 1) > 1 and indata.shape[1] > 1:
                indata = indata[:, :1]

            # Apply mixer settings (gain + simple RMS gate)
            with self.__audio_lock:
                gain = float(self.__gain)
                thr = float(self.__gate_threshold)

            try:
                rms_now = float(np.sqrt(np.mean(indata.astype(np.float32) ** 2)) / 32768.0)
                with self.__audio_lock:
                    # Soft smoothing to avoid jitter in the meter.
                    self.__level = 0.80 * float(self.__level) + 0.20 * min(1.0, rms_now * 8.0)
            except Exception:
                pass

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
                channels = int(self.__input_channels)
                dtype = str(self.__dtype)

            cfg = self.__pick_working_stream_config(dev)
            if cfg is None:
                fail_count += 1
                if self.__logger and fail_count in (1, 5):
                    self.__logger.warning("[Audio] no working microphone stream configuration found")
                with self.__audio_lock:
                    self.__level = 0.0
                time.sleep(min(5.0, 0.4 * (2 ** min(fail_count, 4))))
                continue

            dev, fs, channels, dtype = cfg
            with self.__audio_lock:
                self.__device = dev
                self.__fs = int(fs)
                self.__input_channels = int(channels)
                self.__dtype = str(dtype)

            try:
                rec = vosk.KaldiRecognizer(self.__model, fs)
                with sd.InputStream(
                    callback=callback,
                    channels=channels,
                    samplerate=fs,
                    dtype=dtype,
                    blocksize=0,
                    device=dev,
                ):
                    self._stop_event.wait(timeout=0.25)
                fail_count = 0
            except sd.PortAudioError as e:
                msg = str(e)
                if self.__logger:
                    self.__logger.warning(f"[Audio] PortAudioError: {msg}")

                fail_count += 1
                backoff = min(5.0, 0.2 * (2 ** min(fail_count, 5)))
                if fail_count >= 8 and self.__logger:
                    self.__logger.warning("[Audio] microphone unavailable for a while; keeping retries with backoff")
                with self.__audio_lock:
                    self.__level = 0.0
                time.sleep(backoff)
            except Exception as e:
                if self.__logger:
                    self.__logger.error(f"[Audio] listen loop error: {e}")
                with self.__audio_lock:
                    self.__level = 0.0
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