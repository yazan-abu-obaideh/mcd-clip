import asyncio
import logging
import os
import platform
import queue
import threading
import uuid
from asyncio import subprocess

from mcd_clip.bike_rendering.cad_builder import BikeCadFileBuilder
from mcd_clip.resource_utils import resource_path

TEMP_DIR = "bikes"

LOGGER_NAME = "BikeCadLogger"

WINDOWS = "Windows"

RENDERER_TIMEOUT = 30
RENDERER_TIMEOUT_GRANULARITY = 1


class RenderingService:
    def __init__(self, renderer_pool_size, cad_builder=BikeCadFileBuilder()):
        os.makedirs(os.path.join(os.path.dirname(__file__), TEMP_DIR), exist_ok=True)
        self._renderer_pool = queue.Queue(maxsize=renderer_pool_size)
        self.cad_builder = cad_builder
        for i in range(renderer_pool_size):
            self._renderer_pool.put(BikeCad())

    def render_object(self, bike_object, seed_bike_id):
        return self.render(self.cad_builder.build_cad_from_object(bike_object,
                                                                  seed_bike_id))

    def render(self, bike_xml):
        renderer = self._get_renderer()
        result = renderer.render(bike_xml)
        self._renderer_pool.put(renderer)  # This will never block as is - no new elements
        # are ever added, so the pool will always have room for borrowed renderers.
        return result

    def _get_renderer(self):
        return self._renderer_pool.get(timeout=RENDERER_TIMEOUT / 2)


class BikeCad:
    def __init__(self):
        self._expected_success = self._get_expected_success()
        self._event_loop_lock = threading.Lock()
        with self._event_loop_lock:
            self._event_loop = asyncio.new_event_loop()
            self._instance = self._event_loop.run_until_complete(self._init_instance())
        self._log_info("Started BikeCAD process!")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.kill()

    def render(self, bike_xml):
        bike_path = self._generate_bike_path()
        self._write_to_file(bike_path, bike_xml)
        self._export_svg_from_list([bike_path])
        os.remove(bike_path)
        image_path = bike_path.replace(".bcad", ".svg")
        image_bytes = self._read_image(image_path)
        os.remove(image_path)
        return image_bytes

    def _read_image(self, image_path):
        with open(image_path, "rb") as file:
            image_bytes = file.read()
        return image_bytes

    def _write_to_file(self, bike_path, bike_xml):
        with open(bike_path, "w") as file:
            file.write(bike_xml)

    def _generate_bike_path(self):
        return os.path.join(os.path.dirname(__file__), TEMP_DIR, f"{str(uuid.uuid4())}.bcad")

    def _export_svgs(self, folder):
        self._run("svg<>" + folder + "\n")

    def _export_pngs(self, folder):
        self._run("png<>" + folder + "\n")

    def _export_svg_from_list(self, files):
        self._run("svglist<>" + "<>".join(files) + "\n")

    def _export_png_from_list(self, files):
        self._run("pnglist<>" + "<>".join(files) + "\n")

    async def _init_instance(self):
        command = f"java -Djava.awt.headless=false -jar  {resource_path('BikeCAD.jar')}"
        process = await asyncio.create_subprocess_shell(bytes(command, 'utf-8'),
                                                        stdin=subprocess.PIPE,
                                                        stdout=subprocess.PIPE,
                                                        stderr=subprocess.PIPE)
        self._log_info("BikeCAD instance running")
        return process

    def kill(self):
        self._instance.kill()

    def _run(self, command):
        self._log_info(f"Running command {command}...")
        self._instance.stdin.write(bytes(command, 'UTF-8'))
        self._await_termination()

    def _await_termination(self):

        async def get_latest_signal():
            return await self._instance.stdout.readline()

        async def get_error_signal():
            return await self._instance.stderr.readline()

        async def await_termination_timed():
            try:
                await asyncio.wait_for(await_termination(), RENDERER_TIMEOUT)
            except asyncio.exceptions.TimeoutError:
                self._log_error(f"Renderer timed out!")
                raise Exception("Something went wrong: rendering took too long")

        async def await_termination():
            while True:
                self._log_info("Loop...")
                signal = await self._wait_or_pass(get_latest_signal())
                if signal == self._get_expected_success():
                    return
                signal = await self._wait_or_pass(get_error_signal())
                if signal:
                    self._log_error(f"Renderer threw an exception! {signal}")
                    raise Exception(f"Something went wrong: {signal}")

        with self._event_loop_lock:
            self._event_loop.run_until_complete(await_termination_timed())

    def _get_expected_success(self):
        if platform.system() == WINDOWS:
            return b'Done!\r\n'
        else:
            return b'Done!\n'

    async def _wait_or_pass(self, future):
        try:
            signal = await asyncio.wait_for(future, RENDERER_TIMEOUT_GRANULARITY / 2)
            self._log_info(f"{signal}")
            return signal
        except asyncio.exceptions.TimeoutError:
            return None

    def _log_info(self, log_message):
        logging.getLogger(LOGGER_NAME).info(log_message)

    def _log_error(self, log_message):
        logging.getLogger(LOGGER_NAME).error(log_message)
