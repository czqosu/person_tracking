#!/usr/bin/env python3
"""
Person tracking pipeline using NVIDIA DeepStream.

Pipeline:
    filesrc -> qtdemux -> h264parse -> nvv4l2decoder -> nvstreammux
    -> nvinfer (PeopleNet) -> nvtracker -> nvdsosd
    -> nvvideoconvert -> nvv4l2h264enc -> h264parse -> mp4mux -> filesink
"""

import os
import sys
import time
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

import pyds

PERSON_CLASS_ID = 0
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(SCRIPT_DIR, "..", "config")


class TrackingPipeline:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = os.path.abspath(input_path)
        self.output_path = os.path.abspath(output_path)

        Gst.init(None)
        self.pipeline = None
        self.loop = None
        self.frame_count = 0

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _make(self, factory: str, name: str) -> Gst.Element:
        el = Gst.ElementFactory.make(factory, name)
        if not el:
            raise RuntimeError(f"Failed to create element: {factory} ({name})")
        return el

    def build(self) -> None:
        self.pipeline = Gst.Pipeline.new("person-tracking")

        # Source
        src = self._make("filesrc", "src")
        src.set_property("location", self.input_path)

        qtdemux = self._make("qtdemux", "qtdemux")
        h264parse_in = self._make("h264parse", "h264parse-in")
        decoder = self._make("nvv4l2decoder", "decoder")
        decoder.set_property("enable-max-performance", True)

        # Stream mux
        streammux = self._make("nvstreammux", "streammux")
        streammux.set_property("width", 1920)
        streammux.set_property("height", 1080)
        streammux.set_property("batch-size", 1)
        streammux.set_property("batched-push-timeout", 4000000)
        streammux.set_property("live-source", False)

        # Inference (PeopleNet)
        pgie = self._make("nvinfer", "pgie")
        pgie.set_property(
            "config-file-path", os.path.join(CONFIG_DIR, "pgie_peoplenet.txt")
        )

        # Tracker
        tracker = self._make("nvtracker", "tracker")
        tracker.set_property(
            "ll-lib-file",
            "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
        )
        tracker.set_property(
            "ll-config-file", os.path.join(CONFIG_DIR, "tracker_config.yml")
        )
        tracker.set_property("tracker-width", 640)
        tracker.set_property("tracker-height", 384)
        tracker.set_property("display-tracking-id", True)

        # OSD
        osd = self._make("nvdsosd", "osd")

        # Encoder branch
        nvvidconv = self._make("nvvideoconvert", "nvvidconv")
        encoder = self._make("nvv4l2h264enc", "encoder")
        encoder.set_property("bitrate", 4000000)
        encoder.set_property("preset-level", 1)
        encoder.set_property("insert-sps-pps", True)
        h264parse_out = self._make("h264parse", "h264parse-out")
        mp4mux = self._make("mp4mux", "mp4mux")
        filesink = self._make("filesink", "filesink")
        filesink.set_property("location", self.output_path)
        filesink.set_property("sync", False)

        # Add all to pipeline
        for el in [
            src, qtdemux, h264parse_in, decoder, streammux,
            pgie, tracker, osd, nvvidconv, encoder,
            h264parse_out, mp4mux, filesink,
        ]:
            self.pipeline.add(el)

        # Link static chain: src -> qtdemux (dynamic pad)
        src.link(qtdemux)
        qtdemux.connect("pad-added", self._on_qtdemux_pad, h264parse_in)

        # h264parse_in -> decoder -> streammux
        h264parse_in.link(decoder)
        dec_src = decoder.get_static_pad("src")
        mux_sink = streammux.get_request_pad("sink_0")
        dec_src.link(mux_sink)

        # streammux -> pgie -> tracker -> osd -> nvvidconv -> encoder -> h264parse_out -> mp4mux -> filesink
        for a, b in [
            (streammux, pgie), (pgie, tracker), (tracker, osd),
            (osd, nvvidconv), (nvvidconv, encoder),
            (encoder, h264parse_out), (h264parse_out, mp4mux), (mp4mux, filesink),
        ]:
            if not a.link(b):
                raise RuntimeError(f"Failed to link {a.get_name()} -> {b.get_name()}")

        # Probe on tracker src pad: filter persons, set display text
        tracker.get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER, self._tracker_probe, None
        )

        print(f"[Pipeline] Input:  {self.input_path}")
        print(f"[Pipeline] Output: {self.output_path}")

    # ------------------------------------------------------------------
    # Pad callbacks
    # ------------------------------------------------------------------

    def _on_qtdemux_pad(self, qtdemux, pad, h264parse):
        caps = pad.get_current_caps() or pad.query_caps(None)
        if caps and "video" in caps.to_string():
            sink = h264parse.get_static_pad("sink")
            if not sink.is_linked():
                pad.link(sink)

    # ------------------------------------------------------------------
    # Probe: keep only person class, annotate with track ID
    # ------------------------------------------------------------------

    def _tracker_probe(self, pad, info, _):
        buf = info.get_buffer()
        if not buf:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        l_frame = batch_meta.frame_meta_list
        while l_frame:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            self.frame_count += 1
            to_remove = []
            l_obj = frame_meta.obj_meta_list

            while l_obj:
                try:
                    obj = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                if obj.class_id != PERSON_CLASS_ID:
                    to_remove.append(obj)
                else:
                    # Draw green box
                    obj.rect_params.border_color.set(0.0, 1.0, 0.0, 1.0)
                    obj.rect_params.border_width = 2
                    obj.rect_params.has_bg_color = 0

                    # Label with track ID
                    obj.text_params.display_text = f"ID:{obj.object_id}"
                    obj.text_params.x_offset = int(obj.rect_params.left)
                    obj.text_params.y_offset = max(0, int(obj.rect_params.top) - 18)
                    obj.text_params.font_params.font_name = "Serif"
                    obj.text_params.font_params.font_size = 10
                    obj.text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
                    obj.text_params.set_bg_clr = 1
                    obj.text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.8)

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            for obj in to_remove:
                pyds.nvds_remove_obj_meta_from_frame(frame_meta, obj)

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> None:
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Failed to set pipeline to PLAYING")

        start = time.time()
        self.loop = GLib.MainLoop()
        print("[Pipeline] Processing...")
        try:
            self.loop.run()
        except KeyboardInterrupt:
            pass
        finally:
            self.pipeline.set_state(Gst.State.NULL)
            elapsed = time.time() - start
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            print(f"[Pipeline] Done. {self.frame_count} frames in {elapsed:.1f}s ({fps:.1f} fps)")
            print(f"[Pipeline] Output saved: {self.output_path}")

    def _on_bus_message(self, bus, msg):
        t = msg.type
        if t == Gst.MessageType.EOS:
            print("[Pipeline] EOS")
            self.loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            print(f"[ERROR] {err.message}\n{dbg}")
            self.loop.quit()
