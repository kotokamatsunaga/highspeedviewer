from metavision_core.event_io import EventsIterator
from metavision_sdk_core import OnDemandFrameGenerationAlgorithm, ColorPalette
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIAction, UIKeyEvent
import argparse
import numpy as np
from collections import deque
import cv2
import threading
from metavision_hal import DeviceDiscovery, DeviceConfig

def parse_args():
    parser = argparse.ArgumentParser(description='Metavision Viewer')
    parser.add_argument(
        '-b', '--background-image', dest='background_image', required=True,
        help="Path to the background RGB image")
    parser.add_argument(
        '-o', '--output-video', dest='output_video', required=False, default="replay_output.avi",
        help="Path to save the replay video")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    rgb_image = cv2.imread(args.background_image)
    if rgb_image is None:
        raise ValueError(f"Could not load background image from {args.background_4image}")
    device_config = DeviceConfig()
    device_config.enable_biases_range_check_bypass(True)
    device = DeviceDiscovery.open("", device_config)
    device.get_i_ll_biases().set("bias_hpf",+50)
    mv_iterator = EventsIterator.from_device(device=device, delta_t=1000)
    height, width = mv_iterator.get_size()

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = None
    max_delay_seconds = 10
    frame_buffer = deque(maxlen=10000)
    replay_buffer = None
    display_delay_seconds = 0
    replay_enabled = False
    replay_frame_counter = 0
    video_counter = 0 
    lock = threading.Lock()

    with MTWindow(title="Metavision Events Viewer", width=width, height=height,
                  mode=BaseWindow.RenderMode.BGR) as window:
        def keyboard_cb(key, scancode, action, mods):
            nonlocal video_counter, replay_enabled, display_delay_seconds, replay_frame_counter, replay_buffer, video_writer
            if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                window.set_close_flag()
            elif ord('0') <= key <= ord('9'):
                if action == UIAction.PRESS:
                    digit = key - ord('0')
                    if 1 <= digit <= max_delay_seconds:
                        display_delay_seconds = digit
                        replay_frame_counter = int(display_delay_seconds * 1000)
                        replay_buffer = list(frame_buffer)[-replay_frame_counter:]
                        replay_enabled = True
                        video_filename = f"replay_output_{video_counter}.avi"
                        video_writer = cv2.VideoWriter(video_filename, fourcc, 30, (width, height))
                        print(f"Replay started for {display_delay_seconds} seconds. Saving video to {video_filename}")
                        video_counter += 1  
            elif key == UIKeyEvent.KEY_R and action == UIAction.PRESS:
                replay_enabled = False
                replay_buffer = None
                if video_writer:
                    video_writer.release() 
                    video_writer = None
                print(f"Replay disabled.")
        window.set_keyboard_callback(keyboard_cb)

        event_frame_gen = OnDemandFrameGenerationAlgorithm(width=width, height=height, accumulation_time_us=1000,
                                                           palette=ColorPalette.Dark)

        def generate_and_buffer_frame(ts):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            event_frame_gen.generate(ts, frame)
            frame_buffer.append(frame)

        def display_frame():
            nonlocal replay_enabled, replay_frame_counter, replay_buffer, video_writer
            with lock:
                if replay_enabled and replay_buffer:
                    frame_to_display = replay_buffer.pop(0) 
                    combined_frame = cv2.addWeighted(rgb_image, 0.8, frame_to_display, 0.7, 0)
                else:
                    replay_enabled = False
                    replay_buffer = None
                    replay_frame_counter =0
                    if video_writer:
                        video_writer.release()
                        video_writer = None
                        print("Replay finished.")
                    if frame_buffer:
                        frame_to_display = frame_buffer[-1]
                        combined_frame = cv2.addWeighted(rgb_image, 0.8, frame_to_display, 0.7, 0)
                if replay_enabled and video_writer:
                    video_writer.write(cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB))
                window.show_async(combined_frame)
        last_ts = 0
        current_ts = 0

        for evs in mv_iterator:
            EventLoop.poll_and_dispatch()
            event_frame_gen.process_events(evs)

            if evs.size > 0:
                current_ts = evs['t'][-1] 
                generate_and_buffer_frame(current_ts)
            if current_ts - last_ts >= 33333:
                last_ts = current_ts
                display_frame()

            if window.should_close():
                break
            
        if video_writer:
            video_writer.release()

if __name__ == "__main__":
    main()