import threading
import unittest

from farmer import Farmer


class ZoomTests(unittest.TestCase):
    def fixture(self, limit=6):
        farmer = Farmer.__new__(Farmer)
        farmer.live = True
        farmer.stop_event = threading.Event()
        farmer.config = {"vision": {"playfield": [0, 0, 1920, 905], "max_zoom_out_ticks": limit}}
        events = []

        class Window:
            def desktop_point(self, point):
                return point

        class GUI:
            def moveTo(self, *point):
                events.append(("move", point))

            def scroll(self, ticks):
                events.append(("scroll", ticks))

        farmer.window = Window()
        farmer.gui = GUI()
        frames = iter(range(1, 10))
        farmer.screenshot = lambda: next(frames)
        farmer.sleep = lambda seconds: None
        farmer.matches_screen = lambda image, state: True
        farmer.zoom_view_changed = lambda before, after: True
        return farmer, events

    def test_plain_wheel_stops_when_view_is_ready(self):
        farmer, events = self.fixture()

        class Planner:
            def view_ready(self, image):
                return image >= 2

        result = farmer.normalize_scouting_view(0, Planner())
        self.assertEqual(result, 2)
        self.assertEqual([e for e in events if e[0] == "scroll"], [("scroll", -1), ("scroll", -1)])

    def test_zoom_limit_and_changed_screen_prevent_further_input(self):
        farmer, events = self.fixture(2)

        class Planner:
            def view_ready(self, image):
                return False

        self.assertEqual(farmer.normalize_scouting_view(0, Planner()), 2)
        self.assertEqual(sum(e[0] == "scroll" for e in events), 2)
        farmer, events = self.fixture()
        farmer.matches_screen = lambda image, state: False
        with self.assertRaisesRegex(RuntimeError, "Scouting screen changed"):
            farmer.normalize_scouting_view(0, Planner())
        self.assertEqual(events, [])

    def test_preview_and_cancellation_do_not_scroll(self):
        farmer, events = self.fixture()
        farmer.live = False
        self.assertEqual(farmer.normalize_scouting_view(0), 0)
        self.assertEqual(events, [])
        farmer.live = True
        farmer.gui.moveTo = lambda *point: farmer.stop_event.set()

        class Planner:
            def view_ready(self, image):
                return False

        with self.assertRaisesRegex(RuntimeError, "Stopped by user"):
            farmer.normalize_scouting_view(0, Planner())
        self.assertEqual(events, [])

    def test_no_wheel_response_stops_after_two_attempts(self):
        farmer, events = self.fixture()
        farmer.zoom_view_changed = lambda before, after: False

        class Planner:
            def view_ready(self, image):
                return False

        self.assertEqual(farmer.normalize_scouting_view(0, Planner()), 2)
        self.assertEqual(sum(e[0] == "scroll" for e in events), 2)

    def test_view_change_ignores_excluded_ui_but_detects_scene_changes(self):
        from PIL import Image, ImageDraw

        from zoom_review import view_changed

        before = Image.new("RGB", (200, 200), "green")
        after = before.copy()
        ImageDraw.Draw(after).rectangle((0, 0, 99, 39), fill="white")
        self.assertFalse(view_changed(before, after, [0, 0, 200, 200], [(0, 0, 110, 50)]))
        ImageDraw.Draw(after).rectangle((70, 70, 150, 150), fill="black")
        self.assertTrue(view_changed(before, after, [0, 0, 200, 200], [(0, 0, 110, 50)]))


if __name__ == "__main__":
    unittest.main()
