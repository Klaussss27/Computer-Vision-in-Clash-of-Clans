import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from vision import (
    Detection,
    Detector,
    VisualPlanner,
    deployment_boundary,
    make_plan,
    route_to_collector,
)


class VisionTests(unittest.TestCase):
    def test_route_cannot_start_on_an_obstacle_or_outside_usable_space(self):
        terrain = np.zeros((12, 12), np.uint8)
        usable = np.ones((12, 12), bool)
        for value in (1, 2):
            terrain[5, 0] = value
            layers, path = route_to_collector(terrain, [(0, 5)], (8, 5), usable)
            self.assertIsNone(layers)
            self.assertEqual(path, [])
        terrain[5, 0] = 0
        usable[5, 0] = False
        self.assertIsNone(route_to_collector(terrain, [(0, 5)], (8, 5), usable)[0])
        usable[5, 0] = True
        self.assertEqual(route_to_collector(terrain, [(0, 5)], (8, 5), usable)[0], 0)

    def test_offline_route_preview_cannot_execute_or_enable_live_settings(self):
        from farmer import Farmer

        image = Image.new("RGB", (400, 300), "#406030")
        ImageDraw.Draw(image).rectangle((80, 60, 320, 240), outline=(160, 35, 20), width=2)
        objects = [
            Detection("gold_mine", (100, 100, 20, 20), 0.99),
            Detection("wall", (250, 100, 8, 8), 0.99),
        ]

        class Stub:
            kinds = {"wall", "gold_mine"}

            def detect(self, image):
                return objects

        planner = VisualPlanner.__new__(VisualPlanner)
        planner.settings = {
            "playfield": [0, 0, 400, 300],
            "boundary_mode": "closed",
            "geometry_validated": False,
            "planning_preview_only": True,
        }
        planner.detector = Stub()
        self.assertFalse(any(t.eligible for t in planner.analyze(image).targets))
        proposal = planner.analyze_offline(image)
        self.assertTrue(proposal.targets[0].eligible)
        self.assertTrue(proposal.proposal_only)
        self.assertFalse(planner.settings["geometry_validated"])
        farmer = Farmer.__new__(Farmer)
        with self.assertRaisesRegex(RuntimeError, "Offline proposal cannot execute"):
            farmer.visual_attack(proposal, planner)

    def test_interface_occluded_collector_is_skipped_without_hiding_visible_target(self):
        hidden = Detection("gold_mine", (75, 100, 30, 40), 0.99)
        visible = Detection("elixir_collector", (210, 130, 25, 40), 0.99)
        wall = Detection("wall", (150, 200, 8, 8), 0.99)
        settings = {"playfield": [0, 0, 300, 300], "ui_exclusions": [(70, 95, 40, 50)]}
        plan = make_plan(
            [hidden, visible, wall],
            [(60, 60), (240, 60), (240, 240), (60, 240)],
            settings,
            {"wall", "gold_mine", "elixir_collector"},
        )
        self.assertFalse(plan.targets[0].eligible)
        self.assertIn("interface", plan.targets[0].reason)
        self.assertTrue(plan.targets[1].eligible)

    def test_clipped_collector_cannot_be_an_attack_target(self):
        plan = make_plan(
            [
                Detection("gold_mine", (100, 0, 40, 50), 0.99),
                Detection("wall", (220, 220, 8, 8), 0.99),
            ],
            [(60, 0), (240, 0), (240, 240), (60, 240)],
            {"playfield": [0, 0, 300, 300]},
            {"wall", "gold_mine"},
        )
        self.assertFalse(plan.targets[0].eligible)
        self.assertIn("clipped", plan.targets[0].reason)

    def test_target_artwork_allows_contact_without_erasing_a_wall(self):
        terrain = np.ones((300, 300), bool)
        terrain[100:180, 75:140] = False
        collector = Detection("gold_mine", (75, 100, 65, 80), 0.99)
        settings = {"playfield": [0, 0, 300, 300], "grid_size": 8, "use_jump_spells": False}
        boundary = [(60, 60), (240, 60), (240, 240), (60, 240)]
        wall = Detection("wall", (220, 220, 8, 8), 0.99)
        plan = make_plan(
            [collector, wall], boundary, settings, {"wall", "gold_mine"}, terrain=terrain
        )
        self.assertTrue(plan.targets[0].eligible)
        # A recognized wall enclosure crossing the target box must survive.
        walls = [
            Detection("wall", box, 0.99)
            for box in [(88, 112, 48, 8), (88, 184, 48, 8), (88, 112, 8, 80), (136, 112, 8, 80)]
        ]
        plan = make_plan(
            [collector] + walls, boundary, settings, {"wall", "gold_mine"}, terrain=terrain
        )
        self.assertFalse(plan.targets[0].eligible)

    def test_unrecognized_non_grass_barrier_does_not_become_an_exposed_route(self):
        terrain = np.ones((300, 300), np.uint8)
        cv2.rectangle(terrain, (96, 96), (200, 200), 0, 8)
        collector = Detection("gold_mine", (140, 140, 20, 20), 0.99)
        boundary = [(60, 60), (240, 60), (240, 240), (60, 240)]
        settings = {"playfield": [0, 0, 300, 300], "grid_size": 8, "max_approach_pixels": 240}
        ignored_wall = Detection("wall", (60, 60, 8, 8), 0.99)
        blocked = make_plan(
            [collector, ignored_wall],
            boundary,
            settings,
            {"wall", "gold_mine"},
            terrain=terrain.astype(bool),
        )
        self.assertFalse(blocked.targets[0].eligible)
        gate = Detection("wall", (88, 128, 24, 48), 0.99)
        recognized = make_plan(
            [collector, gate],
            boundary,
            settings,
            {"wall", "gold_mine"},
            terrain=terrain.astype(bool),
        )
        self.assertEqual(recognized.targets[0].layers, 1)
        self.assertTrue(recognized.targets[0].eligible)

    def test_moved_camera_is_not_evidence_of_destroyed_collector(self):
        planner = VisualPlanner.__new__(VisualPlanner)
        planner.settings = {"playfield": [0, 0, 200, 200]}
        rgb = np.random.default_rng(19).integers(0, 255, (200, 200, 3), dtype=np.uint8)
        image = Image.fromarray(rgb)
        planner.reference_gray = planner.frame_gray(image)
        self.assertTrue(planner.aligned(image))
        self.assertFalse(planner.aligned(Image.fromarray(np.roll(rgb, 16, axis=1))))

    def test_matching_follows_moved_collector_and_finds_multiple_instances(self):
        rng = np.random.default_rng(7)
        sample = rng.integers(0, 255, (16, 20, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder, "gold_mine")
            directory.mkdir()
            Image.fromarray(sample).save(directory / "sample.png")
            detector = Detector(
                {
                    "template_dir": folder,
                    "playfield": [0, 0, 160, 120],
                    "scales": [1],
                    "confidence": 0.99,
                }
            )
            for positions in ([(10, 20), (100, 80)], [(40, 60), (120, 10)]):
                frame = np.zeros((120, 160, 3), np.uint8)
                for x, y in positions:
                    frame[y : y + 16, x : x + 20] = sample
                detected = detector.detect(Image.fromarray(frame))
                self.assertEqual({item.box[:2] for item in detected}, set(positions))

    def test_routes_count_zero_one_and_two_separate_wall_layers(self):
        inside = np.ones((41, 41), dtype=bool)
        for expected in range(3):
            walls = np.zeros((41, 41), np.uint8)
            for offset in (8, 14)[:expected]:
                cv2.rectangle(walls, (offset, offset), (40 - offset, 40 - offset), 1, 1)
            layers, path = route_to_collector(walls, [(0, 20)], (20, 20), inside)
            self.assertEqual(layers, expected)
            self.assertEqual(path[0], (0, 20))
            self.assertEqual(path[-1], (20, 20))

    def test_incomplete_perimeter_does_not_produce_deployment_points(self):
        collector = Detection("gold_mine", (80, 80, 20, 20), 0.99)
        plan = make_plan([collector], [], {"playfield": [0, 0, 200, 200]}, {"wall"})
        self.assertFalse(any(t.eligible for t in plan.targets))
        self.assertTrue(any("perimeter" in message for message in plan.problems))

    def test_boundary_requires_closed_red_line(self):
        image = Image.new("RGB", (300, 240), "#406030")
        draw = ImageDraw.Draw(image)
        draw.rectangle((40, 40, 260, 200), outline=(160, 35, 20), width=2)
        self.assertTrue(deployment_boundary(image, [0, 0, 300, 240]))
        draw.rectangle((35, 90, 45, 130), fill="#406030")
        self.assertEqual(deployment_boundary(image, [0, 0, 300, 240]), [])

    def test_two_layers_are_skipped_by_the_target_planner(self):
        collector = Detection("gold_mine", (190, 190, 20, 20), 0.99)
        detections = [collector]
        for offset in (100, 150):
            far = 400 - offset
            detections.extend(
                [
                    Detection("wall", (offset, offset, far - offset, 8), 0.99),
                    Detection("wall", (offset, far, far - offset, 8), 0.99),
                    Detection("wall", (offset, offset, 8, far - offset), 0.99),
                    Detection("wall", (far, offset, 8, far - offset + 8), 0.99),
                ]
            )
        plan = make_plan(
            detections,
            [(50, 50), (350, 50), (350, 350), (50, 350)],
            {"playfield": [0, 0, 400, 400], "grid_size": 8},
            {"wall", "gold_mine"},
        )
        self.assertEqual(len(plan.targets), 1)
        self.assertFalse(plan.targets[0].eligible)
        self.assertEqual(plan.targets[0].layers, 2)

    def test_target_routes_move_with_the_collectors(self):
        settings = {"playfield": [0, 0, 400, 400], "grid_size": 8}
        boundary = [(50, 50), (350, 50), (350, 350), (50, 350)]
        spawns = []
        for box in [(80, 180, 20, 20), (290, 180, 20, 20)]:
            detections = [
                Detection("gold_mine", box, 0.99),
                Detection("wall", (190, 60, 8, 8), 0.99),
            ]
            plan = make_plan(detections, boundary, settings, {"wall", "gold_mine"})
            target = plan.targets[0]
            self.assertTrue(target.eligible)
            self.assertEqual(target.layers, 0)
            spawns.append(target.spawn)
        self.assertLess(spawns[0][0], 50)
        self.assertGreater(spawns[1][0], 350)

    def test_one_layer_places_jump_on_detected_wall_band(self):
        detections = [Detection("gold_mine", (190, 190, 20, 20), 1)]
        for box in [(120, 120, 160, 8), (120, 280, 160, 8), (120, 120, 8, 168), (280, 120, 8, 168)]:
            detections.append(Detection("wall", box, 1))
        plan = make_plan(
            detections,
            [(50, 50), (350, 50), (350, 350), (50, 350)],
            {"playfield": [0, 0, 400, 400], "grid_size": 8},
            {"wall", "gold_mine"},
        )
        self.assertTrue(plan.targets[0].eligible)
        self.assertEqual(plan.targets[0].layers, 1)
        self.assertIsNotNone(plan.targets[0].jump)

    def test_missing_collector_is_not_automatically_destroyed(self):
        collector = Detection("gold_mine", (10, 10, 40, 40), 1)
        planner = VisualPlanner.__new__(VisualPlanner)
        planner.settings = {"playfield": [0, 0, 200, 200]}
        image = Image.fromarray(
            np.random.default_rng(2).integers(0, 255, (200, 200, 3), dtype=np.uint8)
        )
        planner.reference_gray = planner.frame_gray(image)

        class StubDetector:
            found = []

            def detect(self, image):
                return self.found

        planner.detector = StubDetector()
        self.assertFalse(planner.destroyed(image, collector))
        planner.detector.found = [Detection("rubble", (12, 12, 35, 35), 1)]
        self.assertTrue(planner.destroyed(image, collector))
        planner.detector.found.append(collector)
        self.assertFalse(planner.destroyed(image, collector))


if __name__ == "__main__":
    unittest.main()
