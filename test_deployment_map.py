import unittest

import numpy as np
from PIL import Image, ImageDraw

from deployment_map import DeploymentMap, combine_regions, red_edge_mask, visible_deployment
from vision import Detection, make_plan


class DeploymentMapTests(unittest.TestCase):
    def test_combining_estimates_never_expands_deployable_space(self):
        usable = np.ones((20, 30), bool)
        first = np.zeros_like(usable)
        first[5:15, 5:12] = True
        second = np.zeros_like(usable)
        second[5:15, 10:20] = True
        maps = [
            DeploymentMap(mask, ~mask, np.zeros_like(mask), usable, (0, 0, 30, 20))
            for mask in (first, second)
        ]
        combined = combine_regions(maps, [(6, 10), (18, 10)])
        self.assertFalse(combined.problem)
        self.assertTrue(combined.inside[10, 6])
        self.assertTrue(combined.inside[10, 18])
        self.assertFalse(np.any(combined.outside & ~maps[0].outside))
        self.assertFalse(np.any(combined.outside & ~maps[1].outside))
        self.assertTrue(combine_regions(maps, [(25, 10)]).problem)

    def test_experimental_ridge_separates_thin_red_line_from_solid_red_area(self):
        image = Image.new("RGB", (160, 120), "#406030")
        draw = ImageDraw.Draw(image)
        draw.line((10, 20, 70, 65), fill=(160, 35, 20), width=1)
        draw.rectangle((90, 20, 140, 90), fill=(160, 35, 20))
        edge = red_edge_mask(np.asarray(image), "ridge")
        self.assertTrue(edge[35, 30])
        self.assertFalse(edge[50, 115])
        with self.assertRaises(ValueError):
            red_edge_mask(np.asarray(image), "unknown")

    def test_dense_red_buildings_do_not_discard_a_supported_enclosure(self):
        image = Image.new("RGB", (300, 240), "#406030")
        draw = ImageDraw.Draw(image)
        draw.rectangle((40, 40, 260, 200), outline=(160, 35, 20), width=2)
        draw.rectangle((44, 44, 256, 196), fill=(160, 35, 20))
        draw.rectangle((60, 60, 130, 120), fill="#406030")
        region = visible_deployment(image, [0, 0, 300, 240], collector_centers=[(150, 110)])
        self.assertFalse(region.problem)
        self.assertTrue(region.inside[110, 150])
        self.assertTrue(region.outside[110, 10])

    def test_separate_deployment_islands_are_retained(self):
        image = Image.new("RGB", (400, 240), "#406030")
        draw = ImageDraw.Draw(image)
        draw.rectangle((30, 40, 160, 200), outline=(160, 35, 20), width=2)
        draw.rectangle((230, 60, 350, 180), outline=(160, 35, 20), width=2)
        region = visible_deployment(
            image, [0, 0, 400, 240], collector_centers=[(100, 110), (280, 110)]
        )
        self.assertFalse(region.problem)
        self.assertTrue(region.inside[110, 100])
        self.assertTrue(region.inside[110, 280])
        self.assertTrue(region.outside[110, 200])

    def frame(self):
        image = Image.new("RGB", (300, 240), "#406030")
        ImageDraw.Draw(image).line(
            [(150, -40), (270, 110), (150, 230), (30, 110), (150, -40)], fill=(160, 35, 20), width=2
        )
        return image

    def test_clipped_perimeter_can_separate_visible_regions(self):
        region = visible_deployment(self.frame(), [0, 0, 300, 240], collector_centers=[(150, 110)])
        self.assertFalse(region.problem)
        self.assertTrue(region.inside[110, 150])
        self.assertTrue(region.outside[110, 10])

    def test_open_edges_and_ui_occluded_anchors_are_rejected(self):
        image = self.frame()
        ImageDraw.Draw(image).rectangle((20, 85, 50, 135), fill="#406030")
        region = visible_deployment(image, [0, 0, 300, 240], collector_centers=[(150, 110)])
        self.assertTrue(region.problem)
        region = visible_deployment(
            self.frame(), [0, 0, 300, 240], [(140, 100, 20, 20)], [(150, 110)]
        )
        self.assertTrue(region.problem)
        self.assertFalse(region.usable[110, 150])
        self.assertFalse(region.inside[110, 150])

    def test_missing_anchors_cannot_validate_region(self):
        self.assertTrue(visible_deployment(self.frame(), [0, 0, 300, 240]).problem)

    def test_experimental_wall_references_cannot_authorize_attack(self):
        plan = make_plan(
            [
                Detection("gold_mine", (80, 80, 20, 20), 0.99),
                Detection("wall", (120, 120, 10, 10), 0.99),
            ],
            [(40, 40), (160, 40), (160, 160), (40, 160)],
            {"playfield": [0, 0, 200, 200], "planning_preview_only": True},
            {"wall", "gold_mine"},
        )
        self.assertFalse(any(t.eligible for t in plan.targets))
        self.assertTrue(any("Experimental" in p for p in plan.problems))


if __name__ == "__main__":
    unittest.main()
