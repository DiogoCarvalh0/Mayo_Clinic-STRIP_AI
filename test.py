import multiprocessing
import os
from typing import Tuple

import cv2
import numpy as np
import openslide
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator
from PIL import Image
from skimage.morphology import disk
from tqdm import tqdm


class Tile:
    """
    Class for information about the tile

    Based on (deep-histopath/deephistopath/wsi/tiles.py):
    https://github.com/CODAIT/deep-histopath/tree/c8baf8d47b6c08c0f6c7b1fb6d5dd6b77e711c33
    """

    def __init__(
        self,
        generator: DeepZoomGenerator,
        tile_level: int,
        col: int,
        row: int,
        basename: str,
    ):
        self.generator = generator
        self.tile_level = tile_level if tile_level >= 0 else self.generator.level_count + tile_level
        self.col = col
        self.row = row
        self.basename = basename

        self.tile = self._get_tile()
        self.tile_np = np.array(self.tile)

        self.tissue_percentage = self._tissue_percentage()
        self.optical_density_percentage = self._optical_density_percentage()
        self.s_and_v_factor = self._hsv_saturation_and_value_factor()
        self.tissue_quantity_factor = self._tissue_quantity_factor()
        self.score = self._score_tile()

    def _get_tile(self) -> Image:
        """
        Generates the Image based on the DeepZoomGenerator, the tile level and the col and row

        Returns:
            Image: Respective Image.
        """
        return self.generator.get_tile(self.tile_level, (self.col, self.row))

    def _tissue_percentage(self) -> float:
        """
        Uses edge detection (Canny) to see where the tissue is, as the
        tissue should be full of edges while the background should not.
        The algorithm follows the following steps:
            . Convert image from RGB to gray
            . Canny (for edge detection)
            . Closing (dilatation -> erosion) to combine detections
            . Dilatation
            . Percentage of binary image that represent the tissue

        Returns:
            float: Percentage of the image that is tissue (from 0 to 1).
        """
        tile = np.uint8(cv2.cvtColor(self.tile_np, cv2.COLOR_BGR2GRAY))
        tile = cv2.Canny(tile, 255 / 3, 255)
        tile = cv2.morphologyEx(tile, cv2.MORPH_CLOSE, kernel=disk(10))  # closing
        tile = cv2.dilate(tile, kernel=disk(10))  # dilation
        tile = tile / 255  # normalize values between 0 and 1

        return tile.mean()

    def get_optical_density_tile(self, tile: np.array) -> np.array:
        """
        Convert a tile to optical density values.

        Args:
        tile (np.array): A 3D NumPy array of shape (tile_size, tile_size, channels).

        Returns:
            np.array: A 3D NumPy array of shape (tile_size, tile_size, channels) representing optical density values.
        """
        tile = tile.astype(np.float64)
        # od = -np.log10(tile/255 + 1e-8)
        od = -np.log((tile + 1) / 240)

        return od

    def _optical_density_percentage(self) -> float:
        """
        It is based on the optical density of the image.
        The algorithm follows the following steps:
            . Calculate the optical density values of the tile
            . Binarize the image with the threshold of 0.15
            . Closing (dilatation -> erosion) to combine detections
            . Dilatation
            . Percentage of binary image that represent the tissue

        Returns:
            float: Percentage of the image that is tissue based on optical density (from 0 to 1).
        """
        tile = self.get_optical_density_tile(self.tile_np)
        beta = 0.15
        tile = np.uint8(np.min(tile, axis=2) >= beta)
        tile = cv2.morphologyEx(tile, cv2.MORPH_CLOSE, kernel=disk(5))  # closing
        tile = cv2.dilate(tile, kernel=disk(5))  # dilation

        return tile.mean()

    def _hsv_saturation_and_value_factor(self) -> float:
        """
        Function to reduce scores of tiles with narrow HSV saturations and values since
        saturation and value standard deviations should be relatively broad if the tile
        contains significant tissue.

        Returns:
            float: Saturation and value factor, where 1 is no effect and less than 1 means
            the standard deviations of saturation and value are relatively small.
        """
        hsv = cv2.cvtColor(self.tile_np, cv2.COLOR_RGB2HSV)
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        s_std = np.std(s)
        v_std = np.std(v)

        if s_std < 0.05 and v_std < 0.05:
            factor = 0.4
        elif s_std < 0.05 or v_std < 0.05:
            factor = 0.7
        else:
            factor = 1

        return factor**2

    def _tissue_quantity_factor(self) -> float:
        """
        Obtain a scoring factor based on the quantity of tissue in a tile.

        Returns:
            float: Scoring factor based on the tile tissue quantity.
        """

        high_tissue_amount = 0.8
        low_tissue_amount = 0.2

        if self.tissue_percentage >= high_tissue_amount:
            return 1.0

        if (self.tissue_percentage >= low_tissue_amount) and (self.tissue_percentage < high_tissue_amount):
            return 0.2

        if self.tissue_percentage < low_tissue_amount:
            return 0.1

        return 0.0

    def _score_tile(self) -> float:
        """
        Score tile based on tissue percentage, color factor, saturation/value factor, and tissue quantity factor.

        Formula base from:
        https://developer.ibm.com/articles/an-automatic-method-to-identify-tissues-from-big-whole-slide-images-pt4/

        Returns:
            float: Score.
        """
        combined_factor = self.s_and_v_factor * self.tissue_quantity_factor
        score = ((self.tissue_percentage * 100) ** 2) * np.log(1 + combined_factor) / 1000.0

        # scale score to between 0 and 1 (min max scale -> min=0, max=100^2 * log(2)/1000 approx 7)
        score = score / 7.0

        return score

    def save_tile(self, target_dir: str) -> None:
        """
        Saves the image to the targer directory with the name: "{basename}_{col}-{row}.png".

        Args:
            target_dir (str): Directory to save the tile.
        """
        self.tile.save(os.path.join(target_dir, f"{self.basename}_{self.col}-{self.row}.png"))


class WSI:
    """
    Class for WSI

    Based on (deep-histopath/deephistopath/wsi/tiles.py):
    https://github.com/CODAIT/deep-histopath/tree/c8baf8d47b6c08c0f6c7b1fb6d5dd6b77e711c33
    """

    def __init__(
        self,
        path: str,
        tile_size: int = 256,
        overlap: int = 0,
        tile_level: int = -1,
        tissue_threshold: float = 0.90,
    ):
        self.path = path
        self.tile_size = tile_size
        self.overlap = overlap
        self.tile_level = tile_level
        self.tissue_threshold = tissue_threshold

        self.slide = open_slide(self.path)
        self.generator = DeepZoomGenerator(
            self.slide,
            tile_size=self.tile_size,
            overlap=self.overlap,
            limit_bounds=True,
        )
        self.tiles = self._create_tiles()

    def get_top_N_tiles(self, N: int) -> Tuple[Tile, ...]:
        """
        Retrieve the top N tiles ranked by score.

        Returns:
            Tuple[Tile, ...]: List of the tiles ranked by score.
        """

        if N >= len(self.tiles):
            return self.tiles

        sorted_list = sorted(self.tiles, key=lambda t: t.score, reverse=True)

        return sorted_list[:N]

    def _keep_tile(self, tile: Tile, tile_size: int, tissue_threshold: float = 0.9) -> bool:
        """
        Determine if a tile should be kept.

        This filters out tiles based on size, variation and a tissue
        percentag threshold, using a custom algorithm. If a tile has
        height &width equal to (tile_size, tile_size), and contains
        greaterthan or equal to the given percentage, then it will
        be kept; otherwise, it will be filtered out.

        Check 1:
        Makes sure the standard deviation is greater than at least 2.5.

        Check 2:
        Makes sure tissue percentage is greater than tissue_threshold (from 0 to 1)

        Check 3:
        Makes sure optical_density tissue percentage is greater than tissue_threshold (from 0 to 1)

        Args:
            tile (Tile): Tile object.
            tile_size (int): Desired tile size.
            tissue_threshold (float, optional): Percentage of the image that has to be tissue. Defaults to 0.9.

        Returns:
            bool: A Boolean indicating whether or not a tile should be kept for future usage.
        """
        if tile.tile_np.shape[0:2] != (tile_size, tile_size):
            return False

        check1 = np.std(tile.tile_np) > 2.5
        check2 = tile.tissue_percentage >= tissue_threshold
        check3 = tile.optical_density_percentage >= tissue_threshold

        return check1 and check2 and check3

    def _create_tiles(self) -> Tuple[Tile, ...]:
        """
        Creates all the tiles from the WSI that pass the "_keep_tile" test.

        Returns:
            Tuple[Tile, ...]: Tuple with all the tiles.
        """
        tiles = []

        cols, rows = self.generator.level_tiles[self.tile_level]

        for row in range(rows):
            for col in range(cols):
                tile = Tile(
                    generator=self.generator,
                    tile_level=self.tile_level,
                    col=col,
                    row=row,
                    basename=os.path.basename(self.path).split(".")[0],
                )

                if self._keep_tile(tile, self.tile_size, self.tissue_threshold):
                    tiles.append(tile)

        return tiles
