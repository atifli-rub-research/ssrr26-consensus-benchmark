from benchmark.observations.core.types import Crop
from benchmark.observations.core.interfaces import Cropper
from typing import List, Optional
from PIL import Image as PILImage
from PIL.Image import Image


class GridCropper(Cropper):
    def __init__(
        self,
        cell_size: tuple[int, int] = (256, 256),
        stride: Optional[tuple[int, int]] = None,
        pad_color: tuple[int, int, int] = (0, 0, 0),
    ):
        """
        Grid-based image cropper with non-overlapping fixed-size cells.

        The stride is always equal to the cell size. Boundary crops are padded
        instead of shifted inward, so no additional boundary overlap is created.
        """
        self.cell_size = cell_size
        self.stride = cell_size
        self.pad_color = pad_color

        if stride is not None and stride != cell_size:
            raise ValueError(
                "GridCropper uses non-overlapping tiles only: "
                f"stride must equal cell_size. Got stride={stride}, "
                f"cell_size={cell_size}."
            )

    @staticmethod
    def _generate_positions(
        start: int,
        end: int,
        cell_size: int,
    ) -> List[int]:
        """
        Generate regular non-overlapping grid positions.

        Boundary crops are not shifted inward. If the last crop extends beyond
        the image extent, the missing region is padded during crop generation.
        """
        positions = list(range(start, end, cell_size))

        if not positions:
            positions.append(start)

        return positions

    def generate_cells(
        self,
        image: Image,
        metadata: Optional[dict] = None,
    ) -> List[Crop]:
        metadata = dict(metadata) if metadata is not None else {}

        image_w, image_h = image.size

        crop_w, crop_h = self.cell_size

        region_x1, region_y1, region_x2, region_y2 = (
            0,
            0,
            image_w,
            image_h,
        )

        x_positions = self._generate_positions(
            start=region_x1,
            end=region_x2,
            cell_size=crop_w,
        )

        y_positions = self._generate_positions(
            start=region_y1,
            end=region_y2,
            cell_size=crop_h,
        )

        crops = []

        source_id = metadata.get("source_id", "image")
        tile_index = 0

        for row, y in enumerate(y_positions):
            for col, x in enumerate(x_positions):
                # Logical tile bounds in full-image coordinates.
                # These may extend beyond the actual image at the boundary.
                tile_x1 = x
                tile_y1 = y
                tile_x2 = x + crop_w
                tile_y2 = y + crop_h

                # Valid source region inside the actual image.
                src_x1 = max(region_x1, tile_x1)
                src_y1 = max(region_y1, tile_y1)
                src_x2 = min(region_x2, tile_x2)
                src_y2 = min(region_y2, tile_y2)

                crop_image = PILImage.new(
                    "RGB",
                    (crop_w, crop_h),
                    self.pad_color,
                )

                if src_x2 > src_x1 and src_y2 > src_y1:
                    valid_region = image.crop(
                        (
                            src_x1,
                            src_y1,
                            src_x2,
                            src_y2,
                        )
                    )

                    paste_x = src_x1 - tile_x1
                    paste_y = src_y1 - tile_y1

                    crop_image.paste(
                        valid_region,
                        (
                            paste_x,
                            paste_y,
                        ),
                    )

                cell_id = f"{source_id}_r{row:04d}_c{col:04d}"

                crop_metadata = dict(metadata)
                crop_metadata.update(
                    {
                        "tile_index": tile_index,
                        "cell_id": cell_id,
                        "row": row,
                        "col": col,

                        # Logical tile bounds.
                        "crop_box": (
                            tile_x1,
                            tile_y1,
                            tile_x2,
                            tile_y2,
                        ),
                        "crop_width": crop_w,
                        "crop_height": crop_h,

                        # Existing names used by downstream code.
                        "global_x1": tile_x1,
                        "global_y1": tile_y1,
                        "global_x2": tile_x2,
                        "global_y2": tile_y2,

                        # Alias names.
                        "x_min": tile_x1,
                        "y_min": tile_y1,
                        "x_max": tile_x2,
                        "y_max": tile_y2,

                        # Actual image region copied into the crop.
                        "valid_global_x1": src_x1,
                        "valid_global_y1": src_y1,
                        "valid_global_x2": src_x2,
                        "valid_global_y2": src_y2,

                        # Padding metadata.
                        "padding_left": max(0, region_x1 - tile_x1),
                        "padding_top": max(0, region_y1 - tile_y1),
                        "padding_right": max(0, tile_x2 - region_x2),
                        "padding_bottom": max(0, tile_y2 - region_y2),
                    }
                )

                crops.append(
                    Crop(
                        cell_id=cell_id,
                        image=crop_image,
                        row=row,
                        col=col,
                        metadata=crop_metadata,
                    )
                )

                tile_index += 1

        return crops
