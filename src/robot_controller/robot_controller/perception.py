import os
from datetime import datetime

import cv2
import numpy as np
from cv_bridge import CvBridge


class PerceptionController:

    def __init__(self, logger):
        self.bridge = CvBridge()
        self.logger = logger

        self.latest_bgr = None
        self.latest_depth = None

        self.save_dir = '/opt/erc_ws/erc_images'
        os.makedirs(self.save_dir, exist_ok=True)

        self.template_dir = (
            '/opt/erc_ws/src/robot_controller/'
            'robot_controller/digit_templates'
        )

        self.digit_templates = {}

        for digit in range(1, 6):
            path = os.path.join(
                self.template_dir,
                f'{digit}.png'
            )

            template = cv2.imread(
                path,
                cv2.IMREAD_GRAYSCALE
            )

            if template is None:
                raise RuntimeError(
                    f'Could not load digit template: {path}'
                )

            self.digit_templates[digit] = template

        self.logger.info(
            'Loaded shelf digit templates 1-5'
        )

    # ============================================================
    # CAMERA UPDATES
    # ============================================================

    def update_rgb(self, msg):
        try:
            self.latest_bgr = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )

        except Exception as e:
            self.logger.error(
                f'RGB conversion failed: {e}'
            )

    def update_depth(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='passthrough'
            )

        except Exception as e:
            self.logger.error(
                f'Depth conversion failed: {e}'
            )

    # ============================================================
    # SHELF NUMBER DETECTION
    # ============================================================

    def detect_target_shelf(self, target_column):

        if self.latest_bgr is None:
            return False, None

        frame = self.latest_bgr.copy()

        image_height, image_width = frame.shape[:2]

        # Numbers are in top strip of competition image
        roi_height = int(image_height * 0.22)

        roi = frame[
            0:roi_height,
            :
        ]

        gray = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY
        )

        # Detect black digits
        _, thresh = cv2.threshold(
            gray,
            100,
            255,
            cv2.THRESH_BINARY_INV
        )

        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []

        for contour in contours:

            x, y, w, h = cv2.boundingRect(
                contour
            )

            # Filter tiny noise
            if h < 20:
                continue

            if w < 5:
                continue

            # Filter objects too large to be digits
            if w > 80 or h > 80:
                continue

            candidates.append(
                (x, y, w, h)
            )

        if not candidates:
            return False, None

        # Sort left-to-right
        candidates.sort(
            key=lambda c: c[0]
        )

        recognised = []

        for x, y, w, h in candidates:

            padding = 5

            x1 = max(
                0,
                x - padding
            )

            y1 = max(
                0,
                y - padding
            )

            x2 = min(
                roi.shape[1],
                x + w + padding
            )

            y2 = min(
                roi.shape[0],
                y + h + padding
            )

            digit_crop = thresh[
                y1:y2,
                x1:x2
            ]

            digit_crop = cv2.resize(
                digit_crop,
                (50, 70),
                interpolation=cv2.INTER_NEAREST
            )

            best_digit = None
            best_score = -1.0

            for digit, template in self.digit_templates.items():

                if template.shape != digit_crop.shape:
                    template = cv2.resize(
                        template,
                        (
                            digit_crop.shape[1],
                            digit_crop.shape[0]
                        ),
                        interpolation=cv2.INTER_NEAREST
                    )

                result = cv2.matchTemplate(
                    digit_crop,
                    template,
                    cv2.TM_CCOEFF_NORMED
                )

                score = float(
                    result[0][0]
                )

                if score > best_score:
                    best_score = score
                    best_digit = digit

            # Ignore weak matches
            if best_score < 0.60:
                continue

            cx = int(
                x + w / 2
            )

            cy = int(
                y + h / 2
            )

            recognised.append({
                'digit': best_digit,
                'score': best_score,
                'bbox': (x, y, w, h),
                'center_x': cx,
                'center_y': cy
            })

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f'{best_digit} {best_score:.2f}',
                (x, max(18, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

        if not recognised:
            return False, None

        # Find requested shelf number
        target_matches = [
            item
            for item in recognised
            if item['digit'] == target_column
        ]

        if not target_matches:
            return False, {
                'all_digits': recognised,
                'annotated': frame
            }

        # Pick strongest match if multiple
        target = max(
            target_matches,
            key=lambda x: x['score']
        )

        x, y, w, h = target['bbox']

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3
        )

        cv2.putText(
            frame,
            f'TARGET SHELF {target_column}',
            (x, min(
                image_height - 10,
                y + h + 25
            )),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        return True, {
            'digit': target_column,
            'score': target['score'],
            'bbox': target['bbox'],
            'center_x': target['center_x'],
            'center_y': target['center_y'],
            'image_width': image_width,
            'image_height': image_height,
            'annotated': frame,
            'all_digits': recognised
        }

    # ============================================================
    # BOOK COLOUR DETECTION
    # ============================================================

    def detect_book_colour(self, colour):

        if self.latest_bgr is None:
            return False, None

        colour = colour.lower()

        image_height, image_width = self.latest_bgr.shape[:2]

        # Target shelf column has already been centered.
        # Restrict colour detection to that column.
        column_x1 = int(image_width * 0.30)
        column_x2 = int(image_width * 0.70)

        column_image = self.latest_bgr[
            :,
            column_x1:column_x2
        ]

        hsv = cv2.cvtColor(
            column_image,
            cv2.COLOR_BGR2HSV
        )

        masks = []

        if colour == 'red':

            masks.append(
                cv2.inRange(
                    hsv,
                    np.array([0, 100, 70]),
                    np.array([10, 255, 255])
                )
            )

            masks.append(
                cv2.inRange(
                    hsv,
                    np.array([170, 100, 70]),
                    np.array([180, 255, 255])
                )
            )

        elif colour == 'blue':

            masks.append(
                cv2.inRange(
                    hsv,
                    np.array([95, 80, 50]),
                    np.array([135, 255, 255])
                )
            )

        elif colour == 'green':

            masks.append(
                cv2.inRange(
                    hsv,
                    np.array([35, 70, 50]),
                    np.array([90, 255, 255])
                )
            )

        elif colour == 'yellow':

            masks.append(
                cv2.inRange(
                    hsv,
                    np.array([18, 100, 100]),
                    np.array([38, 255, 255])
                )
            )

        else:
            self.logger.error(
                f'Unsupported book colour: {colour}'
            )

            return False, None

        mask = masks[0]

        for extra_mask in masks[1:]:
            mask = cv2.bitwise_or(
                mask,
                extra_mask
            )

        kernel = np.ones(
            (5, 5),
            np.uint8
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []

        for contour in contours:

            area = cv2.contourArea(
                contour
            )

            if area < 300:
                continue

            x, y, w, h = cv2.boundingRect(
                contour
            )

            if h < 20 or w < 5:
                continue

            candidates.append(
                (area, x, y, w, h, contour)
            )

        if not candidates:
            return False, None

        # We already centered the requested shelf column before
        # approaching. Choose the coloured book closest to that
        # column's image center, NOT simply the largest contour.
        #
        # Candidate coordinates are still relative to column_image.
        crop_center_x = (
            column_image.shape[1] / 2.0
        )

        def candidate_score(candidate):
            area, x, y, w, h, contour = candidate
            cx = x + w / 2.0

            # Main priority: stay on the centered target column.
            horizontal_distance = abs(
                cx - crop_center_x
            )

            # Slight preference for a larger/cleaner detection.
            return (
                horizontal_distance,
                -area
            )

        candidates.sort(
            key=candidate_score
        )

        _, x, y, w, h, selected_contour = candidates[0]

        # Depth must come ONLY from the selected coloured object,
        # not from every object of the same colour in the ROI.
        selected_mask = np.zeros_like(mask)

        cv2.drawContours(
            selected_mask,
            [selected_contour],
            -1,
            255,
            thickness=cv2.FILLED
        )

        # Convert cropped-column coordinate back to full image.
        x += column_x1

        cx = int(
            x + w / 2
        )

        cy = int(
            y + h / 2
        )

        image_height, image_width = (
            self.latest_bgr.shape[:2]
        )

        row = self.estimate_row(
            cy,
            image_height
        )

        depth = self.get_depth_from_mask(
            selected_mask,
            column_x1
        )

        annotated = self.latest_bgr.copy()

        cv2.rectangle(
            annotated,
            (x, y),
            (x + w, y + h),
            (255, 255, 255),
            3
        )

        text = (
            f'{colour} book | row {row}'
        )

        if depth is not None:
            text += (
                f' | depth {depth:.2f}m'
            )

        cv2.putText(
            annotated,
            text,
            (x, max(25, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        return True, {
            'bbox': (x, y, w, h),
            'center_x': cx,
            'center_y': cy,
            'image_width': image_width,
            'image_height': image_height,
            'row': row,
            'depth': depth,
            'annotated': annotated
        }

    # ============================================================
    # HELPERS
    # ============================================================

    def estimate_row(self, cy, image_height):

        relative_y = (
            cy / float(image_height)
        )

        if relative_y < 0.35:
            return 1

        if relative_y < 0.50:
            return 2

        if relative_y < 0.65:
            return 3

        return 4

    def get_depth_from_mask(
        self,
        colour_mask,
        x_offset=0
    ):
        """
        Measure depth only from pixels classified as the book colour.

        This avoids reading the shelf/background through the
        rectangular bounding box around a thin book.
        """

        if self.latest_depth is None:
            return None

        depth_h, depth_w = self.latest_depth.shape[:2]
        mask_h, mask_w = colour_mask.shape[:2]

        # Map cropped colour ROI back onto full depth image.
        x1 = int(x_offset)
        x2 = min(depth_w, x1 + mask_w)

        usable_w = x2 - x1

        if usable_w <= 0:
            return None

        usable_h = min(
            depth_h,
            mask_h
        )

        mask = colour_mask[
            :usable_h,
            :usable_w
        ].copy()

        # Erode slightly to avoid colour/depth boundary pixels.
        kernel = np.ones(
            (3, 3),
            np.uint8
        )

        mask = cv2.erode(
            mask,
            kernel,
            iterations=1
        )

        depth_crop = self.latest_depth[
            :usable_h,
            x1:x2
        ].astype(np.float32)

        valid = depth_crop[
            (mask > 0)
            & np.isfinite(depth_crop)
            & (depth_crop > 0)
        ]

        if valid.size == 0:
            return None

        # Convert mm -> m if necessary.
        if np.median(valid) > 20.0:
            valid = valid / 1000.0

        valid = valid[
            (valid > 0.15)
            & (valid < 5.0)
        ]

        if valid.size == 0:
            return None

        # With only actual coloured pixels, median is appropriate.
        return float(
            np.median(valid)
        )


    def get_depth_in_bbox(self, x, y, w, h):
        """
        Estimate object depth from inside its bounding box.

        Uses an inset region and a low percentile so thin books do
        not get replaced by farther shelf/background depth values.
        """

        if self.latest_depth is None:
            return None

        depth_h, depth_w = self.latest_depth.shape[:2]

        # Inset bbox to avoid shelf edges/background.
        pad_x = max(1, int(w * 0.20))
        pad_y = max(1, int(h * 0.15))

        x1 = max(0, x + pad_x)
        x2 = min(depth_w, x + w - pad_x)

        y1 = max(0, y + pad_y)
        y2 = min(depth_h, y + h - pad_y)

        if x2 <= x1 or y2 <= y1:
            return None

        patch = self.latest_depth[
            y1:y2,
            x1:x2
        ].astype(np.float32)

        valid = patch[
            np.isfinite(patch) & (patch > 0)
        ]

        if valid.size == 0:
            return None

        # Convert millimetres if necessary.
        if np.median(valid) > 20.0:
            valid = valid / 1000.0

        # Reject unrealistic values.
        valid = valid[
            (valid > 0.15) &
            (valid < 5.0)
        ]

        if valid.size == 0:
            return None

        # Near-side percentile works better for a thin object
        # standing in front of a deeper shelf cavity.
        return float(
            np.percentile(valid, 20)
        )

    def get_depth_at_pixel(self, x, y):

        if self.latest_depth is None:
            return None

        height, width = (
            self.latest_depth.shape[:2]
        )

        if (
            x < 0
            or y < 0
            or x >= width
            or y >= height
        ):
            return None

        radius = 3

        x1 = max(
            0,
            x - radius
        )

        x2 = min(
            width,
            x + radius + 1
        )

        y1 = max(
            0,
            y - radius
        )

        y2 = min(
            height,
            y + radius + 1
        )

        patch = self.latest_depth[
            y1:y2,
            x1:x2
        ].astype(
            np.float32
        )

        valid = patch[
            np.isfinite(patch)
        ]

        valid = valid[
            valid > 0
        ]

        if valid.size == 0:
            return None

        depth = float(
            np.median(valid)
        )

        if depth > 20.0:
            depth /= 1000.0

        return depth

    def pixel_to_camera_xyz(self, u, v, depth):
        """
        Convert RGB pixel + depth to XYZ in
        head_front_camera_color_optical_frame.

        Camera intrinsics from /camera_info:
            fx = 337.2096252441406
            fy = 337.2096562385559
            cx = 320.0
            cy = 180.0
        """

        if depth is None or depth <= 0:
            return None

        fx = 337.2096252441406
        fy = 337.2096562385559
        cx = 320.0
        cy = 180.0

        z = float(depth)

        x = (float(u) - cx) * z / fx
        y = (float(v) - cy) * z / fy

        return {
            'x': x,
            'y': y,
            'z': z,
        }

    def horizontal_error(self, result):

        cx = result['center_x']
        width = result['image_width']

        return (
            cx - width / 2.0
        ) / (width / 2.0)

    def save_annotated(
        self,
        image,
        prefix
    ):

        timestamp = datetime.now().strftime(
            '%Y%m%d_%H%M%S_%f'
        )

        filename = os.path.join(
            self.save_dir,
            f'{prefix}_{timestamp}.png'
        )

        cv2.putText(
            image,
            datetime.now().isoformat(),
            (
                10,
                image.shape[0] - 15
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        cv2.imwrite(
            filename,
            image
        )

        self.logger.info(
            f'Saved annotated image: {filename}'
        )

        return filename
