import cv2
import numpy as np

BEAM_COUNTS = (256, 512)
GAIN_BYTES = 4

FOV_DEG = 130.0

SUPPRESS_WIDTH = 8
SUPPRESS_PAD = 2
SUPPRESS_BANDS = 8
SUPPRESS_THRESHOLD = 2.0

def gain_column_count(width):
	for beams in BEAM_COUNTS:
		if width == beams:
			return 0
		if width == beams + GAIN_BYTES:
			return GAIN_BYTES
		if width == beams + GAIN_BYTES // 2:
			return GAIN_BYTES // 2

	return 0


def row_gains(img, columns):
	block = np.ascontiguousarray(img[:, :columns])

	return block.view(np.uint32).reshape(-1).astype(np.float64)


def strip_gains(img, normalise=False):
	columns = gain_column_count(img.shape[1])
	if columns == 0:
		return img

	beams = img[:, columns:]
	if not normalise:
		return beams

	gains = row_gains(img, columns)
	valid = gains > 0.0
	if not valid.any():
		return beams

	scale = np.ones_like(gains)
	scale[valid] = np.sqrt(np.median(gains[valid]) / gains[valid])

	limits = np.iinfo(beams.dtype)

	return np.clip(beams.astype(np.float32) * scale[:, None], limits.min, limits.max).astype(beams.dtype)


def suppress_center_beam(img, width=SUPPRESS_WIDTH, pad=SUPPRESS_PAD, bands=SUPPRESS_BANDS, threshold=SUPPRESS_THRESHOLD):
	if width <= 0:
		return img

	f = img.astype(np.float32)
	h, w = f.shape
	c = w // 2
	half = int(width)

	lo = max(c - half, 0)
	hi = min(c + half + 1, w)
	rlo = max(c - half * (1 + int(pad)), 0)
	rhi = min(c + half * (1 + int(pad)) + 1, w)

	win = f[:, rlo:rhi]
	il = lo - rlo
	ir = hi - rlo

	nbands = max(1, min(int(bands), h // 8))
	edges = np.linspace(0, h, nbands + 1).astype(int)
	prof = np.stack([np.median(win[a:b], axis=0) for a, b in zip(edges[:-1], edges[1:])])

	ref = np.median(np.concatenate([prof[:, :il], prof[:, ir:]], axis=1), axis=1, keepdims=True)
	excess = prof - ref
	excess[:, :il] = 0.0
	excess[:, ir:] = 0.0
	excess[excess < threshold] = 0.0

	if nbands > 1:
		excess = cv2.resize(excess, (excess.shape[1], h), interpolation=cv2.INTER_LINEAR)
	else:
		excess = np.repeat(excess, h, axis=0)

	out = f.copy()
	out[:, rlo:rhi] = win - excess

	if np.issubdtype(img.dtype, np.integer):
		limits = np.iinfo(img.dtype)
		return np.clip(out, limits.min, limits.max).astype(img.dtype)

	return np.maximum(out, 0.0)


def beam_bearings(count, half_fov, spacing='sine'):
	u = 2.0 * np.arange(count, dtype=np.float64) / max(1, count - 1) - 1.0

	if spacing == 'angle':
		return u * half_fov

	return np.arcsin(np.clip(u * np.sin(half_fov), -1.0, 1.0))


def bearing_to_column(bearing, columns, half_fov):
	return ((np.sin(bearing) / np.sin(half_fov)) + 1.0) / 2.0 * (columns - 1)


def plane_det(ground):
	return ground[0, 0] * ground[1, 1] - ground[0, 1] * ground[1, 0]


def plane_inverse(forward, cross, ground, det):
	a = (ground[1, 1] * forward - ground[0, 1] * cross) / det
	b = (ground[0, 0] * cross - ground[1, 0] * forward) / det

	return a, b


def polar_remap(rows, columns, canvas_h, canvas_w, half_fov, scale, bin_m, ground=None, edge_cut=0):
	if ground is None:
		ground = np.eye(2)

	det = plane_det(ground)

	cx = canvas_w // 2
	cy = canvas_h - 1

	ys, xs = np.indices((canvas_h, canvas_w))
	cross = (xs - cx) * scale
	forward = (cy - ys) * scale

	a, b = plane_inverse(forward, cross, ground, det)
	r = np.sqrt(a * a + b * b)
	bearing = np.arctan2(b, a)

	col = bearing_to_column(bearing, columns, half_fov)
	row = r / bin_m - 0.5

	valid = (
		(col >= edge_cut) & (col <= (columns - 1) - edge_cut) &
		(row >= 0) & (row <= rows - 1) &
		(np.abs(bearing) <= half_fov)
	)

	map_x = np.where(valid, col, 0).astype(np.float32)
	map_y = np.where(valid, row, 0).astype(np.float32)

	return map_x, map_y, valid
