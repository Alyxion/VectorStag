# High-Quality Subpixel Rendering Without Supersampling

## Gap Artifacts - FIXED

**Status**: All major visual gaps fixed
**Root cause**: Stroke polygon construction created problematic seam edge

### Issue History:

#### Session 1: Initial Gap Reduction
- Changed `_stroke_closed_polygon_segmented` from per-quad to unified polygon
- Reduced gaps from 203 to 27 (87% reduction)

#### Session 2: Circle Stroke Gap Fix (Current)
**Problem**: Circle strokes (like copyleft.svg) had massive gaps on the right side
- Rows y=252-255 had entire right half of stroke missing
- Caused by single-polygon construction with seam edge

**Root Cause Analysis**:
1. `render_stroke_closed_polygon` in Rust built stroke as: `left_points + reversed(right_points)`
2. This created a diagonal seam edge from inner[last] to outer[last]
3. At rightmost point of circle, seam edge was nearly horizontal
4. Scanline fill condition `y1 <= screen_y < y2` excluded edge ending exactly at screen_y
5. Result: 4 rows near center had no right-side stroke

**Fix Applied** (`vectorstag_rust/src/lib.rs:618-706`):
```rust
// OLD: Single polygon with problematic seam
let stroke_polygon = left_points + reversed(right_points);
// Seam edge from left[359] to right[359] caused gaps

// NEW: Two separate contours with nonzero winding rule
let outer_edges = build_edges(&right_points, +1);  // Outer = add winding
let inner_edges = build_edges(&left_points, -1);   // Inner = subtract winding
// Fill using winding count: winding != 0 means inside stroke
```

**Key Changes**:
1. Build edges for outer contour (right_points) with +1 winding direction
2. Build edges for inner contour (left_points) with -1 winding direction
3. Use nonzero winding rule: fill when winding count != 0
4. Eliminates seam edge entirely - each contour is self-closing

### Results:
- copyleft.svg: 182 gap pixels → 0 gap pixels
- android.svg: 0 gap pixels (was aspect ratio issue, not gaps)
- book-image.svg: 0 gap pixels
- All 124 tests pass

#### Session 3: Android Stroke Join Fix (Completed)
**Problem**: Android logo body had hard rectangular edges despite `stroke-linejoin="round"`. Later, "lacking filling" artifacts appeared on corners.

**Root Cause 1 (Hard Edges)**:
- `render_stroke_closed_polygon` treated `round` joins as `miter` for the base polygon.
- The sharp miter tip extended beyond the rounded arc, making the corner look sharp.
- **Fix**: Treat `round` joins as `bevel` for the base polygon (cutting the corner short) so the arc determines the silhouette.

**Root Cause 2 (Missing Fills/Gaps)**:
- Incorrect logic for selecting "Inner" vs "Outer" side of the turn.
- `perp` vector points *Right* relative to travel.
- **Right Turn** (`cross > 0`): Outer side is *Left* (`p - perp`). Inner is *Right*.
- **Left Turn** (`cross < 0`): Outer side is *Right* (`p + perp`). Inner is *Left*.
- Previous logic had these swapped or inconsistent, causing:
    - Arcs drawn on the inner side (invisible/useless).
    - Miter connections missing on the inner side (gaps).

**Refactoring**:
- Rewrote `render_stroke_closed_polygon` to use a "Segment + Join" approach.
- **Segments**: Full-width rectangles for each edge.
- **Joins**: Explicit geometry filling the gap on the Outer side (Arc for round, Triangle for bevel, Quad for miter).
- **Inner Side**: Naturally handled by segment overlap (implicit miter).

**Results**:
- `android.svg` renders correctly with rounded corners and no gaps.

**Doubts / Future Improvements**:
- **Tessellation on rounded corners**: The outer rounded corners (body) are approximated with 8 line segments per 90-degree turn. This may show minor faceting compared to analytical curves, though visually sufficient with AA.
- Future optimization: The "Segment + Join" approach creates 2x segments and N joins. For simple miter joins, the original continuous polygon approach was faster. Consider hybrid approach or optimization for simple paths.

### Gap detection tool:
```bash
python scripts/detect_gaps.py --categories lucide --limit 100
python scripts/detect_gaps.py --file samples/svg/copyleft.svg
```

### Visual inspection:
```bash
python scripts/visual_inspect.py --count 5
# Open: file:///tmp/vectorstag_inspect_*/index.html
```

---

## Summary
- **Goal**: Render to large surfaces (1920x1080+) with 8x8 quality antialiasing without creating massive supersampling buffers
- **Algorithm**: Signed-trapezoid-area coverage calculation (AGG/stb_truetype style)
- **API**: New `Canvas` class for direct numpy array rendering with subpixel accuracy
- **Target**: 99.9% visual match with 8x8 supersampling, ~10x memory savings

## Problem Statement

Current approach uses supersampling:
- 1920x1080 with 3x antialias = 5760x3240 buffer = **70MB per render**
- 8x8 supersampling = 15360x8640 = **530MB per render**
- Memory-bound, especially for video/animation pipelines

## Solution: Analytical Coverage Calculation

### Algorithm Overview (from [stb_truetype](https://nothings.org/gamedev/rasterize/) / [AGG](https://agg.sourceforge.net/antigrain.com/doc/index.html))

Instead of sampling subpixels, compute **exact pixel coverage** analytically:

1. **Signed Trapezoid Areas**: Each edge contributes a right-extending trapezoid
2. **Two-Array Method**:
   - Array A: Direct trapezoid areas intersecting pixels
   - Array X: Cumulative "height" for pixels right of edges
3. **Linear Stepping**: Coverage differences between adjacent pixels = `dy/dx`
4. **Signed Winding**: Holes work via opposite winding direction (areas cancel)

**Complexity**: O(p log p) where p = pixels on polygon edges
**Quality**: Equivalent to infinite supersampling (within floating-point precision)

---

## New Rust Module: `canvas.rs`

### Core Data Structures

```rust
/// Analytical edge for coverage calculation
struct AnalyticalEdge {
    x_top: f32,      // X at top of edge
    y_top: f32,      // Top Y
    y_bottom: f32,   // Bottom Y
    dx_per_dy: f32,  // X increment per Y unit (inverse slope)
    direction: i8,   // +1 for downward, -1 for upward (winding)
}

/// Canvas for direct numpy rendering
pub struct Canvas {
    width: usize,
    height: usize,
    // No internal buffer - renders directly to provided array
}
```

### Core Algorithm Functions

```rust
/// Compute per-pixel coverage using signed trapezoid areas
fn compute_coverage_scanline(
    edges: &[AnalyticalEdge],
    y: f32,                    // Current scanline (can be fractional)
    coverage: &mut [f32],      // Output: coverage per pixel [0.0, 1.0]
    x_offset: f32,             // Subpixel X offset
) {
    // For each active edge crossing this scanline:
    // 1. Compute X intersection
    // 2. Add trapezoid area contribution to pixels
    // 3. Track cumulative coverage for pixels to the right
}

/// Fill polygon with analytical AA directly to RGBA array
#[pyfunction]
pub fn canvas_fill_polygon_aa(
    dst: PyReadwriteArray3<u8>,  // Target RGBA array (modified in-place)
    points: Vec<(f32, f32)>,     // Polygon vertices (float coords)
    r: u8, g: u8, b: u8, a: u8,  // Fill color
    fill_rule: u8,               // 0=nonzero, 1=evenodd
) -> PyResult<()>
```

### Canvas Methods (Rust + Python bindings)

#### 1. Polygon Rendering

```rust
// Basic polygon fill with analytical AA
canvas_fill_polygon_aa(dst, points, r, g, b, a, fill_rule)

// Polygon with holes (multi-contour)
canvas_fill_multi_polygon_aa(dst, contours, r, g, b, a, fill_rule)

// Polygon with stroke
canvas_fill_stroke_polygon_aa(
    dst, points,
    fill_color, stroke_color, stroke_width,
    linejoin, miterlimit, fill_rule
)

// Polygon with gradient fill
canvas_fill_polygon_gradient_aa(
    dst, points,
    gradient_type,  // 0=linear, 1=radial
    gradient_params, stops, colors,
    fill_rule
)
```

#### 2. Line/Path Rendering

```rust
// Antialiased line with subpixel endpoints
canvas_stroke_line_aa(
    dst,
    x1: f32, y1: f32, x2: f32, y2: f32,
    r, g, b, a,
    width: f32,
    linecap: u8,  // 0=butt, 1=round, 2=square
)

// Polyline (open path) with joins
canvas_stroke_polyline_aa(
    dst, points,
    r, g, b, a,
    width: f32,
    linecap: u8,
    linejoin: u8,  // 0=miter, 1=round, 2=bevel
    miterlimit: f32,
)

// Dashed line
canvas_stroke_dashed_aa(
    dst, points,
    r, g, b, a, width,
    dash_array: Vec<f32>,
    dash_offset: f32,
    linecap, linejoin, miterlimit,
)
```

#### 3. Rectangle Rendering

```rust
// Axis-aligned rectangle (fast path)
canvas_fill_rect_aa(
    dst,
    x: f32, y: f32, width: f32, height: f32,
    r, g, b, a,
)

// Rounded rectangle
canvas_fill_rounded_rect_aa(
    dst,
    x, y, width, height,
    rx: f32, ry: f32,  // Corner radii
    r, g, b, a,
)

// Rectangle with stroke
canvas_stroke_rect_aa(
    dst,
    x, y, width, height,
    r, g, b, a,
    stroke_width: f32,
)
```

#### 3b. Circle/Ellipse Rendering (Native AA)

```rust
// Filled circle with analytical AA
canvas_fill_circle_aa(
    dst,
    cx: f32, cy: f32, radius: f32,
    r, g, b, a,
)

// Filled ellipse with analytical AA
canvas_fill_ellipse_aa(
    dst,
    cx: f32, cy: f32,
    rx: f32, ry: f32,  // Semi-axes
    r, g, b, a,
)

// Stroked circle
canvas_stroke_circle_aa(
    dst,
    cx, cy, radius,
    r, g, b, a,
    stroke_width: f32,
)

// Stroked ellipse
canvas_stroke_ellipse_aa(
    dst,
    cx, cy, rx, ry,
    r, g, b, a,
    stroke_width: f32,
)
```

#### 4. Image/Mask Operations

```rust
// Blit image with polygon mask (analytical AA on mask edges)
canvas_masked_blit_aa(
    dst: PyReadwriteArray3<u8>,
    src: PyReadonlyArray3<u8>,
    mask_polygon: Vec<(f32, f32)>,  // Mask shape
    dst_x: f32, dst_y: f32,          // Destination (subpixel)
    src_x: i32, src_y: i32,          // Source region
    width: i32, height: i32,
    opacity: f32,
)

// Transformed image blit with mask
canvas_masked_blit_transformed_aa(
    dst, src,
    mask_polygon,
    transform: [f32; 6],  // Affine transform matrix
    opacity: f32,
)
```

#### 5. Gradient Support (Per-Pixel Inline Computation)

Gradients are computed per-pixel during the coverage pass for optimal cache efficiency.
No separate gradient buffer is created - color is interpolated inline as coverage is applied.

```rust
// Linear gradient fill (inline color computation)
canvas_fill_linear_gradient_aa(
    dst, points,
    x1, y1, x2, y2,  // Gradient vector
    stops: Vec<f32>,
    colors: Vec<(u8,u8,u8,u8)>,
    spread_method: u8,  // 0=pad, 1=repeat, 2=reflect
    fill_rule: u8,
)

// Radial gradient fill (inline color computation)
canvas_fill_radial_gradient_aa(
    dst, points,
    cx, cy, r,       // Circle
    fx, fy, fr,      // Focal point
    stops, colors,
    spread_method, fill_rule,
)

// Circle with gradient fill
canvas_fill_circle_gradient_aa(
    dst,
    cx, cy, radius,
    gradient_type, gradient_params,
    stops, colors, spread_method,
)
```

**Implementation**: During scanline processing, for each pixel with non-zero coverage:
1. Compute gradient t-value based on pixel position
2. Interpolate color from stops
3. Apply coverage-weighted alpha compositing

---

## Implementation Details

### Signed Trapezoid Algorithm (per scanline)

```rust
fn process_scanline_coverage(
    edges: &mut [AnalyticalEdge],
    y_scanline: f32,
    row_coverage: &mut [f32],
    width: usize,
) {
    // Clear coverage array
    row_coverage.fill(0.0);

    for edge in edges.iter_mut() {
        if edge.y_top > y_scanline + 1.0 || edge.y_bottom < y_scanline {
            continue;  // Edge not active
        }

        // Compute X intersection at scanline
        let y_clamp_top = y_scanline.max(edge.y_top);
        let y_clamp_bot = (y_scanline + 1.0).min(edge.y_bottom);
        let height = y_clamp_bot - y_clamp_top;

        let x_at_top = edge.x_top + (y_clamp_top - edge.y_top) * edge.dx_per_dy;
        let x_at_bot = edge.x_top + (y_clamp_bot - edge.y_top) * edge.dx_per_dy;

        let x_min = x_at_top.min(x_at_bot);
        let x_max = x_at_top.max(x_at_bot);

        // Distribute coverage across affected pixels
        let px_start = (x_min.floor() as i32).max(0) as usize;
        let px_end = ((x_max.ceil() as i32) + 1).min(width as i32) as usize;

        for px in px_start..px_end {
            let px_left = px as f32;
            let px_right = px_left + 1.0;

            // Compute trapezoid area within this pixel
            let area = compute_trapezoid_area(
                x_at_top, x_at_bot, height,
                px_left, px_right,
            );

            row_coverage[px] += area * edge.direction as f32;
        }
    }

    // Apply winding rule (nonzero or evenodd)
    for cov in row_coverage.iter_mut() {
        *cov = cov.abs().min(1.0);  // nonzero
        // For evenodd: *cov = (cov.abs() % 2.0).min(1.0);
    }
}
```

### Trapezoid Area Calculation

```rust
fn compute_trapezoid_area(
    x_top: f32, x_bot: f32,    // Edge X at top/bottom of scanline
    height: f32,               // Clipped height within scanline
    px_left: f32, px_right: f32,  // Pixel boundaries
) -> f32 {
    // Clip edge to pixel X boundaries
    let left = px_left;
    let right = px_right;

    // Compute intersection of edge with pixel
    let x_min = x_top.min(x_bot);
    let x_max = x_top.max(x_bot);

    if x_max <= left || x_min >= right {
        // Edge entirely outside pixel
        if x_min >= right {
            return height;  // Pixel fully to left of edge = fully covered
        }
        return 0.0;  // Pixel fully to right of edge
    }

    // Partial coverage: compute actual trapezoid area
    // (simplified - full impl handles all edge cases)
    let avg_x = (x_top + x_bot) / 2.0;
    let coverage = ((right - avg_x) / (right - left)).clamp(0.0, 1.0);
    coverage * height
}
```

### Performance Optimizations

1. **Edge sorting**: Sort edges by y_top once, maintain active edge list
2. **Scanline stepping**: Process rows in order, step edge X values incrementally
3. **SIMD**: Use packed f32 operations for coverage accumulation
4. **Fast paths**:
   - Fully covered rows: direct fill without per-pixel calculation
   - Horizontal/vertical edges: simplified coverage
   - Axis-aligned rectangles: 4-edge fast path

---

## Python Canvas Class

```python
# vectorstag/canvas.py

class Canvas:
    """High-performance subpixel rendering canvas."""

    def __init__(self, target: np.ndarray):
        """
        Initialize canvas with target numpy array.

        Args:
            target: RGBA numpy array shape (H, W, 4), dtype=uint8
        """
        self._target = target
        self._width = target.shape[1]
        self._height = target.shape[0]

    def fill_polygon(
        self,
        points: List[Tuple[float, float]],
        color: Tuple[int, int, int, int],
        fill_rule: str = 'nonzero',
    ) -> None:
        """Fill polygon with analytical antialiasing."""
        vectorstag_rust.canvas_fill_polygon_aa(
            self._target, points,
            *color,
            0 if fill_rule == 'nonzero' else 1
        )

    def stroke_line(
        self,
        x1: float, y1: float,
        x2: float, y2: float,
        color: Tuple[int, int, int, int],
        width: float = 1.0,
        cap: str = 'butt',
    ) -> None:
        """Draw antialiased line with subpixel endpoints."""
        cap_map = {'butt': 0, 'round': 1, 'square': 2}
        vectorstag_rust.canvas_stroke_line_aa(
            self._target,
            x1, y1, x2, y2,
            *color, width,
            cap_map[cap]
        )

    def fill_rect(
        self,
        x: float, y: float,
        width: float, height: float,
        color: Tuple[int, int, int, int],
    ) -> None:
        """Fill axis-aligned rectangle."""
        vectorstag_rust.canvas_fill_rect_aa(
            self._target, x, y, width, height, *color
        )

    def fill_rounded_rect(
        self,
        x: float, y: float,
        width: float, height: float,
        rx: float, ry: float,
        color: Tuple[int, int, int, int],
    ) -> None:
        """Fill rounded rectangle."""
        vectorstag_rust.canvas_fill_rounded_rect_aa(
            self._target, x, y, width, height, rx, ry, *color
        )

    def masked_blit(
        self,
        src: np.ndarray,
        mask_polygon: List[Tuple[float, float]],
        dst_x: float, dst_y: float,
        opacity: float = 1.0,
    ) -> None:
        """Blit image with polygon mask."""
        vectorstag_rust.canvas_masked_blit_aa(
            self._target, src,
            mask_polygon,
            dst_x, dst_y,
            0, 0, src.shape[1], src.shape[0],
            opacity
        )

    # ... additional methods for gradients, strokes, etc.
```

---

## Files to Create/Modify

### New Files
1. **`vectorstag_rust/src/canvas.rs`** (~800 lines)
   - Core analytical AA algorithm
   - All canvas rendering functions

2. **`vectorstag/canvas.py`** (~200 lines)
   - Python Canvas class wrapper
   - Convenience methods

3. **`tests/test_canvas.py`** (~400 lines)
   - Comparison tests vs 8x8 supersampling

### Modified Files
1. **`vectorstag_rust/src/lib.rs`**
   - Add `mod canvas;`
   - Export canvas functions via PyO3

2. **`vectorstag/__init__.py`**
   - Export Canvas class

---

## Testing Strategy

### Comparison Tests vs 8x8 Supersampling

```python
def test_polygon_fill_accuracy():
    """Compare analytical AA to 8x8 supersampling."""
    # Create test polygon
    polygon = [(100.3, 50.7), (200.8, 150.2), (50.1, 200.9)]

    # Method 1: 8x8 supersampling (reference)
    big = np.zeros((800, 800, 4), dtype=np.uint8)
    vectorstag_rust.fill_polygon_to_array(big, scale_points(polygon, 8), ...)
    reference = downscale_8x(big)

    # Method 2: Analytical AA
    result = np.zeros((100, 100, 4), dtype=np.uint8)
    canvas = Canvas(result)
    canvas.fill_polygon(polygon, (255, 0, 0, 255))

    # Compare with STRICT tolerances (user-specified)
    diff = np.abs(reference.astype(int) - result.astype(int))
    max_diff = diff.max()
    avg_diff = diff.mean()

    # Strict: max 2 levels per channel, average < 0.1
    assert max_diff <= 2, f"Max pixel diff {max_diff} > 2"
    assert avg_diff < 0.1, f"Avg diff {avg_diff} >= 0.1"
```

### Test Tolerance (Strict)
- **Max per-channel difference**: ≤ 2 (out of 255)
- **Average difference**: < 0.1
- Applies to all primitives: polygons, circles, strokes, gradients, blits

### Test Coverage
- Polygon fills (convex, concave, self-intersecting)
- Multi-contour polygons (with holes)
- Circles and ellipses (filled and stroked)
- Strokes (all join/cap types)
- Rectangles (aligned, rotated, rounded)
- Gradients (linear, radial, with masks)
- Image blitting (with polygon masks)
- Edge cases (subpixel positions, thin features, near-parallel edges, tangent circles)

---

## Performance Targets

| Operation | Current (8x8 SS) | Target (Analytical) | Speedup |
|-----------|------------------|---------------------|---------|
| Polygon fill (1000 vertices) | 15ms | 2ms | 7.5x |
| Rectangle | 0.5ms | 0.05ms | 10x |
| Gradient polygon | 25ms | 5ms | 5x |
| Masked blit | 10ms | 3ms | 3x |
| Memory (1920x1080) | 530MB | 8MB | 66x |

---

## Implementation Order

1. **Phase 1**: Core algorithm in Rust
   - `compute_coverage_scanline()` function
   - `canvas_fill_polygon_aa()` with basic color fill
   - Unit tests for coverage accuracy

2. **Phase 2**: Polygon variations
   - Multi-contour (holes)
   - Even-odd fill rule
   - Gradient fills (per-pixel inline)

3. **Phase 3**: Circles and ellipses (native AA)
   - Analytical circle coverage (distance-based)
   - Ellipse via affine-transformed circle
   - Stroked circles/ellipses

4. **Phase 4**: Strokes and lines
   - Line with caps
   - Polyline with joins
   - Dashed lines

5. **Phase 5**: Rectangles and fast paths
   - Axis-aligned rectangles
   - Rounded rectangles
   - Optimized edge cases

6. **Phase 6**: Image operations
   - Masked blit
   - Transformed blit

7. **Phase 7**: Python wrapper and tests
   - Canvas class
   - Comprehensive test suite (strict tolerance: max 2, avg <0.1)
   - Benchmark comparisons vs 8x8 supersampling

---

## References

- [stb_truetype Rasterizer](https://nothings.org/gamedev/rasterize/) - Signed trapezoid algorithm
- [Anti-Grain Geometry](https://agg.sourceforge.net/antigrain.com/doc/index.html) - Scanline coverage
- [Fast Polygon Rendering (2024)](https://aykevl.nl/2024/02/tinygl-polygon/) - A-buffer optimizations
- [Analytical Anti-Aliasing](https://blog.frost.kiwi/analytical-anti-aliasing/) - Edge-based coverage
