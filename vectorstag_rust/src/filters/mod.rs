//! SVG Filter Primitives implementation
//!
//! This module provides implementations for SVG filter primitives.
//! Each filter is in its own submodule for maintainability.

use pyo3::prelude::*;

// Utility modules
pub mod color_utils;
pub mod blur_utils;

// Filter primitive modules
pub mod flood;
pub mod offset;
pub mod blend;
pub mod composite;
pub mod merge;
pub mod color_matrix;
pub mod component_transfer;
pub mod morphology;
pub mod convolve;
pub mod turbulence;
pub mod displacement;
pub mod tile;
pub mod lighting;
pub mod gaussian_blur;
pub mod drop_shadow;
pub mod source_alpha;

// Re-export impl functions for internal use (these are used by svg_renderer)
pub use flood::fe_flood_impl_f32;
pub use offset::fe_offset_impl_f32;
pub use blend::fe_blend_impl_f32;
pub use composite::fe_composite_impl_f32;
pub use merge::fe_merge_impl_f32;
pub use color_matrix::fe_color_matrix_impl_f32;
pub use component_transfer::fe_component_transfer_impl_f32;
pub use morphology::fe_morphology_impl_f32;
pub use convolve::fe_convolve_matrix_impl_f32;
pub use turbulence::fe_turbulence_impl_f32;
pub use displacement::fe_displacement_map_impl_f32;
pub use tile::fe_tile_impl_f32;
pub use lighting::{fe_diffuse_lighting_impl_f32, fe_specular_lighting_impl_f32};
pub use gaussian_blur::fe_gaussian_blur_impl_f32;
pub use drop_shadow::fe_drop_shadow_impl_f32;
pub use source_alpha::get_source_alpha_impl_f32;

/// Register filter module functions
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(flood::fe_flood, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(offset::fe_offset, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(blend::fe_blend, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(composite::fe_composite, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(merge::fe_merge, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(color_matrix::fe_color_matrix, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(component_transfer::fe_component_transfer, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(morphology::fe_morphology, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(convolve::fe_convolve_matrix, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(turbulence::fe_turbulence, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(displacement::fe_displacement_map, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(tile::fe_tile, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(lighting::fe_diffuse_lighting, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(lighting::fe_specular_lighting, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(gaussian_blur::fe_gaussian_blur, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(drop_shadow::fe_drop_shadow, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(source_alpha::get_source_alpha, m)?)?;
    Ok(())
}
