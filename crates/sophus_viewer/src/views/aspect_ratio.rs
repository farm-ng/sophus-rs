use linked_hash_map::LinkedHashMap;
use sophus_image::ImageSize;
use sophus_sensor::DynCamera;

use crate::views::View;

pub(crate) trait HasAspectRatio {
    fn aspect_ratio(&self) -> f32;

    fn view_size(&self) -> ImageSize;

    fn intrinsics(&self) -> &DynCamera<f64, 1>;
}

pub(crate) fn get_median_aspect_ratio(views: &LinkedHashMap<String, View>) -> f32 {
    let mut aspect_ratios = std::vec::Vec::with_capacity(views.len());
    for (_, widget) in views.iter() {
        aspect_ratios.push(widget.aspect_ratio());
    }
    aspect_ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = aspect_ratios.len();
    if n % 2 == 1 {
        aspect_ratios[n / 2]
    } else {
        0.5 * aspect_ratios[n / 2] + 0.5 * aspect_ratios[n / 2 - 1]
    }
}

pub(crate) fn get_max_size(
    views: &LinkedHashMap<String, View>,
    available_width: f32,
    available_height: f32,
) -> (f32, f32) {
    let median_aspect_ratio = get_median_aspect_ratio(views);

    let mut max_width = 0.0;
    let mut max_height = 0.0;

    let n = views.len() as u32;
    for num_cols in 1..=n {
        let num_rows: f32 = ((n as f32) / (num_cols as f32)).ceil();

        let w: f32 = available_width / (num_cols as f32);
        let h = (w / median_aspect_ratio).min(available_height / num_rows);
        let w = median_aspect_ratio * h;
        if w > max_width {
            max_width = w;
            max_height = h;
        }
    }

    (max_width, max_height)
}

pub(crate) fn get_adjusted_view_size(view: &View, max_width: f32, max_height: f32) -> (f32, f32) {
    let aspect_ratio = view.aspect_ratio();
    let view_size = view.view_size();
    let view_width = max_width.min(max_height * aspect_ratio);
    let view_height = max_height.min(max_width / aspect_ratio);
    let view_width = view_width.min(view_size.width as f32);
    let view_height = view_height.min(view_size.height as f32);
    (view_width, view_height)
}
