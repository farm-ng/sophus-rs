use super::{
    pixel::{PixelTrait, ScalarTrait, P},
    view::ImageViewTrait,
};

pub trait InterpolationTrait<'a, const CHANNELS: usize, FieldScalar: ScalarTrait + 'static> {
    fn as_bilinear_field_view(&'a self) -> BilinearView<'a, CHANNELS, FieldScalar, Self>
    where
        Self: ImageViewTrait<'a, CHANNELS, FieldScalar> + Sized;
}

impl<
        'a,
        const CHANNELS: usize,
        FieldScalar: ScalarTrait + 'static,
        T: ImageViewTrait<'a, CHANNELS, FieldScalar>,
    > InterpolationTrait<'a, CHANNELS, FieldScalar> for T
{
    fn as_bilinear_field_view(&'a self) -> BilinearView<'a, CHANNELS, FieldScalar, Self>
    where
        Self: ImageViewTrait<'a, CHANNELS, FieldScalar> + Sized,
    {
        BilinearView {
            view: self,
            _field_scalar: std::marker::PhantomData,
        }
    }
}

pub trait FieldViewTrait<'a, const CHANNELS: usize, FieldScalar: ScalarTrait> {
    fn interp(&self, point: nalgebra::Vector2<f64>) -> P<CHANNELS, FieldScalar>;
}

pub struct BilinearView<'a, const CHANNELS: usize, FieldScalar: ScalarTrait + 'static, View>
where
    View: ImageViewTrait<'a, CHANNELS, FieldScalar>,
{
    view: &'a View,
    _field_scalar: std::marker::PhantomData<FieldScalar>,
}

impl<'a, const CHANNELS: usize, FieldScalar: ScalarTrait + 'static, View>
    FieldViewTrait<'a, CHANNELS, FieldScalar> for BilinearView<'a, CHANNELS, FieldScalar, View>
where
    View: ImageViewTrait<'a, CHANNELS, FieldScalar>,
{
    fn interp(&self, uv: nalgebra::Vector2<f64>) -> P<CHANNELS, FieldScalar> {
        let img = self.view;
        let image_size = img.size();

        let iu = uv[0].trunc() as usize;
        let iv = uv[1].trunc() as usize;
        let frac_u = uv[0].fract();
        let frac_v = uv[1].fract();
        let u = iu;
        let v = iv;

        let u_corner_case = u == image_size.width - 1;
        let v_corner_case = v == image_size.height - 1;

        let val00 = img.pixel(u, v);
        let val01 = if v_corner_case {
            val00
        } else {
            img.pixel(u, v + 1)
        };
        let val10 = if u_corner_case {
            val00
        } else {
            img.pixel(u + 1, v)
        };
        let val11 = if u_corner_case || v_corner_case {
            val00
        } else {
            img.pixel(u + 1, v + 1)
        };

        let val = val00.scale((1.0 - frac_u) * (1.0 - frac_v))
            + val01.scale((1.0 - frac_u) * frac_v)
            + val10.scale(frac_u * (1.0 - frac_v))
            + val11.scale(frac_u * frac_v);
        return val;
    }
}

#[cfg(test)]
mod test {
    use crate::image::{
        interpolation::FieldViewTrait,
        mut_image::MutImage,
        view::ImageViewTrait,
        interpolation::InterpolationTrait, layout::ImageSize,
    };

    #[test]
    fn simple1() {
        let img =
            MutImage::with_size_from_function(ImageSize::new(2, 2), |x, y| [x as f64 + y as f64].into());
        let field = img.as_bilinear_field_view();

        assert_eq!(img.pixel(0, 0), [0.0].into());
        assert_eq!(img.pixel(0, 1), [1.0].into());

        assert_eq!(
            field.interp(nalgebra::Vector2::new(0.0, 0.0)),
            [0.0].into()
        );

        assert_eq!(
            field.interp(nalgebra::Vector2::new(1.0, 1.0)),
            [2.0].into()
        );

        assert_eq!(
            field.interp(nalgebra::Vector2::new(0.5, 0.5)),
            [(0.0 + 1.0 + 1.0 + 2.0) / 4.0].into()
        );
    }

    #[test]
    fn simple2() {
        let img = MutImage::with_size_from_function(ImageSize::new(3, 3), |x, _| [x as f64].into());
        let field = img.as_bilinear_field_view();

        assert_eq!(img.pixel(2, 2), [2.0].into());

        for yi in (0..=20).map(|i| i as f64 * 0.1) {
            for xi in (0..=20).map(|i| i as f64 * 0.1) {
                let v = field.interp(nalgebra::Vector2::new(xi, yi).cast()).x;

                assert!((v - xi).abs() < 1e-10, "xi={}, yi={}, v={}", xi, yi, v);
            }
        }
    }

    #[test]
    fn simple3() {
        let (w, h) = (10, 20);

        let op = |x: f64, y: f64| x + 5.0 * y;

        let img = MutImage::with_size_from_function(ImageSize::new(w, h), |x, y| {
            [op(x as f64, y as f64)].into()
        });
        let field = img.as_bilinear_field_view();

        for yi in (0..=(h - 1) * 10).map(|i| i as f64 * 0.1) {
            for xi in (0..=(w - 1) * 10).map(|i| i as f64 * 0.1) {
                let v = field.interp(nalgebra::Vector2::new(xi, yi).cast()).x;
                let expected = op(xi, yi);
                assert!(
                    (v - expected).abs() < 1e-10,
                    "v:{} != expected: {} at position ({},{})",
                    v,
                    expected,
                    xi,
                    yi
                );
            }
        }
    }
}
