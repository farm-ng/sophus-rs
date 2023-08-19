use super::{
    layout::ImageSize,
    pixel::{PixelTrait, ScalarTrait, P},
    view::ImageViewTrait,
};

// A continuous view over a discrete image through interpolation.
pub trait FieldViewTrait<'a, const CHANNELS: usize, FieldScalar: ScalarTrait> {
    fn interp(&self, point: nalgebra::Vector2<f64>) -> P<CHANNELS, FieldScalar>;
}

//////////////////////////////////////////////////////////////////////////////
/// Bilinear interpolation

pub trait BilinearTrait<'a, const CHANNELS: usize, FieldScalar: ScalarTrait + 'static> {
    fn bilinear_view(&'a self) -> BilinearView<'a, CHANNELS, FieldScalar, Self>
    where
        Self: ImageViewTrait<'a, CHANNELS, FieldScalar> + Sized,
    {
        BilinearView {
            view: self,
            _field_scalar: std::marker::PhantomData,
        }
    }
}

impl<
        'a,
        const CHANNELS: usize,
        FieldScalar: ScalarTrait + 'static,
        T: ImageViewTrait<'a, CHANNELS, FieldScalar>,
    > BilinearTrait<'a, CHANNELS, FieldScalar> for T
{
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
        let ImageSize { width, height } = img.size();

        let uv = boundary_neumann(uv, width, height);

        let iu = uv[0].trunc() as usize;
        let iv = uv[1].trunc() as usize;
        let frac_u = uv[0].fract();
        let frac_v = uv[1].fract();
        let u = iu;
        let v = iv;

        let u_corner_case = u == width - 1;
        let v_corner_case = v == height - 1;

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

//////////////////////////////////////////////////////////////////////////////
/// Nearest Neighbour interpolation

pub trait NearestNeighbourTrait<'a, const CHANNELS: usize, FieldScalar: ScalarTrait + 'static> {
    fn nearest_neighbour_view(&'a self) -> NearestNeighbourView<'a, CHANNELS, FieldScalar, Self>
    where
        Self: ImageViewTrait<'a, CHANNELS, FieldScalar> + Sized,
    {
        NearestNeighbourView {
            view: self,
            _field_scalar: std::marker::PhantomData,
        }
    }
}

impl<
        'a,
        const CHANNELS: usize,
        FieldScalar: ScalarTrait + 'static,
        T: ImageViewTrait<'a, CHANNELS, FieldScalar>,
    > NearestNeighbourTrait<'a, CHANNELS, FieldScalar> for T
{
    // Nothing needed.
}

pub struct NearestNeighbourView<'a, const CHANNELS: usize, FieldScalar: ScalarTrait + 'static, View>
where
    View: ImageViewTrait<'a, CHANNELS, FieldScalar>,
{
    view: &'a View,
    _field_scalar: std::marker::PhantomData<FieldScalar>,
}

// Neumann boundary condition is the fancy way of saying "clamp to the image dimensions".
fn boundary_neumann(
    uv: nalgebra::Vector2<f64>,
    width: usize,
    height: usize,
) -> nalgebra::Vector2<f64> {
    uv.zip_map(
        &nalgebra::Vector2::new((width - 1) as f64, (height - 1) as f64),
        f64::min,
    )
    .zip_map(&nalgebra::Vector2::zeros(), f64::max)
}

impl<'a, const CHANNELS: usize, FieldScalar: ScalarTrait + 'static, View>
    FieldViewTrait<'a, CHANNELS, FieldScalar>
    for NearestNeighbourView<'a, CHANNELS, FieldScalar, View>
where
    View: ImageViewTrait<'a, CHANNELS, FieldScalar>,
{
    fn interp(&self, uv: nalgebra::Vector2<f64>) -> P<CHANNELS, FieldScalar> {
        let dim = self.view.size();
        let uv = boundary_neumann(uv, dim.width, dim.height);
        let uv = uv.map(|x| x.round() as usize);
        self.view.pixel(uv.x, uv.y)
    }
}

//////////////////////////////////////////////////////////////////////////////
/// Bicubic interpolation
/// Based on https://www.shadertoy.com/view/4df3Dn (Simon Green no less)

fn w0(a: f64) -> f64 {
    (1.0 / 6.0) * (a * (a * (-a + 3.0) - 3.0) + 1.0)
}

fn w1(a: f64) -> f64 {
    (1.0 / 6.0) * (a * a * (3.0 * a - 6.0) + 4.0)
}

fn w2(a: f64) -> f64 {
    (1.0 / 6.0) * (a * (a * (-3.0 * a + 3.0) + 3.0) + 1.0)
}

fn w3(a: f64) -> f64 {
    (1.0 / 6.0) * (a * a * a)
}

fn g0(a: f64) -> f64 {
    w0(a) + w1(a)
}

fn g1(a: f64) -> f64 {
    w2(a) + w3(a)
}

fn h0(a: f64) -> f64 {
    -1.0 + w1(a) / (w0(a) + w1(a))
}

fn h1(a: f64) -> f64 {
    1.0 + w3(a) / (w2(a) + w3(a))
}

pub trait BicubicTrait<'a, const CHANNELS: usize, FieldScalar: ScalarTrait + 'static> {
    fn bicubic_view(&'a self) -> BicubicView<'a, CHANNELS, FieldScalar, Self>
    where
        Self: ImageViewTrait<'a, CHANNELS, FieldScalar> + Sized,
    {
        BicubicView {
            view: self,
            _field_scalar: std::marker::PhantomData,
        }
    }
}

impl<
        'a,
        const CHANNELS: usize,
        FieldScalar: ScalarTrait + 'static,
        T: ImageViewTrait<'a, CHANNELS, FieldScalar>,
    > BicubicTrait<'a, CHANNELS, FieldScalar> for T
{
    // Nothing needed.
}

pub struct BicubicView<'a, const CHANNELS: usize, FieldScalar: ScalarTrait + 'static, View>
where
    View: ImageViewTrait<'a, CHANNELS, FieldScalar>,
{
    view: &'a View,
    _field_scalar: std::marker::PhantomData<FieldScalar>,
}

impl<'a, const CHANNELS: usize, FieldScalar: ScalarTrait + From<f64> + 'static, View>
    FieldViewTrait<'a, CHANNELS, FieldScalar> for BicubicView<'a, CHANNELS, FieldScalar, View>
where
    View: ImageViewTrait<'a, CHANNELS, FieldScalar>,
{
    fn interp(&self, uv: nalgebra::Vector2<f64>) -> P<CHANNELS, FieldScalar> {
        type Vec2 = nalgebra::Vector2<f64>;

        let bilin = self.view.bilinear_view();

        // let uv = uv + Vec2::new(0.5, 0.5);
        let iuv = uv.map(|x| x.floor());
        let fuv = uv.map(|x| x.fract());

        let gx = [g0(fuv.x), g1(fuv.x)];
        let gy = [g0(fuv.y), g1(fuv.y)];

        let h0x = h0(fuv.x);
        let h1x = h1(fuv.x);
        let h0y = h0(fuv.y);
        let h1y = h1(fuv.y);

        let ps = [
            Vec2::new(iuv.x + h0x, iuv.y + h0y),
            Vec2::new(iuv.x + h1x, iuv.y + h0y),
            Vec2::new(iuv.x + h0x, iuv.y + h1y),
            Vec2::new(iuv.x + h1x, iuv.y + h1y),
        ];

        let ls = ps.map(|p| bilin.interp(p).map(|x| x.to_f64().unwrap()));

        let res: P<CHANNELS, f64> =
            gy[0] * (gx[0] * ls[0] + gx[1] * ls[1]) + gy[1] * (gx[0] * ls[2] + gx[1] * ls[3]);
        res.map(|x| x.into())
    }
}

//////////////////////////////////////////////////////////////////////////////
/// Tests

#[cfg(test)]
mod test {
    use crate::image::{
        interpolation::*, layout::ImageSize, mut_image::MutImage, view::ImageViewTrait,
    };

    #[test]
    fn bilinear_basic() {
        let img = MutImage::with_size_from_function(ImageSize::new(2, 2), |x, y| {
            [x as f64 + y as f64].into()
        });
        let field = img.bilinear_view();

        assert_eq!(img.pixel(0, 0), [0.0].into());
        assert_eq!(img.pixel(0, 1), [1.0].into());

        assert_eq!(field.interp(nalgebra::Vector2::new(0.0, 0.0)), [0.0].into());

        assert_eq!(field.interp(nalgebra::Vector2::new(1.0, 1.0)), [2.0].into());

        assert_eq!(
            field.interp(nalgebra::Vector2::new(0.5, 0.5)),
            [(0.0 + 1.0 + 1.0 + 2.0) / 4.0].into()
        );
    }

    #[test]
    fn bilinear_x() {
        let img = MutImage::with_size_from_function(ImageSize::new(3, 3), |x, _| [x as f64].into());
        let field = img.bilinear_view();

        assert_eq!(img.pixel(2, 2), [2.0].into());

        for yi in (0..=20).map(|i| i as f64 * 0.1) {
            for xi in (0..=20).map(|i| i as f64 * 0.1) {
                let v = field.interp(nalgebra::Vector2::new(xi, yi).cast()).x;

                assert!((v - xi).abs() < 1e-10, "xi={}, yi={}, v={}", xi, yi, v);
            }
        }
    }

    fn check_field_against_op<'a, F : Fn(f64, f64) -> f64>(
        width: usize,
        height: usize,
        field: &impl FieldViewTrait<'a, 1, f64>,
        op: F,
    ) {
        // super-sample by 10 to very interpolation against ground truth op
        for yi in (0..=(height - 1) * 10).map(|i| i as f64 * 0.1) {
            for xi in (0..=(width - 1) * 10).map(|i| i as f64 * 0.1) {
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

    #[test]
    fn bilinear_op() {
        let (w, h) = (10, 20);
        let op = |x: f64, y: f64| x + 5.0 * y;

        let img = MutImage::with_size_from_function(ImageSize::new(w, h), |x, y| {
            [op(x as f64, y as f64)].into()
        });
        check_field_against_op(10, 20, &img.bilinear_view(), op);
    }

    #[test]
    fn bicubic_consistent() {
        let (w, h) = (10, 20);
        let op = |x: f64, y: f64| x + 5.0 * y;

        let img = MutImage::with_size_from_function(ImageSize::new(w, h), |x, y| {
            [op(x as f64, y as f64)].into()
        });
        check_field_against_op(10, 20, &img.bicubic_view(), op);
    }
}
