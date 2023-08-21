use super::layout::ImageSizeTrait;
use super::mut_view::MutImageView;
use crate::image::{pixel::{ScalarTrait, P}, layout::{ImageLayout, ImageSize}};

pub struct TileView<'a, const N: usize, T: ScalarTrait + 'static> {
    // Some non-overlapping subview of the original image
    pub view: MutImageView<'a, N, T>,

    // the x,y coordinates of the tile in the original image
    pub tile_origin: (usize, usize),
}

pub struct TiledImageView<'a, const N: usize, T: ScalarTrait + 'static> {
    original_view: MutImageView<'a, N, T>,
    pub tiles: Vec<TileView<'a, N, T>>,
}

impl<'a, const N: usize, T: ScalarTrait> MutImageView<'a, N, T> {
    // TODO: Handle the case where the image size is not a multiple of the tile size
    pub fn into_tiles(self, tile_width: usize, tile_height: usize) -> TiledImageView<'a, N, T> {
        assert!(self.layout.width() % tile_width == 0);
        assert!(self.layout.height() % tile_height == 0);

        let num_tiles = (self.layout.width() / tile_width) * (self.layout.height() / tile_height);
        let mut tiles = Vec::with_capacity(num_tiles);

        let layout = &self.layout;
        let stride = layout.stride;

        for y in 0..(layout.height() / tile_height) {
            let mut offset = stride * y * tile_height;
            for x in 0..(layout.width() / tile_width) {
                let tile_data =
                    &mut self.mut_slice[offset..offset + ((tile_height - 1) * stride + tile_width)];
                let tile_ptr = tile_data.as_mut_ptr();
                let tile_slice =
                    unsafe { std::slice::from_raw_parts_mut(tile_ptr, tile_height * stride) };

                tiles.push(TileView {
                    tile_origin: (x * tile_width, y * tile_height),
                    view: MutImageView {
                        layout: ImageLayout{
                            size : ImageSize {
                                width: tile_width,
                                height: tile_height,
                            },
                            stride: self.layout.stride,
                        },
                        mut_slice: tile_slice,
                    },
                });

                offset += tile_width;
            }
        }

        TiledImageView {
            original_view: self,
            tiles: tiles,
        }
    }

    // For some reason, we have to access the slice directly
    // (instead of via mut_row_slice as MutImageViewTrait::mut_pixel does)
    // in order to avoid duplicate borrow of mutable reference
    pub fn mut_pixel_direct(&mut self, x: usize, y: usize) -> &mut P<N, T> {
        &mut self.mut_slice[y * self.layout.stride + x]
    }
}

impl<'a, const N: usize, T: ScalarTrait> TiledImageView<'a, N, T> {
    pub fn into_image(self) -> MutImageView<'a, N, T> {
        self.original_view
    }
}

impl<'a, const N: usize, T: ScalarTrait + 'static> From<TiledImageView<'a, N, T>>
    for MutImageView<'a, N, T>
{
    fn from(value: TiledImageView<'a, N, T>) -> Self {
        value.original_view
    }
}

#[test]
fn tiled_view_semantics_test() {
    use crate::image::layout::ImageSize;
    use crate::image::mut_image::MutImage;
    use crate::image::mut_view::MutImageViewTrait;
    use crate::image::pixel::P;

    let v = P::<1, u8>::new(0);

    let image_width = 100;
    let image_height = 100;
    let tile_size = 10;
    let mut image =
        MutImage::with_size(ImageSize::from_width_and_height(image_width, image_height));
    let view = image.mut_view();
    // let image = MutImageView {
    //     mut_slice: &mut image_data,
    //     layout: ImageLayout::from_width_and_height(image_width, image_height),
    // };

    let mut tiled_image = view.into_tiles(tile_size, tile_size);
    *tiled_image.tiles[0].view.mut_pixel_direct(0, 0) = v;
    *tiled_image.tiles[1].view.mut_pixel_direct(0, 0) = v;

    // We can't use the original image because into_tiles takes ownership of it
    // Continue working with the complete image after it has been updated by
    // taking ownership back from the TiledImage.
    let mut image = tiled_image.into_image();
    *image.mut_pixel(0, 0) = v
}
