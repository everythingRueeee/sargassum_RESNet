from osgeo import gdal, osr
import os
import numpy as np

os.environ['PROJ_LIB'] = r'E:\anaconda3\envs\sargassum_dl\Library\share\proj'


def create_tile(multi_raster, mask_raster, index_raster, output_dir, tile_size=256):
    ds = gdal.Open(multi_raster)
    mask_ds = gdal.Open(mask_raster)
    index_ds = gdal.Open(index_raster)

    if ds is None or mask_ds is None or index_ds is None:
        raise RuntimeError("无法打开输入影像、mask影像或索引影像")

    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    bands = ds.RasterCount

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    x_tiles = int(np.ceil(xsize / tile_size))
    y_tiles = int(np.ceil(ysize / tile_size))

    geotransform = ds.GetGeoTransform()
    projection = ds.GetProjectionRef()

    for i in range(x_tiles):
        for j in range(y_tiles):
            x_offset = i * tile_size
            y_offset = j * tile_size
            read_width = min(tile_size, xsize - x_offset)
            read_height = min(tile_size, ysize - y_offset)

            if read_width < tile_size:
                x_offset = max(0, x_offset - (tile_size - read_width))
                read_width = tile_size
            if read_height < tile_size:
                y_offset = max(0, y_offset - (tile_size - read_height))
                read_height = tile_size

            data = ds.ReadAsArray(x_offset, y_offset, read_width, read_height)
            mask_data = mask_ds.ReadAsArray(x_offset, y_offset, read_width, read_height)
            index_data = index_ds.ReadAsArray(x_offset, y_offset, read_width, read_height)

            if np.any(mask_data > 0):
                img_opt_dir = os.path.join(output_dir, 'image')
                if not os.path.exists(img_opt_dir):
                    os.makedirs(img_opt_dir)
                output_file = os.path.join(img_opt_dir, f"{i}_{j}.tif")

                driver = gdal.GetDriverByName("GTiff")
                out_ds = driver.Create(output_file, read_width, read_height, bands, ds.GetRasterBand(1).DataType)

                for band in range(bands):
                    out_band = out_ds.GetRasterBand(band + 1)
                    out_band.WriteArray(data[band] if bands > 1 else data)
                    out_band.FlushCache()

                # 设置地理参考和投影信息
                out_geo_transform = list(geotransform)
                out_geo_transform[0] += x_offset * geotransform[1]
                out_geo_transform[3] += y_offset * geotransform[5]

                out_ds.SetGeoTransform(out_geo_transform)
                out_ds.SetProjection(projection)

                del out_ds

                # 保存mask数据
                mask_opt_dir = os.path.join(output_dir, 'mask')
                if not os.path.exists(mask_opt_dir):
                    os.makedirs(mask_opt_dir)
                mask_output_file = os.path.join(mask_opt_dir, f"{i}_{j}_mask.tif")
                mask_driver = gdal.GetDriverByName("GTiff")
                mask_out_ds = mask_driver.Create(mask_output_file, read_width, read_height, 1,
                                                 mask_ds.GetRasterBand(1).DataType)
                mask_out_band = mask_out_ds.GetRasterBand(1)
                mask_out_band.WriteArray(mask_data)
                mask_out_band.FlushCache()

                mask_out_ds.SetGeoTransform(out_geo_transform)
                mask_out_ds.SetProjection(projection)

                del mask_out_ds

                # 保存index数据
                index_opt_dir = os.path.join(output_dir, 'index')
                if not os.path.exists(index_opt_dir):
                    os.makedirs(index_opt_dir)
                index_output_file = os.path.join(index_opt_dir, f"{i}_{j}_index.tif")
                index_driver = gdal.GetDriverByName("GTiff")
                index_out_ds = index_driver.Create(index_output_file, read_width, read_height, 1,
                                                   index_ds.GetRasterBand(1).DataType)
                index_out_band = index_out_ds.GetRasterBand(1)
                index_out_band.WriteArray(index_data)
                index_out_band.FlushCache()

                index_out_ds.SetGeoTransform(out_geo_transform)
                index_out_ds.SetProjection(projection)

                del index_out_ds

    del ds
    del mask_ds
    del index_ds


multi_raster = r'G:\藻\MSI\ECS\train\0505SVT\S2B_MSI_2022_05_05_02_38_17_T51SVT_Rrc_GEO.tif'
index_raster = r'G:\藻\MSI\ECS\train\0505SVT\index\S2B_MSI_2022_05_05_02_38_17_T51SVT_Rrc_GEO_AFAI.tif'
mask_raster = r'G:\藻\MSI\ECS\train\0505SVT\class\S2B_MSI_2022_05_05_02_38_17_T51SVT_Rrc_GEO_algae.tif'
output_dir = r'G:\藻\MSI\ECS\train\0505SVT'

create_tile(multi_raster, mask_raster, index_raster, output_dir)
