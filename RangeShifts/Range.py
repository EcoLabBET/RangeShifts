from . import RangeGeospatial

import geopandas as gpd
import numpy as np

import shapely
import pyproj

class Range:
    def __init__(self, gdf):
        """
        Initialize the Range object with a GeoDataFrame.

        Parameters:
        gdf (geopandas.geodataframe.GeoDataFrame): A GeoDataFrame containing spatial data.

        Attributes:
        _original_crs (pyproj.crs.CRS): The original CRS of the GeoDataFrame.
        _equal_area_crs (pyproj.crs.CRS): An equal-area projection CRS for calculations.
        name (str): The name of the Range (optional).
        crs (pyproj.crs.CRS): The current CRS of the Range.
        Range (geopandas.geoseries.GeoSeries): A GeoSeries containing the range geometries.
        bbox (numpy.ndarray): The bounding box of the Range.
        points (geopandas.geoseries.GeoSeries): Centroid points of the range geometries.
        convex_hull (shapely.geometry.Polygon): The convex hull of the range geometries.
        _concave_hull (shapely.geometry.Polygon): The concave hull of the range geometries (calculated upon request).
        centroid (shapely.geometry.Point): The centroid of the range geometries.
        _geometric_median (shapely.geometry.Point): The geometric median of the range geometries (calculated upon request).
        area (numpy.ndarray): An array of areas of the range geometries.
        density (float): The density of the range (calculated upon request).
        Abundance (int): The number of geometries in the Range.

        """
        self._original_crs = gdf.crs
        self._equal_area_crs = pyproj.CRS.from_proj4("+proj=eck4 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
        self.name = None

        # Equal Area Projection (to facilitate calculations )
        gdf = gdf.to_crs(self._equal_area_crs)
        self.crs = gdf.crs

        # crs dependent attributes (they contain geometry)
        self.Range = gdf['geometry'].reset_index(drop=True)
        self.bbox = gdf.total_bounds
        self.points = shapely.centroid(self.Range)
        self.points.crs = self._equal_area_crs # self.points is a GeoSeries and needs to have a crs

        self.convex_hull = gdf['geometry'].unary_union.convex_hull
        self._concave_hull = shapely.Polygon()

        ## Derived from other attributes (require calculation) --|
        self.centroid = shapely.centroid(shapely.MultiPoint(self.points))
        self._geometric_median = shapely.Point()

        # crs independent attributes
        self.area = np.vstack([item.area for item in gdf['geometry']])
        self.density = None
        self.Abundance = len(self.Range)

    @property
    def concave_hull(self):
        """
        Property that returns the concave hull of the range geometries.

        Returns:
        shapely.geometry.Polygon: The concave hull.
        """
        if self._concave_hull.is_empty:

          # Handle Unequal Area
          if self.crs != self._equal_area_crs:
            reprojection = pyproj.Transformer.from_crs(self.crs, self._equal_area_crs, always_xy=True).transform
            range = shapely.ops.transform(reprojection,self.Range)

            geoseries = gpd.GeoSeries(range,crs=self._equal_area_crs)
            self._concave_hull = calc_concave_hull(geoseries)

            reprojection = pyproj.Transformer.from_crs(self._equal_area_crs, self.crs, always_xy=True).transform
            self._concave_hull = shapely.ops.transform(reprojection,self._concave_hull)

          else:
            range = self.Range

            geoseries = gpd.GeoSeries(range,crs=self._equal_area_crs)
            self._concave_hull = calc_concave_hull(geoseries)
        return self._concave_hull

    @property
    def geometric_median(self):
        """
        Property that returns the geometric median of the range geometries.

        Returns:
        shapely.geometry.Point: The geometric median.
        """
        if self._geometric_median.is_empty:

          # Handle Unequal Area
          if self.crs != self._equal_area_crs:
            reprojection = pyproj.Transformer.from_crs(self.crs, self._equal_area_crs, always_xy=True).transform
            points = shapely.ops.transform(reprojection,self.points)

            points = np.vstack([np.array(point.coords) for point in points])
            self._geometric_median = Weiszfield_geom_median(points)

            reprojection = pyproj.Transformer.from_crs(self._equal_area_crs, self.crs, always_xy=True).transform
            self._geometric_median = shapely.ops.transform(reprojection,self._geometric_median)

          else:
            points = self.points_array()

            self._geometric_median = Weiszfield_geom_median(points)
        return self._geometric_median


    def calc_density(self, shapefile_path):
        '''
        Calculate the density of the range based on the area of occurrence and the area of intersection with coastline.

        Parameters:
        shapefile_path (str): The path to a shapefile for coastline data.

        Returns:
        float: The density of the range.

        Note:
        density = [Area of Occurence]/[Area of intersection of Convex_Hull with Coastline]

        This definition might be problematic, because it is dependent on the geographical
        entity chosen for intersection
        '''
        if self.density is None:

          # Handle Unequal Area
          if self.crs != self._equal_area_crs:
            reprojection = pyproj.Transformer.from_crs(self.crs, self._equal_area_crs, always_xy=True).transform
            hull = shapely.ops.transform(reprojection,self.convex_hull)
          else:
            hull = self.convex_hull

          # Calculate Density
          coastline = gpd.read_file(shapefile_path).to_crs(self._equal_area_crs)
          coastline = coastline.unary_union
          intersection = shapely.intersection(hull,coastline)

          self.density = np.sum(self.area)/intersection.area

        return self.density

    def calc_edge_geometric_median(self, edge, a):
        '''
        Calculate the geometric median based on the percentage of grid-cells of the specified edge.

        Parameters:
        edge (str): 'N' for North, 'S' for South, 'E' for East, or 'W' for West.
        a (float): A value between 0 and 1 to determine the percentage of grid-cells to consider.

        Returns:
        shapely.geometry.Point: The geometric median based on the edge and percentage.

        Future Improvements:
        - Internally it can use self.reproject() to handle the .crs
        '''
        assert self.crs == self._equal_area_crs, 'Such calculations are performed only in equal area projection. Please change the projection before calling Range.calc_edge_geometric_median()'

        if not (0 <= a <= 1):
            raise ValueError("Parameter 'a' must be between 0 and 1.")
        top = True if edge == 'N' or edge == 'E' else False
        coord = 1 if edge == 'N' or edge == 'S' else 0

        #sort descendingly
        xy_array = self.points_array()
        xy_array = xy_array[xy_array[:, coord].argsort()[::-1]]
        split_index = int(np.ceil(len(xy_array)*a))

        xy_array = xy_array[:split_index] if top else xy_array[-split_index:]

        #calc representative point
        return Weiszfield_geom_median(xy_array)

    def points_array(self):
        '''
        Convert the points to a numpy array.

        Returns:
        numpy.ndarray: A numpy array containing points.
        '''
        points_array = np.vstack([np.array(point.coords) for point in self.points])
        return points_array

    def reproject(self, target_crs):
        '''
        Reproject the Range object to the given CRS, affecting only CRS-dependent attributes.

        Parameters:
        target_crs (pyproj.crs.CRS): The target CRS for reprojection.
        '''
        source_crs = self.crs

        ## By utilising pyproj and shapely.ops.transform() directly
        reprojection = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True).transform

        self.convex_hull = shapely.ops.transform(reprojection,self.convex_hull)
        self._concave_hull = shapely.ops.transform(reprojection,self._concave_hull)
        self.centroid = shapely.ops.transform(reprojection,self.centroid)
        self._geometric_median = shapely.ops.transform(reprojection,self._geometric_median)

        ## By using the geoseries
        self.Range = self.Range.to_crs(target_crs)
        self.points = self.points.to_crs(target_crs)
        self.bbox = self.Range.total_bounds

        ## Update crs
        self.crs = target_crs
